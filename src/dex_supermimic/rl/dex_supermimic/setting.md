# Dex SuperMimic 网络与输入 / 输出布局说明

本文对应当前 Hand 2 配置。若资产数量、actuator 数量、actuator 参数长度或 context 配置变化，具体维度会随之变化。每个训练 run 会写出权威布局：

```text
input_layout.json
context.json
```

主要代码位置：

- 网络定义：`networks.py`
- task frame / dynamics / critic 重排：`observations.py`
- simulator context、contact / termination 标签：`sim.py`
- 物理状态、delta、SO(3) 积分：`state.py`
- 网络组装、递归 rollout、训练更新：`runner.py`
- 仿真状态快照（实验评估）：`snapshot.py`
- 仿真端 observation / reward / command 定义：`../dex_reorient/`

---

## 0. 总体设计

当前 Dex SuperMimic 采用 SuperMimic 式的 **WM-as-simulator** 结构：

```text
WM
 └── 只预测物理状态与 contact
      ├── x1 / S_hat
      ├── contact logits
      └── 外部扰动 push 后得到下一步 x

Analytic model
 ├── tau
 ├── reward
 ├── cage counter
 ├── command / controller memory
 ├── policy observation history
 ├── WM dynamics features
 ├── critic privileged context
 └── FailureHead privileged features

Startup simulator reader
 └── 只在 rollout 开始时读取 startup context：friction、Kp/Kd ratios、
     default gains、force limit、gear、encoder bias 等

Exogenous replay / sampling
 └── 只处理无法由物理状态推出的量：action/u、measured or sampled push、
     sampled next goal
```

换句话说：

> task memory 不再 teacher-force 给 WM；
> `hold/window/cage/progress/goal/controller` 全部由 WM 输出的 `x1`
> 经过可微 FK 和解析状态机递归推导。

这使 dex 的递归关系与 G1 SuperMimic 对齐：

```text
physics:      closed-loop from WM output
task state:   analytic recursion from WM output
critic:       reads recursively derived privileged features
failure:      reads recursively derived privileged features
```

---

## 1. 四个网络总览

| 网络 | 输入维度 | 输出维度 | 主干 | 输出含义 |
|---|---:|---:|---|---|
| Actor | 207 | 20 | `512 → 256 → 128 → 20`，ELU | 20 个 normalized raw joint action，最后 `tanh` |
| World Model / WM | 213 | 84 | `2048 → 2048 → 2048 → 84`，SiLU + LayerNorm | 52 维物理 delta + 32 维 contact logits |
| FailureHead | 326 | 1 | `256 → 128 → 1`，ELU | 进入该真实/想象状态时发生真实 termination 的 logit |
| Critic | 268 | 1 | `512 → 256 → 128 → 1`，ELU | 当前 privileged task state 的价值 `V(s)` |

输入帧：

- Actor 使用 policy observation 的 **3 帧 history**，每项 flatten 后拼接。
- WM 与 FailureHead 都没有 history。WM 输入 transition 前的单帧 dynamics state 与
  action / disturbance，预测 transition 后的 `x1`；FailureHead 输入对应的
  **post-transition** `x1 / S_hat` 单帧 derived privileged features，预测该 transition
  是否发生真实 termination。
- Reorientation Critic 与 PPO 一样使用 3 帧 task term history，再附加固定仿真上下文；真实历史来自环境，想象历史由预测状态递推。下文保留的旧单帧维度示例请以 `setting_hand1_reorient.md` / `setting_hand2_reorient.md` 与运行时 `input_layout.json` 为准。
- Actor 不使用 simulator privileged context。
- WM、FailureHead、Critic 使用 startup simulator privileged context。

---

## 2. Actor 输入：207 维

Actor 输入直接来自 `env.reset() / env.step()["policy"]`，也就是 policy observation group。它不是 critic context。

### 2.1 当前帧 policy observation：69 维

| 序号 | term | 维度 | 具体内容 |
|---:|---|---:|---|
| 1 | `noisy_joint_angles` | 20 | 20 个手指关节位置，按 soft joint limit 归一化到 `[-1,1]`，叠加 uniform 噪声 |
| 2 | `qpos_error` | 20 | 当前 normalized joint position 减去当前 normalized EMA target |
| 3 | `cube_pos_in_tag` | 3 | cube root 位置在 wrist tag frame 下的 3D 位置 |
| 4 | `cube_ori_error` | 6 | tag frame 下 cube 相对 goal 的旋转误差矩阵后两行展平 |
| 5 | `action_history` | 20 | 上一个 control step 的 20 维 raw action |
| | **合计** | **69** | |

噪声 / outlier：

- joint angle：uniform `[-0.06, +0.06]`。
- cube position：2% outlier `[-0.5,0.5]^3`，另加 uniform `[-0.008,+0.008]`。
- cube orientation error：2% 随机姿态 outlier，另加 `N(0,0.05^2)`。
- `qpos_error`、`action_history` 不额外加配置端噪声。

### 2.2 三帧 history：69 × 3 = 207

```text
Actor input = concat(
    noisy_joint_angles history 3 × 20,
    qpos_error            history 3 × 20,
    cube_pos_in_tag       history 3 × 3,
    cube_ori_error        history 3 × 6,
    action_history        history 3 × 20,
) = 60 + 60 + 9 + 18 + 60 = 207
```

### 2.3 Actor 结构

```text
RunningNorm(207)
Linear(207 → 512) + ELU
Linear(512 → 256) + ELU
Linear(256 → 128) + ELU
Linear(128 → 20)
tanh
```

输出层初始化为 `orthogonal(gain=0.01)`，bias 为 0。

训练采样时：

```text
action = clamp(actor(obs) + 0.15 * N(0, I), -1, 1)
```

imagination 阶段直接使用 actor 输出，不加 exploration noise。

### 2.4 Actor action 到 WM 的 `u`

WM 不使用 actor raw action，而使用实际传给 actuator 的 joint position target：

```text
raw = default + 0.5 * clamp(action, -1, 1)
raw = clamp(raw, soft_lower, soft_upper)
target_c = alpha * raw + (1 - alpha) * previous_c1
warmup 前 target_c = default
u = target_c - encoder_bias
```

因此 `u` 已经包含：

- action limit；
- EMA smoothing；
- warmup；
- startup encoder bias correction。

`u` 是 WM 唯一接收的控制输入。

---

## 3. WM：只做物理动力学 + contact

### 3.1 WM 输入：213 维

```python
WorldModel.inputs(dynamics, u):
    concat(dynamics, u)
```

布局：

```text
[ dynamics 193 | actual control target u 20 ]
```

`dynamics` 由 `TaskObservations.dynamics()` 构造：

```text
[ q 20
| qd 20
| cube position 3
| cube rotation 9
| cube linear velocity 3
| cube angular velocity 3
| simulator context 106
| disturbance cache 9 ]
```

即：

```text
dynamics = physical state 58 + simulator context 106 + disturbance 9
WM input = dynamics 193 + u 20 = 213
```

#### WM 不接收的量

以下 task/command/controller 信息全部不进 WM：

```text
goal
hold / goal_timer / window / window_timer
reward_window / reward_hold / reward_snapshot
within
cage counter
age
count
c1 / c2 / a1 / a2
state_progress
critic-only task terms
```

原因：

1. 它们不直接改变物理转移，物理只由 `x_t, u_t, xi, disturbance` 决定。
2. 在 imagination 中它们由 `x1` 和递归状态机得到。
3. 若输入给 WM，会造成 task context teacher forcing 和 state inconsistency。

#### disturbance cache：9 维

来自 auxiliary 的：

```text
[ pert_dir 3 | pert_velocity 6 ]
```

它表示最近一次 interval disturbance 的方向和速度缓存。这是外部事件状态，不能由物理 `x` 推出，因此保留为 dynamics 条件。

supervised WM training 时由 replay 的 measured push 重建；
policy imagination 时按任务分布采样并递归更新。

---

### 3.2 simulator context：106 维

`SimReader.context()` 从 startup 随机化后的 MuJoCo model 读取一次。它是 WM、FailureHead、Critic 共享的 privileged startup context。

#### 3.2.1 friction：66 维

```text
66 = 22 active collision geoms × 3 friction components
```

只包含 robot / object 中 `geom_contype` 或 `geom_conaffinity` 非零的 active collision geom。

过滤掉：

- visual-only geom；
- collision overlay 禁用的 `*_mesh_disabled`；
- calibration rig table；
- non-colliding mount；
- `cube_visual`。

#### 3.2.2 actuator context：40 维

```text
20 Kp ratios + 20 Kd ratios
```

- `Kp_ratio = current gainprm[...,0] / default gainprm[...,0]`
- `Kd_ratio = current biasprm[...,2] / default biasprm[...,2]`

`biasprm[...,2]` 为负数，分母保持负号，避免近零时翻转。

完整 `gainprm`、`biasprm`、`forcerange`、`gear` 数组不进入 context；它们 startup 读取一次，用于解析 torque。

### 3.3 WM 结构

```text
RunningNorm(213)
Linear(213 → 2048) + LayerNorm + SiLU
Linear(2048 → 2048) + LayerNorm + SiLU
Linear(2048 → 2048) + LayerNorm + SiLU
Linear(2048 → 84)
```

输出层 weight 和 bias 全部初始化为 0。CUDA 训练时主干默认 bfloat16 autocast，输出转回 float32。

### 3.4 WM 输出：84 维

```python
d1, contact_logits = y.split((52, 32), -1)
```

| 序号 | 输出 | 维度 | 含义 |
|---:|---|---:|---|
| 1 | `d1` | 52 | `x_t → x_{t+1}` 的 frame-local physical delta |
| 2 | `contact_logits` | 32 | reward-phase contact 二分类 logits |
| | **合计** | **84** | |

WM 不再输出：

```text
tau
reward-phase lag state
state_progress
cage counter
goal / command state
```

### 3.5 state delta：52 维

physical state `x` 为 58 维，旋转使用 9 维矩阵；delta 中旋转增量用 3 维旋转向量表示。

| 序号 | delta 分量 | 维度 | 积分方式 |
|---:|---|---:|---|
| 1 | joint position delta | 20 | `q ← q + Δq` |
| 2 | joint velocity delta | 20 | `qd ← qd + Δqd` |
| 3 | cube position delta | 3 | `p ← p + Δp` |
| 4 | cube orientation delta | 3 | `R ← R · exp(Δθ)`，SO(3) 更新 |
| 5 | cube linear velocity delta | 3 | `v ← v + Δv` |
| 6 | cube angular velocity delta | 3 | `w ← w + Δw` |
| | **合计** | **52** | |

反归一化：

```text
delta = raw_delta * delta_norm.std() + delta_norm.mean
```

`delta_norm` 只从真实 transition 的 `difference(x_t, x_{t+1})` 更新。

### 3.6 contact logits：32 维

标签来自 `SimReader.contact()`，固定布局：

```text
[ fingertip contacts 5 | palm/proximal 1 | distal 1 | finger self-collision 25 ]
```

| index | 维度 | 含义 |
|---|---:|---|
| `0:5` | 5 | 五个 fingertip 是否接触 cube |
| `5:6` | 1 | palm / proximal finger 区域是否接触 cube |
| `6:7` | 1 | distal finger 区域是否接触 cube |
| `7:32` | 25 | finger self-collision sensor 的 25 个 primary geom slot |

reward 使用 `sigmoid(contact_logits)`。

### 3.7 WM 训练监督信号

每个 open-loop step 计算：

```text
wm_loss =
    final_state_weight * mean((state_error / error_std) ** 2)
  + contact_weight      * sum(BCEWithLogits(contact_logits, true_contact))
```

默认权重：

```text
final_state_weight = 1.0
contact_weight     = 0.1
```

`error_std` 使用真实采集状态增量的运行标准差，并逐维设置物理误差下限。
最终 loss 对有效时间步求和、除以窗口数量，与 SuperMimic 一致；窗口固定为 8 步。

---

## 4. 解析 `tau`：20 维

`tau` 不再由 WM 输出，而是从 pre-substep state、startup actuator 参数和 `u` 解析计算：

```text
Kp = default_Kp * Kp_ratio
Kd = default_Kd * Kd_ratio

tau = gear * (Kp * (u - q) + Kd * qd)
tau = clamp(tau, -force_limit, force_limit)
```

startup 读取一次：

```text
default Kp
default Kd
force limit
gear
```

reward 若存在 effort limit，则：

```text
normalized_torque = tau / effort_limit
```

并进入 torque penalty。

说明：MuJoCo 在 physics substep 内先求 actuator force，再积分状态。reward snapshot 捕获的是 post-step state，因此 analytic torque 使用 **pre-substep rollout state**，与 MuJoCo actuator law 的求值时序一致。

---

## 5. 递归 task state

WM 输出 `x1` 后，所有 task/controller 状态由 analytic model 递归推进：

### 5.1 controller memory

```text
c2 ← c1
c1 ← target_c
a2 ← a1
a1 ← action
age ← age + 1
```

这些用于 reward、policy observation 和下一步 controller，但不进入 WM。

### 5.2 cage counter

```text
q, qd from x1
        ↓
differentiable FK
        ↓
palm / finger cage AABB
        ↓
check x1.p inside / outside
        ↓
outside: counter + 1
inside:  counter = max(counter - 0.5, 0)
```

用于：

```text
cage_escape reward
analytic cage termination
critic / FailureHead privileged features
```

### 5.3 command state

由 `DexMDP.command_step(x1, memory, next_goal)` 推进：

```text
within
hold
just succeeded
window
window_timer
reward_window
reward_hold
reward_snapshot
goal switch
count
```

其中：

- `within` 由 `x1` 的 cube orientation 与当前 goal 的 angle error 得到。
- goal switch 时新 goal 是外生随机量，supervised replay 中保持当前 goal，imagination 中按 SO(3) 均匀采样。
- `goal` 不进入 WM，只进入 task state / reward / critic / actor observation。

### 5.4 policy observation history

下一步 clean policy observation 由：

```text
x1 + updated memory
        ↓
DexMDP.observation()
        ↓
noisy_observation()
        ↓
append to 3-frame history
```

actor 不读取 simulator observation history。

### 5.5 disturbance cache / push

外部扰动是外生事件，不由物理状态推出。

supervised WM training：

```text
x_physics_next = WM output x1
x_next = x1 + measured replay push
disturbance cache = advance_disturbance(measured push)
```

policy imagination：

```text
sample event / speed / direction
x_next = x1 + sampled push
disturbance cache = advance_disturbance(sampled event)
```

push 只作为外生 transition 条件，不作为 startup simulator context。

---

## 6. Critic / FailureHead 共享 privileged context：268 维

`TaskObservations.context()` 构造：

```text
context = concat(simulator_context, task_frame)
         = concat(106, 162)
         = 268
```

WM 不使用这个 context；它只用于 Critic 和 FailureHead。

### 6.1 task frame：162 维

```text
task_frame = base terms 124 + current privileged terms 38
```

它对应原任务 critic observation 的 clean 当前帧，可由真实 state 或 WM 预测 state 生成同一套 features。

#### 6.1.1 base terms：124 维

| 序号 | term | 维度 | 具体内容 |
|---:|---|---:|---|
| 1 | `joint_angles` | 20 | clean joint position，soft-limit 归一化并 clamp 到 `[-1,1]` |
| 2 | `qpos_error` | 20 | normalized joint position 减 normalized EMA target `c1` |
| 3 | `cube_pos_in_tag` | 3 | clean cube root 位置在 tag frame 下 |
| 4 | `cube_ori_error` | 6 | tag frame 下 cube 相对 goal 的误差矩阵后两行 |
| 5 | `action_history` | 20 | 上一个 raw action，对应 `a2` |
| 6 | `true_joint_pos` | 20 | clean joint position 减 default |
| 7 | `true_joint_vel` | 20 | clean joint velocity 减 default velocity |
| 8 | `fingertip_positions` | 15 | 五个 fingertip body 位置，世界系减 env origin 和 `(0,0,0.5)`，乘 scale `5.0` |
| | **合计** | **124** | |

`true_joint_pos/vel` 与 `fingertip_positions` 在 imagination 中分别来自 WM 预测的 `q/qd` 和 differentiable FK。

#### 6.1.2 current privileged terms：38 维

| 序号 | term | 维度 | 具体内容 |
|---:|---|---:|---|
| 9 | `cube_pos_in_tag_clean` | 3 | clean cube 位置 |
| 10 | `cube_ori_error_clean` | 6 | clean cube-to-goal 误差 6D |
| 11 | `cube_linvel` | 3 | cube linear velocity，palm frame 表达 |
| 12 | `cube_angvel` | 3 | cube angular velocity，palm frame 表达 |
| 13 | `pert_dir` | 3 | 最近 disturbance 方向 |
| 14 | `pert_velocity` | 6 | 最近 disturbance 的 linear/angular velocity 缓存 |
| 15 | `state_progress` | 4 | hold/window/episode progress 和 goal reach count |
| 16 | `cage_counter` | 1 | cage counter / `max_outside_steps`，clamp 最大 1 |
| 17 | `dr_params` | 3 | object mass ratio、joint damping ratio、object size ratio |
| 18 | `palm_rot_6d` | 6 | palm/root 世界姿态旋转矩阵后两行 |
| | **合计** | **38** | |

`state_progress` 定义：

```text
hold progress    = hold_counter / success_hold_steps
window progress  = reward_window_timer / goal_switch_delay
episode progress = episode_length / max_episode_length
goal reach count = 已成功达到 goal 的次数
```

这些量不进 WM，但 critic 保留；它们由 WM 输出的 `x1` 递归推导。

---

## 7. Critic

### 7.1 输入重排

Critic 使用同一个 268 维 context，但重排为：

```python
critic(context):
    concat(context[..., -162:], context[..., :-162])
```

即：

```text
WM无关 context = [ simulator_context 106 | task_frame 162 ]

Critic input   = [ task_frame 162 | simulator_context 106 ]
```

### 7.2 结构

```text
RunningNorm(268)
Linear(268 → 512) + ELU
Linear(512 → 256) + ELU
Linear(256 → 128) + ELU
Linear(128 → 1)
```

### 7.3 输出与目标

输出：

```text
V(s_t)
```

λ-return 更新：

```text
gamma         = 0.996
lam           = 0.95
critic_epochs = 3
```

递推：

```text
ret_H = V(s_H)

boot_t = (1-λ)V(s_{t+1}) + λ ret_{t+1}
ret_t  = r_t + γ continue_t boot_t
```

只回归 `t=0..H-1`；`s_H` 只用于 bootstrap。

---

## 8. FailureHead

FailureHead 是 SuperMimic 式 learned real-termination head。它不从 simulator 读取 imagination 状态，而是使用同一条递归 privileged feature 链。
它与 WM 一样没有 history，但看的不是 transition 前的 `x_t`，而是 transition 后的
`x1 / S_hat` 及由该状态递归得到的 task memory。

### 8.1 输入：326 维

```python
features = concat(x1, context)
```

| 序号 | 输入 | 维度 | 含义 |
|---:|---|---:|---|
| 1 | `x1` / `S_hat` | 58 | transition 后真实 state 或 WM 预测 state |
| 2 | `context` | 268 | 从同一个 `x1` 和递归 task memory 构造的 privileged task/critic context |
| | **合计** | **326** | |

### 8.2 结构

```text
RunningNorm(326)
Linear(326 → 256) + ELU
Linear(256 → 128) + ELU
Linear(128 → 1)
```

### 8.3 语义

```text
sigmoid(output) = P(real simulator terminates on entering this state)
```

标签是：

```text
terminated
```

不是 timeout，不是 truncated，不是 NaN guard。

### 8.4 训练

训练同时使用两个分支：

#### true-state branch

```text
x1_true = replay["x1"]
context_true = analytic recursive context from x1_true
features_true = FailureHead.features(x1_true, context_true)
```

#### predicted-state branch

```text
S_hat = WM rollout output
context_hat = analytic recursive context from S_hat
features_hat = FailureHead.features(S_hat, context_hat)
```

loss：

```text
failure_loss =
    weighted BCE(logits_true, terminated)
  + weighted BCE(logits_hat,  terminated)
```

权重：

```text
replay mask × inverse sampling importance
```

normalization 只从 true-state branch 的 finite training transition 更新；held-out env 不参与。

### 8.5 imagination 中的使用

```text
d_analytic = soft cage termination

p_fail = sigmoid(FailureHead(features(S_hat, recursive_context)))
p_fail = clamp(p_fail, max=failure_clip)

d = 1 - (1 - d_analytic) * (1 - p_fail)
```

这与 SuperMimic 的 continue-head 使用方式一致：

```text
learned failure OR analytic termination
```

默认：

```text
failure_head           = true
failure_warmup_iters   = 300
failure_clip           = 0.1
```

关闭：

```text
--no-failure-head
```

关闭后 imagination 只使用 analytic soft termination。

---

## 9. WM supervised rollout：完全递归

`fit_world_model()` 对每个 sampled window 只读取起点：

```text
x0
memory0
auxiliary0
startup simulator context
```

后续每一步：

```text
dynamics_t = TaskObservations.dynamics(x_t, xi, disturbance_t)
x1, contact = WM(x_t, u_t, dynamics_t)

tau = analytic_pd_torque(x_t, u_t, xi)
cage_next = analytic_cage(x1, memory_t)
memory_next = command_step(x1, memory_t, current_goal)
memory_next.controller_fields = update(action, target)

x_next = x1 + measured_push_t
disturbance_next = advance_disturbance(measured_push_t)

t ← t+1
```

允许 teacher-force 的只有：

```text
u_t / action / EMA target
measured push
startup simulator context
```

不允许 teacher-force：

```text
x1
task memory
cage counter
progress
goal state
critic context
WM dynamics context
```

---

## 10. Policy imagination

每个 imagination step：

```text
context_t = TaskObservations.context(x_t, memory_t, aux_t, xi)
critic value at s_t

action = actor(flattened noisy history)
target, u = analytic EMA controller

dynamics = TaskObservations.dynamics(x, xi, disturbance)
x1, contact_logits = WM(x, u, dynamics)

tau = analytic_pd_torque(x, u, xi)
r, cage = analytic_reward(x1, memory, action, target, sigmoid(contact), tau)

d_analytic = soft cage termination
p_fail = FailureHead(features(x1, recursive_context_next))
d = 1 - (1-d_analytic)(1-p_fail)

memory_next = command_step(x1, memory, sampled_next_goal)
memory_next.controller_fields = update(action, target)

sample interval push
auxiliary_next = advance_disturbance(...)

x_next = x1 + push
history_next = append(noisy_observation(x_next, memory_next))
```

actor optimization 期间冻结：

```text
WM
FailureHead
Critic
```

但保留它们输入的可微梯度路径。

---

## 11. 标签与 replay

collect 时保存：

```text
x                         pre-transition physical state
x1                        post-physics state used by WM/reward
push                      next_x - x1，外部扰动修正
u                         actual actuator target
c                         EMA target before encoder bias
action                    raw actor action
m                         controller/command/cage memory
obs                       policy observation
aux                       palm pose、disturbance cache、DR ratios
contact                   32 维 contact label
torque                    MuJoCo actuator force，用于 parity 诊断
reward                    simulator reward
terminated                real termination label，FailureHead 使用
done                      terminated | truncated，用于 replay mask
valid                     finite/NaN mask
```

replay window 不跨越 episode boundary；invalid transition 会截断后续 open-loop prediction。

FailureHead 标签只使用 `terminated`，timeout/NaN 不作为 failure label。

---

## 12. 维度速查表

```text
Policy current frame         69
Actor input                  207 = 69 × 3 history

Physical state x              58 = 20 q + 20 qd + 3 p + 9 R + 3 v + 3 w
Disturbance cache              9 = 3 direction + 6 velocity

Simulator friction context    66
Simulator actuator context    40 = 20 Kp ratios + 20 Kd ratios
Simulator context total      106

WM dynamics                  193 = 58 physical + 106 simulator + 9 disturbance
WM control target             20
WM input                     213 = 193 + 20

WM state delta                52
WM contact logits             32
WM output                     84 = 52 + 32

Task base terms              124
Task current/privileged       38
Task frame                   162 = 124 + 38

Critic context               268 = 162 task frame + 106 simulator context
Critic input                 268，重排为 [task_frame | simulator_context]
Critic output                  1

FailureHead input            326 = 58 state + 268 recursive privileged context
FailureHead output             1
```

---

## 13. 递归与外生信息总结

| 量 | WM 输入 | WM 输出 | analytic 递归 | Critic/Failure |
|---|---:|---:|---:|---:|
| `x` | ✅ dynamics | delta → `x1` | ✅ + push | ✅ |
| `u` | ✅ | ❌ | controller 生成 | ❌ |
| startup `xi` | ✅ | ❌ | startup 读一次 | ✅ |
| disturbance cache | ✅ dynamics | ❌ | replay/采样更新 | ✅ context |
| push | ❌ | ❌ | replay/采样 | ❌ |
| contact logits | ❌ | ✅ | reward 用 sigmoid | ❌ |
| analytic `tau` | ❌ | ❌ | ✅ | reward 使用 |
| `c1/c2/a1/a2` | ❌ | ❌ | ✅ | task/reward 用 |
| `goal` | ❌ | ❌ | ✅，switch 时外生采样 | ✅ |
| `hold/window/timers` | ❌ | ❌ | ✅ | ✅ |
| `within/count` | ❌ | ❌ | ✅ | ✅ |
| `cage counter` | ❌ | ❌ | ✅ FK + counter | ✅ |
| `age` | ❌ | ❌ | ✅ +1 | ✅ |
| `state_progress` | ❌ | ❌ | ✅ | ✅ |
| termination | ❌ | ❌ | analytic + FailureHead | FailureHead 输入 |

最终结构：

```text
WM: physics + contact
Analytic: torque, reward, cage, command, history, dynamics/critic features
Startup reader: friction, gains, force limits, gear, encoder bias
FailureHead: recursive privileged features → real termination
Critic: recursive privileged features → V(s)
```
