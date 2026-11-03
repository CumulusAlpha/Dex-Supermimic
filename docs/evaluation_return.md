# 权重评估与 Return

在仓库根目录使用现有虚拟环境运行，无需重新训练或启动 viewer。

```bash
# SuperMimic
.venv/bin/python evaluate_return.py \
  --checkpoint logs/dex_supermimic/hand2-reorient/2026-11-01_11-03-28/model_final.pt \
  --output logs/evaluation/supermimic.json

# PPO：相同评估参数
.venv/bin/python evaluate_return.py \
  --checkpoint logs/dex_supermimic/hand2-reorient/2026-11-02_11-01-07/model_final.pt \
  --output logs/evaluation/ppo.json
```

默认参数：`--seeds 1001 1002 1003 --episodes 32 --num-envs 32 --window-seconds 40 --device cuda:0`。
`--episodes` 是每个 seed 的局数，默认总计 96 局。脚本输出 return 的均值、样本标准差、
最小值和最大值、平均存活时间、终止比例；JSON 还保存每局明细及每个 seed 的平均 return。
不传 `--output` 时保存到权重旁的 `model_final.evaluation.json`。

Return 定义为一局内 `env.step()` 返回奖励的未折扣总和，包含终止步奖励。
环境已经处理奖励权重和时间步缩放，脚本不再乘 dt，不加 critic bootstrap。
掉落等终止发生后即结束该局，不把 reset 后下一局奖励计入，也不补足剩余时间。
所有并行槽各统计一局后再开始下一批，避免短局在样本中占比过高。

从权重自动识别 SuperMimic/PPO，恢复 actor 和观测归一化；PPO 使用高斯分布的均值动作，
SuperMimic 使用 actor 输出。全程不更新模型、归一化统计或优化器；SuperMimic 的世界模型
不参与评估，动作进入真实 MuJoCo 仿真环境。

默认使用原训练任务配置（`play=False`），保留观测噪声、域随机化、扰动和终止条件，
关闭课程调度并使用固定最高扰动难度，每局最长 40 秒。可以用
`--window-seconds 50` 对齐原任务 50 秒的时间上限。该固定难度与训练日志中不断变化的
课程状态不同，测试 return 不应直接与训练曲线数字等同。

比较两个模型时使用相同 seeds、并行数、局数、时长和代码版本。随机数由环境批量共享，
相同 seed 不保证不同策略经历逐步相同的扰动（终止和 reset 时刻会改变随机数消耗）；
改变并行数也可能改变轨迹。这里只对 episode 统计标准差，不据此推断多训练 seed 的置信区间。
当前脚本支持 Hand 1/Hand 2 方块重定向任务。

参考 [Sharpa 的评估说明](https://github.com/Mingrui-Yu/sharpa_rl_unilab/blob/0363bdb8b94935acdb35c9c0fa4737a06bca99ed/docs/zh_CN/training.md)
和 [评估实现](https://github.com/Mingrui-Yu/sharpa_rl_unilab/blob/0363bdb8b94935acdb35c9c0fa4737a06bca99ed/src/sharpa_rl_unilab/training/evaluation.py)：
采用确定性策略、固定窗口、逐局累计回报、JSON 输出的方式。Sharpa 的八尺度任务、抓取缓存、
按槽隔离 RNG、权重格式均不同，不能直接运行 `sharpa-eval` 来评估本项目权重，也不能直接比较两种任务的 return 数值。
