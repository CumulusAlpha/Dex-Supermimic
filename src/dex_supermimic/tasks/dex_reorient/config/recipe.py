"""Original 20/50 Hz task recipes, independent of the reference repository."""


def configure_supermimic(cfg, num_envs=1024, control_hz=50):
    if control_hz not in (20, 50):
        raise ValueError("control_hz must be 20 or 50")
    cfg.scene.num_envs = num_envs
    # Match PPO: every critic term retains three frames.
    from dex_supermimic.tasks.dex_reorient.reorient_terms import HISTORY_LENGTH
    for term in cfg.observations["critic"].terms.values():
        term.history_length = HISTORY_LENGTH
    # These four sensors cover all active rewards/metrics. The original three
    # extra force sensors have no consumers in this task.
    required = {
        "tip_object_contact",
        "palm_object_found",
        "distal_finger_object_found",
        "finger_collision",
    }
    cfg.scene.sensors = tuple(sensor for sensor in cfg.scene.sensors if sensor.name in required)
    if control_hz == 50:
        cfg.decimation = 2
        cfg.actions["joint_pos"].ema_alpha = 0.3
        cfg.rewards["action_rate"].weight = 0.0
        cfg.rewards["action_hf"].weight = -30.0
        cfg.rewards["hold_escalation"].weight = 4.694
        cfg.commands["reorient_command"].success_hold_steps = 13
        cfg.commands["reorient_command"].goal_switch_delay = 50
        cfg.rewards["cage_escape"].params.update(max_outside_steps=25, max_count=37.5)
        cfg.terminations["cage_drop"].params["max_outside_steps"] = 25
        cfg.observations["critic"].terms["cage_counter"].params["max_outside_steps"] = 25
    return cfg


def configure_wuji_ppo(cfg, control_hz=50):
    """Match wuji-mjlab's registered PPO task configuration exactly.

    The PPO runner consumes the real observation groups directly.
    """
    if control_hz not in (20, 50):
        raise ValueError("control_hz must be 20 or 50")
    # Match the shared world-model recipe for critic history. Keep this repo's
    # four-sensor training set, which matches all active upstream consumers.
    from dex_supermimic.tasks.dex_reorient.reorient_terms import HISTORY_LENGTH
    for term in cfg.observations["critic"].terms.values():
        term.history_length = HISTORY_LENGTH
    if control_hz == 50:
        cfg.decimation = 2
        cfg.actions["joint_pos"].ema_alpha = 0.3
        cfg.rewards["action_rate"].weight = 0.0
        cfg.rewards["action_hf"].weight = -30.0
        cfg.rewards["hold_escalation"].weight = 4.694
        cfg.commands["reorient_command"].success_hold_steps = 13
        cfg.commands["reorient_command"].goal_switch_delay = 50
        cfg.rewards["cage_escape"].params.update(max_outside_steps=25, max_count=37.5)
        cfg.terminations["cage_drop"].params["max_outside_steps"] = 25
        cfg.observations["critic"].terms["cage_counter"].params["max_outside_steps"] = 25
    return cfg
