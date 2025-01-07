from gymnasium.envs.registration import register

register(
    id="frasa-standup-v0",
    entry_point="frasa_env.env:FRASAEnv",
)

register(
    id="t1-standup-v0",
    entry_point="frasa_env.env:T1StandupEnv",
)
