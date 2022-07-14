from gym.envs.registration import register

register(
    id='gym_dt/DecisionTreeEnv',
    entry_point='gym_dt.envs:DecisionTreeEnv',
    max_episode_steps=500,
)
