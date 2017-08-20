from gym.envs.registration import register

register(
    id='Watten-v0',
    entry_point='gym_watten.envs:WattenEnv',
)