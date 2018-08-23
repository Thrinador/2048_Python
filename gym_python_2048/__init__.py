from gym.envs.registration import register

register(
    id='python_2048-v0',
    entry_point='gym_python_2048.envs:Python_2048Env',
)
