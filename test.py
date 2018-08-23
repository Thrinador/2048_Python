import gym
import gym_python_2048

env = gym.make('python_2048-v0')
highest_score = 0
# env.start_render()
for i_episode in range(100):
    observation = env.reset()
    for t in range(10000):
        # env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            # print("Episode finished after {} timesteps".format(t+1))

            if reward > highest_score:
                highest_score = reward
            break

print("Highest Tile: ", highest_score)
