# Just disables the warning, doesn't enable AVX/FMA
import os
from statistics import median, mean

import gym
import gym_python_2048
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
env = gym.make('python_2048-v0')


def initial_test_function():
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
                if reward > highest_score:
                    highest_score = reward
                break

    print("Highest Tile: ", highest_score)


LR = .01
goal_steps = 10000
score_requirement = 20
initial_games = 2000


def initial_population():
    training_data = []  # [OBS, MOVES]
    scores = []  # all scores:
    accepted_scores = []  # just the scores that met our threshold:
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                numpy_observation = np.array(prev_observation[0]).reshape(16)
                numpy_observation = np.append(
                    numpy_observation, prev_observation[1])
                game_memory.append([numpy_observation, action])

            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 0:
                    output = [1, 0, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1, 0]
                elif data[1] == 3:
                    output = [0, 0, 0, 1]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    return training_data


def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 16, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 32, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 32, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 16, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 4, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(
        -1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]
    print(len(X))
    print(len(y))
    print(len(training_data[0]))
    print(training_data[0])
    print(len(X[0]))
    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500)
    return model


training_data = initial_population()
model = train_model(training_data)

env.start_render()
scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(1000):
        env.render()

        if len(prev_obs) == 0:
            action = env.action_space.sample()
        else:
            action = np.argmax(
                model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

        choices.append(action)
        print(action)
        new_observation, reward, done, info = env.step(action)

        new_observation = np.array(new_observation[0]).reshape(16)
        new_observation = np.append(new_observation, new_observation[1])

        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward

        if done:
            break

    scores.append(score)

print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),
                                        choices.count(0)/len(choices)))
print(score_requirement)
