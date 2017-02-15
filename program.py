import gym
import random
import numpy
env = gym.make('FrozenLake-v0')

class RandomAgent(object):
    def __init__(self, action_space):
        self.qmatrix = [[random.uniform(0, 1) for x in range(4)] for y in range(16)]
        #self.qmatrix = [[0.5 for x in range(4)] for y in range(16)]
        self.lastState = 0
        self.lastAction = -1
        self.updateMatrix = True
        self.DF = 0.5
        self.LF = 0.1

    def act(self, state, reward, done):
        if done:
            self.qmatrix[state] = [reward for x in range(4)]
        if self.lastAction != -1 and self.updateMatrix:
            self.recalculate(state, reward)
        action = numpy.argmax(self.qmatrix[state])
        self.lastAction = action
        self.lastState = state
        return action

    def recalculate(self, state, reward):
        oldValue = self.qmatrix[self.lastState][self.lastAction];
        maxValue = numpy.amax(self.qmatrix[state]) 
        newValue = oldValue + self.LF * (reward + self.DF * maxValue - oldValue)
        self.qmatrix[self.lastState][self.lastAction] = newValue


def run(agent, updateMatrix = True, count = 10000):
    agent.updateMatrix = updateMatrix
    episode_count = count
    wins = 0
    for i_episode in range(episode_count):
        observation = env.reset()
        agent.lastAction = -1
        reward = 0
        done = False
        while True:
            ##env.render()
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)
            if done:
                agent.act(observation, reward, done)
                if reward == 1: wins += 1
                break
    print(wins, episode_count)
    result = float(wins)/episode_count
    print(result)
    # print(agent.qmatrix)

if __name__ == '__main__':
    agent = RandomAgent(env.action_space)
    run(agent)
    run(agent, False, 100)