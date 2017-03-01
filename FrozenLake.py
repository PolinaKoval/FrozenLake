import gym
import random
import numpy
env = gym.make('FrozenLake-v0')

class QLearningAgent(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.qmatrix = [[0.1 for x in range(action_space.n)] for y in range(observation_space.n)]

        self.lastState = 0
        self.lastAction = -1
        self.updateMatrix = True
        self.DF = 0.99
        self.LF = 0.1

    def act(self, state, reward, done, episode):
        if self.lastAction != -1:
            self.recalculate(state, reward)
        action = self.getAction(state, episode)
        self.lastAction = action
        self.lastState = state
        return action

    
    def getAction(self, state, t):
        maxAction = numpy.argmax(self.qmatrix[state])
        cond = random.uniform(0, 1) > float(100)/(t + 1)
        return maxAction if cond else random.randint(0, self.action_space.n - 1)

    def recalculate(self, state, reward):
        oldValue = self.qmatrix[self.lastState][self.lastAction];
        maxValue = numpy.amax(self.qmatrix[state]) 
        newValue = oldValue + self.LF * (reward  + self.DF * maxValue - oldValue)
        self.qmatrix[self.lastState][self.lastAction] = newValue

def run(agent, updateMatrix = True, count = 10000):
    agent.updateMatrix = updateMatrix
    episode_count = count
    wins = 0
    last100 = []
    averageScore = 0
    for i_episode in range(episode_count):
        observation = env.reset()
        agent.lastAction = -1
        reward = 0
        done = False
        while True:
            action = agent.act(observation, reward, done, i_episode)
            observation, reward, done, info = env.step(action)
            if done:
                last100.append(reward)
                if len(last100) > 100:
                    last100.pop(0)
                    averageScore = numpy.mean(last100)
                agent.act(observation, reward, done, i_episode)
                if reward == 1:
                    wins += 1
                break

        if (averageScore >= 0.78):
            print(averageScore, i_episode)
            break

    # print(wins, episode_count)
    # print(float(wins)/episode_count)
    # policy = map(numpy.argmax, agent.qmatrix)
    # print(policy)
    # print(agent.qmatrix)

if __name__ == '__main__':
    agent = QLearningAgent(env.action_space, env.observation_space)
    run(agent)
