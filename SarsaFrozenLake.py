import gym
import random
import numpy
env = gym.make('FrozenLake-v0')


class QLearningAgent(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.qmatrix = [[0 for _ in range(action_space.n)] for _ in range(observation_space.n)]
        self.e = numpy.zeros((self.observation_space.n, self.action_space.n))
        self.last_state = 0
        self.last_action = -1
        self.DF = 0.99
        self.LF = 0.1
        self.lam = 0.2

    def act(self, state, reward, done, episode):
        action = self.get_action(state, episode)
        if self.last_action != -1:
            self.recalculate(state, reward, action)
        self.last_action = action
        self.last_state = state
        if done:
            self.e = numpy.zeros((self.observation_space.n, self.action_space.n))
        return action
    
    def get_action(self, state, t):
        max_action = numpy.argmax(self.qmatrix[state])
        cond = random.uniform(0, 1) > float(100)/(t + 1)
        return max_action if cond else random.randint(0, self.action_space.n - 1)

    def recalculate(self, state, reward, next_action):
        old_value = self.qmatrix[self.last_state][self.last_action]
        next_value = self.qmatrix[state][next_action]
        delta = reward + self.DF * next_value - old_value
        self.e[self.last_state][self.last_action] += 1
        for state_ in range(self.observation_space.n):
            for action_ in range(self.action_space.n):
                self.qmatrix[state_][action_] += self.LF * self.e[state_][action_] * delta
                self.e[state_][action_] *= self.lam * self.DF


def run(agent, count=10000):
    episode_count = count
    wins = 0
    last100 = []
    average_score = 0
    i_episode = 0
    while True:
        i_episode += 1
        observation = env.reset()
        agent.last_action = -1
        reward = 0
        done = False
        while True:
            action = agent.act(observation, reward, done, i_episode)
            observation, reward, done, info = env.step(action)
            if done:
                last100.append(reward)
                if len(last100) > 100:
                    last100.pop(0)
                    average_score = numpy.mean(last100)
                agent.act(observation, reward, done, i_episode)
                if reward == 1:
                    wins += 1
                break

        if average_score >= 0.78:
            print(average_score, i_episode)
            break
    # print(wins, episode_count)
    # print(float(wins)/episode_count)
    # policy = map(numpy.argmax, agent.qmatrix)
    # print(policy)
    # print(agent.qmatrix)

if __name__ == '__main__':
    agent = QLearningAgent(env.action_space, env.observation_space)
    run(agent)
