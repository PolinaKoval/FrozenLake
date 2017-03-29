import gym
import random
import numpy
env = gym.make('FrozenLake-v0')


class SarsaAgent(object):
    def __init__(self, action_space, observation_space, DF, LF, lam, er, er_limit):
        self.action_space = action_space
        self.observation_space = observation_space
        self.qmatrix = [[0.5 for _ in range(action_space.n)] for _ in range(observation_space.n)]
        self.e = numpy.zeros((self.observation_space.n, self.action_space.n))
        self.last_state = 0
        self.last_action = -1
        self.DF = DF
        self.LF = LF
        self.lam = lam
        self.er = er
        self.er_limit = er_limit

    def reset(self):
        self.e = numpy.zeros((self.observation_space.n, self.action_space.n))
        self.last_state = 0
        self.last_action = -1

    def act(self, state, reward, done, episode):
        action = self.get_action(state, episode)
        if self.last_action != -1:
            self.recalculate(state, reward, done, action)
        self.last_action = action
        self.last_state = state

        return action
    
    def get_action(self, state, t):
        max_action = numpy.argmax(self.qmatrix[state])
        cond = random.uniform(0, 1) > self.er or t > self.er_limit
        return max_action if cond else random.randint(0, self.action_space.n - 1)

    def recalculate(self, state, reward, done, next_action):
        old_value = self.qmatrix[self.last_state][self.last_action]
        next_value = self.qmatrix[state][next_action]
        delta = reward + self.DF * next_value - old_value
        self.e[self.last_state][self.last_action] += 1

        if done:
            delta = reward - old_value

        for state_ in range(self.observation_space.n):
            for action_ in range(self.action_space.n):
                self.qmatrix[state_][action_] += self.LF * self.e[state_][action_] * delta
                self.e[state_][action_] *= self.lam * self.DF


def run(agent, count=10000):
    last100 = []
    average_score = 0
    i_episode = 0
    while True:
        i_episode += 1
        agent.reset()
        observation = env.reset()
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
                break

        if average_score >= 0.78:
            print(average_score, i_episode)
            return i_episode

if __name__ == '__main__':
    statistics = []
    for i in xrange(0, 10):
        print "game {}".format(i),
        agent = SarsaAgent(env.action_space, env.observation_space, 0.9, 0.1, 0.2, 0.0, 100)
        statistics.append(run(agent))
    assert len(statistics) == 10
    print statistics
    print numpy.mean(statistics)
