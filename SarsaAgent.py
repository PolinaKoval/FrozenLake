import random
import numpy

class SarsaAgent(object):
    def __init__(self, action_space, observation_space, DF, LF, lam, er, er_limit):
        self.action_space = action_space
        self.observation_space = observation_space
        self.qmatrix = numpy.array([[1.0 for _ in range(action_space.n)] for _ in range(observation_space.n)])
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
        cond = random.random() >= self.er or t > self.er_limit
        return max_action if cond else random.randint(0, self.action_space.n - 1)

    def recalculate(self, state, reward, done, next_action):
        old_value = self.qmatrix[self.last_state][self.last_action]
        next_value = self.qmatrix[state][next_action]
        delta = reward + self.DF * next_value - old_value
        self.e[self.last_state][self.last_action] += 1

        if done:
            delta = reward - old_value

        self.qmatrix += self.LF * self.e * delta
        self.e *= self.lam * self.DF

def agent(action_space, observation_space):
    return SarsaAgent(action_space, observation_space, 0.999, 0.4, 0.2, 0.001, 10000)
