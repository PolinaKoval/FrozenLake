import random
import numpy


class QLearningAgent(object):
    def __init__(self, action_space, observation_space, DF, LF):
        self.action_space = action_space
        self.observation_space = observation_space
        self.qmatrix = [[1.0 for _ in range(action_space.n)] for _ in range(observation_space.n)]
        self.last_state = 0
        self.last_action = -1
        self.DF = DF
        self.LF = LF

    def reset(self):
        self.last_state = 0
        self.last_action = -1

    def act(self, state, reward, done, episode):
        if self.last_action != -1:
            self.recalculate(state, reward, done)
        action = self.get_action(state, episode)
        self.last_action = action
        self.last_state = state
        return action

    def get_action(self, state, t):
        lim = 0.005
        if t < 3000:
            lim = 0.4
        max_action = numpy.argmax(self.qmatrix[state])
        cond = random.random() > lim
        return max_action if cond else random.randint(0, self.action_space.n - 1)

    def recalculate(self, state, reward, done):
        old_value = self.qmatrix[self.last_state][self.last_action]
        max_value = numpy.amax(self.qmatrix[state])
        delta = reward + self.DF * max_value - old_value

        if done:
            delta = reward - old_value

        new_value = old_value + self.LF * delta
        self.qmatrix[self.last_state][self.last_action] = new_value


def agent(action_space, observation_space):
    return QLearningAgent(action_space, observation_space, 0.99, 0.1)
