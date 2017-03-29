from env2048 import Game2048
import random
import numpy
env = Game2048(4)


class QLearningAgent(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.qmatrix = [[4 for _ in range(action_space)] for _ in range(observation_space)]

        self.last_state = 0
        self.last_action = 1
        self.DF = 0.5
        self.LF = 0.1

        self.states = {}

        for action in range(self.action_space):
            for res in range(2):
                self.states[action * 10 + res] = action

    def reset(self):
        self.last_action = 1

    def get_state(self, action, result):
        return self.states[action * 10 + result]

    def act(self, reward, episode, success):
        state = self.get_state(self.last_action, success)
        self.recalculate(state, reward)
        action = self.get_action(state, episode)
        self.last_state = state
        self.last_action = action
        return action
    
    def get_action(self, state, t):
        max_action = numpy.argmax(self.qmatrix[state])
        cond = random.uniform(0, 1) > float(100)/(t + 1)
        return max_action if cond else random.randint(0, self.action_space - 1)

    def recalculate(self, state, reward):
        old_value = self.qmatrix[self.last_state][self.last_action]
        max_value = numpy.amax(self.qmatrix[state])
        new_value = old_value + self.LF * (reward + self.DF * max_value - old_value)
        self.qmatrix[self.last_state][self.last_action] = new_value


def run(agent, count=10000):
    results = []
    episode_count = count
    for i_episode in range(episode_count):
        env.reset()
        agent.reset()
        action = agent.last_action
        observation, reward, done, success = env.step(action)
        score = reward
        for step in range(1000):
            action = agent.act(reward, i_episode, success)
            observation, reward, done, success = env.step(action)
            score += reward
            if done:
                results.append(score)
                agent.act(reward, i_episode, success)
                break

    print numpy.mean(results), numpy.median(results)

    policy = map(numpy.argmax, agent.qmatrix)
    print(policy)
    # print(agent.qmatrix)

if __name__ == '__main__':
    agent = QLearningAgent(env.action_space, env.observation_space)
    run(agent)
