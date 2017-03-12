import gym
import random
import numpy
env = gym.make('FrozenLake-v0')


class QLearningAgent(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.qmatrix = [[0.1 for x in range(action_space.n)] for y in range(observation_space.n)]

        self.last_state = 0
        self.last_action = -1
        self.DF = 0.99
        self.LF = 0.1

    def act(self, state, reward, done, episode):
        if self.last_action != -1:
            self.recalculate(state, reward)
        action = self.get_action(state, episode)
        self.last_action = action
        self.last_state = state
        return action
    
    def get_action(self, state, t):
        max_action = numpy.argmax(self.qmatrix[state])
        cond = random.uniform(0, 1) > float(100)/(t + 1)
        return max_action if cond else random.randint(0, self.action_space.n - 1)

    def recalculate(self, state, reward):
        old_value = self.qmatrix[self.last_state][self.last_action]
        max_value = numpy.amax(self.qmatrix[state])
        new_value = old_value + self.LF * (reward + self.DF * max_value - old_value)
        self.qmatrix[self.last_state][self.last_action] = new_value


def run(agent, count=10000):
    episode_count = count
    wins = 0
    last100 = []
    average_score = 0
    for i_episode in range(episode_count):
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
