import gym
import numpy
from SarsaAgent import agent as sarsa_agent
from QlearningAgent import agent as qlearning_agent

reward_threshold = {
    'FrozenLake-v0': 0.78,
    'FrozenLake8x8-v0': 0.99,
    'Taxi-v2': 8.46
}


agents = {
    'sarsa': sarsa_agent,
    'qlearning': qlearning_agent
}


def run(agent, env, env_reward_threshold, count=1000000):
    last100 = []
    average_score = 0
    i_episode = 0
    while i_episode < count:
        agent.reset()
        observation = env.reset()
        reward = 0
        score = 0
        done = False
        while True:
            action = int(agent.act(observation, reward, done, i_episode))
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                last100.append(score)
                if len(last100) > 100:
                    last100.pop(0)
                    average_score = numpy.mean(last100)
                agent.act(observation, reward, done, i_episode)
                break
        if average_score >= env_reward_threshold:
            return i_episode
        i_episode += 1


def run_agent(env_name, agent_name):
    env = gym.make(env_name)
    env._max_episode_steps = 5000
    statistics = []
    games = 10
    for i in range(0, games):
        agent = agents[agent_name](env.action_space, env.observation_space)
        learning_result = run(agent, env, reward_threshold[env_name])
        statistics.append(learning_result)
    assert len(statistics) == games
    print(statistics)
    print('Env: {}\nAlgo: {}\nResult: {}\n'.format(env_name, agent_name, numpy.mean(statistics)))


if __name__ == '__main__':
    for algo in agents:
        for environment in reward_threshold:
            run_agent(environment, algo)
