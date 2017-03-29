from env2048 import Game2048
import numpy as np
env = Game2048(4)

startegy = [[2, 1], [0, 2], [1, 1], [1, 3]]

statistics = []

for i in range(1000):
    env.reset()
    score = 0
    action = 1
    grid, reward, done, success = env.step(action)
    for j in range(1000):
        action = startegy[action][success]
        grid, reward, done, success = env.step(action)
        score += reward
        if done:
            break
    statistics.append(score)

print np.mean(statistics), np.median(statistics),
