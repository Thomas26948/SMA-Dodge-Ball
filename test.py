import random

n_team = 2
attribute = ['speed', 'strength', 'precision']
team_1 = []
team_2 = []
n_player = 6
for i in range(n_team):
    for j in range(n_player):
        speed = random.random() * 10
        strength = random.random() * 10
        precision = random.random() * 10
        if i==0:
            team_1.append([speed, strength, precision])
        else:
            team_2.append([speed, strength, precision])

import numpy as np

team_1_array = np.array(team_1)
print(team_1)
x = np.exp(team_1_array) / np.sum(np.exp(team_1_array), axis=0)

print(x)
print(np.sum(x, axis=0))

