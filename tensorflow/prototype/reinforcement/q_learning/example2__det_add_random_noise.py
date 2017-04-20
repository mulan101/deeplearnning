'''
Created on 2017. 4. 17.

@author: 한제호
임의의 렌던값을 추가하여  다양한 경로획득
'''
import gym
import numpy as np
from gym.envs.registration import register
import matplotlib.pyplot as plt
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')        

print('env.observation_space.n: ',env.observation_space.n,'env.action_space.n: ',env.action_space.n)
Q = np.zeros([env.observation_space.n,env.action_space.n])
dis = .99
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    
    while not done:
        # noise 추가
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))
        new_state, reward, done, info = env.step(action)
        # Q 업데이트시에 미래의 reward에 대한 dis 적용
        Q[state, action] = reward + dis*np.max(Q[new_state, :])
        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()  

'''
Success rate: 0.9645
Final Q-Table Values
LEFT DOWN RIGHT UP
[[ 0.          0.95099005  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.96059601  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.970299    0.        ]
 [ 0.          0.9801      0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.99        0.        ]
 [ 0.          0.          1.          0.        ]
 [ 0.          0.          0.          0.        ]]
'''
