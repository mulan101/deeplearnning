'''
Created on 2017. 4. 17.

@author: 한제호
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
    kwargs={'map_name' : '4x4', 'is_slippery': True}
)

env = gym.make('FrozenLake-v3')        

print('env.observation_space.n: ',env.observation_space.n,'env.action_space.n: ',env.action_space.n)
Q = np.zeros([env.observation_space.n,env.action_space.n])
learning_rate = .85
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
        Q[state, action] = (1-learning_rate)*Q[state,action] + learning_rate*(reward + dis*np.max(Q[new_state, :]))
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
Success rate: 0.683
Final Q-Table Values
LEFT DOWN RIGHT UP
[[  6.36302982e-01   2.23276646e-02   1.28393092e-02   1.20443685e-02]
 [  4.86405937e-04   1.60744926e-03   2.12226018e-04   6.36303626e-01]
 [  3.07302058e-03   2.32877735e-02   1.07853955e-02   6.13696646e-01]
 [  1.66551539e-03   0.00000000e+00   1.53779888e-03   4.24954900e-01]
 [  7.29732317e-01   1.33852319e-03   2.52507152e-04   2.34286713e-04]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  8.48497679e-02   1.27053629e-06   6.40894126e-04   4.83328919e-06]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  1.56676481e-03   3.88276133e-04   1.00314035e-03   7.64025848e-01]
 [  1.05048499e-03   6.29156072e-01   3.16339113e-03   3.34067470e-03]
 [  9.24693595e-01   7.95874610e-05   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   9.01410643e-01   4.55044275e-03]
 [  0.00000000e+00   9.97263387e-01   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]]
'''
