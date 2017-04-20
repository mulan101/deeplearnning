'''
Created on 2017. 4. 17.

@author: 한제호
'''
import gym
from colorama import init
from gym.envs.registration import register
from prototype.basic.kbhit import KBHit 

init(autoreset=True)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': True}
)

env = gym.make('FrozenLake-v3')
env.render()

key = KBHit()

while True:

    action = key.getarrow();
    if action not in [0, 1, 2, 3]:
        print("Game aborted!")
        break

    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break
