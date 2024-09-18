# import torch


# class Fish:
#     def __init__(self):
#         self.name: int = 3


def fun(a):
    a = 10

aa= 100
fun(aa)
print(aa)


import gym
env = gym.make("MountainCar-v0")
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    env.step(action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated:
        observation = env.reset()
env.close()