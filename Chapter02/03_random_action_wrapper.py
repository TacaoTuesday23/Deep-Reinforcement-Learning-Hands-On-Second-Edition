import gym
from typing import TypeVar
import random

Action = TypeVar('Action')


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        # 对这个地方不理解是因为没有理解parent class的概念
        # 这里的super()是调用父类的构造函数，env就是传入的环境
        # 这里的self.env就是父类的env属性，表示当前环境
        super(RandomActionWrapper, self).__init__(env) # 调用父类的构造函数
        self.epsilon = epsilon # 随机动作的概率

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v1"))

    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, terminated, truncated, _ = env.step(0)
        total_reward += reward
        if terminated or truncated:
            break

    print("Reward got: %.2f" % total_reward)
