import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")  # 升级到 v1

    total_reward = 0.0
    total_steps = 0
    obs, _ = env.reset()  # obs后面加下划线代表的是info，info是一个字典，包含了环境的额外信息，跟step函数的info参数一样

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)  # 原来的done参数被拆分成了terminated和truncated两个参数，terminated表示游戏是否结束，truncated表示游戏是否被截断
        total_reward += reward
        total_steps += 1
        if terminated or truncated:  # 检查是否结束
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
