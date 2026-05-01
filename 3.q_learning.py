import gym
import numpy as np
import warnings

setattr(np, "bool8", np.bool)

warnings.filterwarnings("ignore")

env = gym.make("CliffWalking-v0")

action_n = env.action_space.n

obsevation_n = env.observation_space.n

Q_table = np.zeros(shape=(obsevation_n, action_n))

total_episode = 500

total_step = 1

obsation = env.reset()[0]

lr = 0.2

gamma = 0.9

def predict(obs, total_step):

    epsilon = 1 / total_step

    if np.random.rand() < epsilon:
        action = np.random.choice(action_n)
    
    else:
        max_i = np.argmax(Q_table[obs])
    
        max_v = Q_table[obs][max_i]
        q_list = np.where(Q_table[obs]==max_v)[0]
        action = np.random.choice(q_list)

    return action



for episode in range(total_episode):
    
    total_reward = 0
    while True:

        # 预测

        action = predict(obsation, total_step)
        next_observation, reward, done, truncated, info = env.step(action)

        total_reward += reward

        next_act_index = np.argmax(Q_table[next_observation])

        next_act_v = Q_table[next_observation][next_act_index]
        next_act = np.random.choice(np.where(Q_table[next_observation] == next_act_v)[0])
        
        # 使用公式更新Q table
        if done:
            target_reward = reward
        
        else:
            target_reward = reward + gamma * Q_table[next_observation][next_act]
        
        Q_table[obsation][action] += lr*(target_reward - Q_table[obsation][action])

        total_step += 1

        if done:
            break
    
        else:
            obsation = next_observation

    
    obsation = env.reset()[0]

    print(f"episode:{episode}, rerard:{total_reward}")



np.save("Q_table.npy", Q_table)