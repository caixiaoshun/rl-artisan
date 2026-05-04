# RL Artisan · 强化学习手写实现

> 在 vibe coding 泛滥的今天，依然坚持古法编程的非遗传承人

跟随 [《动手学强化学习》](https://hrl.boyuai.com/chapter/) 手写强化学习算法，拒绝复制粘贴。

---

## 算法进度

| 文件 | 算法 | 环境 | 状态 |
|------|------|------|------|
| `1.多臂老虎机.ipynb` | 多臂老虎机（ε-Greedy、UCB） | 自定义 | ✅ |
| `2.sarsa.py` | Sarsa | CliffWalking-v0 | ✅ |
| `3.q_learning.py` | Q-Learning | CliffWalking-v0 | ✅ |
| `4.dqn.ipynb` | DQN（经验回放 + 目标网络） | CartPole-v1 | ✅ |
| `5.reIinforce.ipynb` | REINFORCE（策略梯度） | CartPole-v1 | ✅ |
| `6.actor-critor.ipynb` | Actor-Critic | CartPole-v1 | ✅ |
| — | DQN 改进（Double DQN、Dueling DQN） | — | 🔲 |
| — | TRPO | — | 🔲 |
| — | PPO | — | 🔲 |
| — | DDPG | — | 🔲 |
| — | SAC | — | 🔲 |
| — | 模型预测控制（MPC） | — | 🔲 |
| — | 离线强化学习 | — | 🔲 |
| — | 多智能体强化学习（IPPO、MADDPG） | — | 🔲 |

---

## 环境依赖

```bash
pip install numpy matplotlib torch gymnasium
```

---

## 学习资源

- 教材：[动手学强化学习](https://hrl.boyuai.com/chapter/)
- 作者：张伟楠、沈键、俞勇
