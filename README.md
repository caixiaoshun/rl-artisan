# 7天强化学习训练营

> 在 vibe coding 泛滥的今天，依然坚持古法编程的非遗传承人

7天，7个算法，从摇臂机到 PPO，全部手写，拒绝复制粘贴。

跟随 [《动手学强化学习》](https://hrl.boyuai.com/chapter/) 完成。

---

## 课程表

| 天数 | 算法 | 文件 | 环境 |
|------|------|------|------|
| Day 1 | 多臂老虎机（ε-Greedy、UCB） | `1.多臂老虎机.ipynb` | 自定义 |
| Day 2 | Sarsa | `2.sarsa.py` | CliffWalking-v0 |
| Day 3 | Q-Learning | `3.q_learning.py` | CliffWalking-v0 |
| Day 4 | DQN（经验回放 + 目标网络） | `4.dqn.ipynb` | CartPole-v1 |
| Day 5 | REINFORCE（策略梯度） | `5.reIinforce.ipynb` | CartPole-v1 |
| Day 6 | Actor-Critic | `6.actor-critor.ipynb` | CartPole-v0 |
| Day 7 | PPO（Clip 版） | `7.ppo.ipynb` | CartPole-v0 |

---

## 踩坑实录

RL 的超参不是调出来的，是试错出来的。监督学习里超参调歪了最多收敛慢，RL 里超参调歪了是**根本不学**。

几个亲历的教训：

- **Actor-Critic 把两个学习率传反了**：跑了 500 个 episode，平均 reward 9.2，比随机策略（约 22）还差。代码逻辑全对，网络结构全对，就因为 actor_lr 和 critic_lr 位置写反了。Critic 太慢，给出的 advantage 全是噪声，Actor 拿着噪声更新，30 个 episode 后策略就塌缩成确定性策略，再也爬不出来。
- 同一份代码换个随机种子，结果可以从"能学"变成"废物"，RL 单次跑分基本没有统计意义。
- Vanilla 算法（仓库里前 6 个）几乎没有任何防护机制，PPO 的 clip 是第一个真正稳定的设计。

**Critic LR 必须 ≥ Actor LR**，通常差 5–10 倍，这条最重要。

---

## 环境依赖

```bash
pip install numpy matplotlib torch gymnasium
```

---

## 参考资料

- 教材：[动手学强化学习](https://hrl.boyuai.com/chapter/)（张伟楠、沈键、俞勇）
