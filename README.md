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
| `5.reinforce.ipynb` | REINFORCE（策略梯度） | CartPole-v1 | ✅ |
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

## 调参玄学守则（亲身踩坑实录）

监督学习里超参调错了，最多就是收敛慢一点；RL 里超参调错了，是**根本不学**。这不是夸张修辞，是日常体验。

### 我已经踩过的坑

- **Actor-Critic 把 actor_lr 和 critic_lr 传反了**：500 个 episode 平均 reward 9.2，比随机策略（约 22）还差。代码逻辑全对，loss 函数全对，网络结构全对——就因为传了个 `1e-2` 而不是 `1e-3`，policy 在前 30 个 episode 就塌缩成确定策略，再也爬不出来
- 一个超参数差一个数量级 = 完全不同的算法
- 同一份代码换个 random seed，结果可以从能学变成废物
- reward 量级没缩放、观测没归一化，很多任务就是不学，跟代码对错无关

### 为什么 RL 这么脆弱

监督学习的数据是**固定**的，超参再烂也只是优化效率问题。

RL 的数据是策略**自己生成**的：

```
策略 → 采样轨迹 → 学习信号 → 更新策略 → 新的轨迹 → ...
```

任何一个环节出问题，整个反馈环就会自我强化地崩掉。**"训练越久数据质量越差"** 这种事在监督学习里不会发生，但在 RL 里是日常。

### 实用守则

1. **Critic LR 必须 ≥ Actor LR**（通常差 5–10 倍）。否则 actor 拿着 critic 给的噪声 advantage 做大步更新，就是在瞎走，瞎走几步就塌缩
2. 不要只看 reward，要同时看 `actor_loss`、`critic_loss`、动作概率分布、advantage 分布——这些里任何一个异常都是早期信号（actor_loss 趋近于 0 = 策略塌缩了）
3. 多跑几个 seed。RL 单次跑分基本没有统计意义
4. 复现别人的算法时，代码对了不代表能跑——超参对齐才是关键
5. PPO / SAC / TD3 这些 "现代" 算法之所以稳，很大一部分原因是它们设计上就在**抑制反馈环爆炸**（PPO 的 clip、SAC 的 entropy 正则、TD3 的延迟更新）。Vanilla 算法（比如本仓库前 6 个）几乎没有任何防护机制，所以特别容易翻车

> 所以这个仓库里，**不学习的版本和能学习的版本，往往只差一个学习率。**

---

## 环境依赖

```bash
pip install numpy matplotlib torch gymnasium
```

---

## 学习资源

- 教材：[动手学强化学习](https://hrl.boyuai.com/chapter/)
- 作者：张伟楠、沈键、俞勇
