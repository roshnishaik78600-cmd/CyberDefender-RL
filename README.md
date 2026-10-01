# CyberDefender-RL

Adaptive Cybersecurity Response Using Reinforcement Learning.

CyberDefender-RL is a simulation-based cybersecurity project that uses **Q-learning** to learn defensive responses to different cyber attacks.

## Features

- Q-learning from scratch
- 4 cyber attack types
- 3 severity levels
- 3 traffic levels
- 4 defensive actions
- Epsilon-greedy exploration
- Q-table learning
- 2,000 training episodes
- RL agent evaluation
- Random baseline comparison
- Training reward visualization
- Learned policy inspection

## Environment

### Attacks
- Port Scan
- Brute Force
- Malware
- DDoS

### Actions
- Monitor
- Block IP
- Isolate Host
- Escalate

## Q-Learning

The agent learns:

```text
Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]