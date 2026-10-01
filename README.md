# CyberDefender-RL

### Adaptive Cybersecurity Response Using Reinforcement Learning

CyberDefender-RL is a **simulation-based cybersecurity reinforcement learning system** that uses **tabular Q-learning** to learn defensive responses to different cyber attack scenarios.

The agent observes an attack state, selects a defensive action, receives a reward or penalty, and progressively learns an optimal response policy.

---

## 🚀 Key Features

* Q-learning implemented from scratch
* 4 simulated cyber attack types
* 3 severity levels
* 3 traffic conditions
* 4 defensive response actions
* Epsilon-greedy exploration
* Q-table based policy learning
* 2,000 training episodes
* 1,000 RL evaluation episodes
* Random-agent baseline comparison
* Training reward visualization
* Moving-average performance analysis
* Learned policy inspection
* Trained Q-table saved for reuse

---

## 🛡️ Cybersecurity Environment

### Attack Types

| Attack      | Example Response    |
| ----------- | ------------------- |
| Port Scan   | Monitor / Block IP  |
| Brute Force | Block IP            |
| Malware     | Isolate Host        |
| DDoS        | Escalate / Block IP |

### State Space

Each security event is represented using:

* **Attack Type:** Port Scan, Brute Force, Malware, DDoS
* **Severity:** Low, Medium, High
* **Traffic:** Low, Medium, High

### Defensive Actions

1. Monitor
2. Block IP
3. Isolate Host
4. Escalate

---

## 🧠 Reinforcement Learning

The project uses **tabular Q-learning**.

The agent learns the value of taking an action in a particular security state using:

```text
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]
```

Where:

* `Q(s,a)` → estimated value of an action
* `α` → learning rate
* `r` → reward received
* `γ` → discount factor
* `s'` → next state

### Exploration vs Exploitation

An **ε-greedy strategy** is used:

```text
Exploration → try a random defensive action
Exploitation → choose the action with the highest Q-value
```

During training, epsilon gradually decreases so the agent moves from exploration toward learned decisions.

---

## 📊 Q-Table

The state-action space contains:

```text
4 attack types
× 3 severity levels
× 3 traffic levels
× 4 actions
= 144 Q-values
```

The trained Q-table is stored at:

```text
models/q_table.npy
```

---

## 📈 Evaluation

The trained agent is evaluated against a **random-action baseline**.

The evaluation compares:

* Average reward
* Positive-response rate
* Learned defensive policy

This provides a simple measurement of whether the learned policy performs better than random action selection within the simulated environment.

---

## 🏗️ Architecture

```text
                Cybersecurity Event
                        │
                        ▼
              ┌─────────────────┐
              │  State Generator │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Q-Learning    │
              │      Agent      │
              └────────┬────────┘
                       │
                Select Action
                       │
                       ▼
              ┌─────────────────┐
              │ Defensive Action│
              └────────┬────────┘
                       │
                       ▼
                Reward / Penalty
                       │
                       ▼
                 Update Q-Table
                       │
                       └──────► Repeat
```

---

## ⚙️ Tech Stack

* **Python**
* **NumPy**
* **Matplotlib**
* **Reinforcement Learning**
* **Q-Learning**
* **Epsilon-Greedy Policy**
* **Tabular Learning**

---

## 📁 Project Structure

```text
CyberDefender-RL/
│
├── cyberdefender.py       # RL environment, training & evaluation
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore             # Git exclusions
│
└── models/
    └── q_table.npy        # Trained Q-table
```

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/roshnishaik78600-cmd/CyberDefender-RL.git
cd CyberDefender-RL
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate it

**Windows:**

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run

```bash
python cyberdefender.py
```

The program trains the agent, evaluates its performance, displays the learned policy, saves the Q-table, and generates the training graph.

---

## 🎯 Learning Objectives

This project demonstrates practical understanding of:

* Reinforcement learning fundamentals
* State and action representation
* Reward engineering
* Q-table construction
* Q-learning updates
* Exploration vs exploitation
* Epsilon decay
* Policy learning
* Model evaluation
* Baseline comparison
* Applying machine learning concepts to cybersecurity

---

## ⚠️ Disclaimer

CyberDefender-RL uses a **simulated cybersecurity environment and synthetic reward rules** for educational purposes.

It is intended to demonstrate reinforcement learning concepts applied to cybersecurity and is **not a production SOC decision-making system**.

---

## 👨‍💻 Author

**Roshni Shaik**

B.Tech CSE — Cybersecurity

GitHub: [roshnishaik78600-cmd](https://github.com/roshnishaik78600-cmd)
