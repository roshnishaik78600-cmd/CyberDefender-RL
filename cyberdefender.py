import os
import random

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CYBERDEFENDER-RL
# Adaptive Cybersecurity Response Using Q-Learning
# ============================================================


# ============================================================
# 1. ENVIRONMENT DEFINITIONS
# ============================================================

ATTACK_TYPES = {
    0: "Port Scan",
    1: "Brute Force",
    2: "Malware",
    3: "DDoS"
}

SEVERITY = {
    0: "Low",
    1: "Medium",
    2: "High"
}

TRAFFIC = {
    0: "Low",
    1: "Medium",
    2: "High"
}

ACTIONS = {
    0: "Monitor",
    1: "Block IP",
    2: "Isolate Host",
    3: "Escalate"
}


# ============================================================
# 2. RL HYPERPARAMETERS
# ============================================================

ALPHA = 0.1
GAMMA = 0.9

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

NUM_ACTIONS = len(ACTIONS)


# ============================================================
# 3. ENVIRONMENT
# ============================================================

def generate_state():
    """
    Generate a random cybersecurity incident.

    State:
    (attack_type, severity, traffic)
    """

    attack_type = random.randint(0, 3)
    severity = random.randint(0, 2)
    traffic = random.randint(0, 2)

    return attack_type, severity, traffic


def get_reward(state, action):
    """
    Return reward for taking an action
    in a particular cybersecurity state.
    """

    attack_type, severity, traffic = state

    # -------------------------
    # Port Scan
    # -------------------------

    if attack_type == 0:

        if action == 0:
            return 5

        elif action == 1:
            return 2

        else:
            return -5

    # -------------------------
    # Brute Force
    # -------------------------

    elif attack_type == 1:

        if action == 1:
            return 8

        elif action == 0:
            return -3

        else:
            return -2

    # -------------------------
    # Malware
    # -------------------------

    elif attack_type == 2:

        if action == 2:
            return 10

        elif action == 3:
            return 8

        elif action == 1:
            return 3

        else:
            return -10

    # -------------------------
    # DDoS
    # -------------------------

    elif attack_type == 3:

        if severity == 2 and action == 3:
            return 10

        elif action == 1:
            return 7

        elif action == 0:
            return -10

        else:
            return -3

    return 0


def get_next_state():
    """
    Generate the next cybersecurity incident.
    """

    return generate_state()


# ============================================================
# 4. Q-TABLE
# ============================================================

# 4 attack types
# 3 severity levels
# 3 traffic levels
# 4 possible actions
#
# Total:
# 4 × 3 × 3 × 4 = 144 Q-values

Q = np.zeros(
    (4, 3, 3, NUM_ACTIONS)
)


# ============================================================
# 5. EPSILON-GREEDY ACTION SELECTION
# ============================================================

def choose_action(state, epsilon):
    """
    Choose an action using epsilon-greedy strategy.
    """

    attack_type, severity, traffic = state

    # -------------------------
    # Exploration
    # -------------------------

    if random.random() < epsilon:

        return random.randint(
            0,
            NUM_ACTIONS - 1
        )

    # -------------------------
    # Exploitation
    # -------------------------

    return int(
        np.argmax(
            Q[
                attack_type,
                severity,
                traffic
            ]
        )
    )


# ============================================================
# 6. Q-LEARNING UPDATE
# ============================================================

def update_q(state, action, reward, next_state):
    """
    Update Q-table using the Q-learning equation.

    Q(s,a) =
        Q(s,a) +
        alpha * (
            reward +
            gamma * max(Q(s',a'))
            - Q(s,a)
        )
    """

    attack_type, severity, traffic = state

    next_attack, next_severity, next_traffic = next_state

    # Current Q-value
    current_q = Q[
        attack_type,
        severity,
        traffic,
        action
    ]

    # Best future Q-value
    best_future_q = np.max(
        Q[
            next_attack,
            next_severity,
            next_traffic
        ]
    )

    # Q-learning update
    new_q = current_q + ALPHA * (
        reward
        + GAMMA * best_future_q
        - current_q
    )

    # Store updated value
    Q[
        attack_type,
        severity,
        traffic,
        action
    ] = new_q


# ============================================================
# 7. TRAINING
# ============================================================

def train(episodes=2000):
    """
    Train the Q-learning agent.
    """

    epsilon = EPSILON

    rewards_history = []

    for episode in range(episodes):

        # Generate state
        state = generate_state()

        # Choose action
        action = choose_action(
            state,
            epsilon
        )

        # Get reward
        reward = get_reward(
            state,
            action
        )

        # Generate next state
        next_state = get_next_state()

        # Update Q-table
        update_q(
            state,
            action,
            reward,
            next_state
        )

        # Save reward
        rewards_history.append(
            reward
        )

        # Reduce exploration
        epsilon = max(
            EPSILON_MIN,
            epsilon * EPSILON_DECAY
        )

        # Show progress
        if (episode + 1) % 500 == 0:

            print(
                f"Episode {episode + 1:4} | "
                f"Reward: {reward:3} | "
                f"Epsilon: {epsilon:.3f}"
            )

    return rewards_history


# ============================================================
# 8. RL AGENT EVALUATION
# ============================================================

def evaluate_rl_agent(episodes=1000):
    """
    Evaluate trained RL agent.

    epsilon = 0 means:
    no exploration.
    """

    total_reward = 0
    positive_responses = 0

    for _ in range(episodes):

        state = generate_state()

        # Exploitation only
        action = choose_action(
            state,
            epsilon=0
        )

        reward = get_reward(
            state,
            action
        )

        total_reward += reward

        if reward > 0:
            positive_responses += 1

    average_reward = (
        total_reward / episodes
    )

    response_rate = (
        positive_responses / episodes
    ) * 100

    return average_reward, response_rate


# ============================================================
# 9. RANDOM BASELINE
# ============================================================

def evaluate_random_agent(episodes=1000):
    """
    Evaluate a random-action agent.
    """

    total_reward = 0
    positive_responses = 0

    for _ in range(episodes):

        state = generate_state()

        # Completely random action
        action = random.randint(
            0,
            NUM_ACTIONS - 1
        )

        reward = get_reward(
            state,
            action
        )

        total_reward += reward

        if reward > 0:
            positive_responses += 1

    average_reward = (
        total_reward / episodes
    )

    response_rate = (
        positive_responses / episodes
    ) * 100

    return average_reward, response_rate


# ============================================================
# 10. DISPLAY Q-TABLE
# ============================================================

def show_q_table():

    print("\n")
    print("=" * 100)
    print("LEARNED Q-TABLE")
    print("=" * 100)

    for attack_type in range(4):

        for severity in range(3):

            for traffic in range(3):

                values = Q[
                    attack_type,
                    severity,
                    traffic
                ]

                best_action = int(
                    np.argmax(values)
                )

                print(
                    f"{ATTACK_TYPES[attack_type]:12} | "
                    f"{SEVERITY[severity]:6} | "
                    f"{TRAFFIC[traffic]:6} | "
                    f"Best: {ACTIONS[best_action]:12} | "
                    f"Q: {np.round(values, 2)}"
                )


# ============================================================
# 11. POLICY SUMMARY
# ============================================================

def show_policy_summary():

    print("\n")
    print("=" * 60)
    print("LEARNED POLICY SUMMARY")
    print("=" * 60)

    for attack_type in range(4):

        print(
            f"\n{ATTACK_TYPES[attack_type]}:"
        )

        for severity in range(3):

            # Use high traffic for summary
            traffic = 2

            values = Q[
                attack_type,
                severity,
                traffic
            ]

            best_action = int(
                np.argmax(values)
            )

            print(
                f"  {SEVERITY[severity]:6} -> "
                f"{ACTIONS[best_action]}"
            )


# ============================================================
# 12. SAVE TRAINED Q-TABLE
# ============================================================

def save_model():

    os.makedirs(
        "models",
        exist_ok=True
    )

    np.save(
        "models/q_table.npy",
        Q
    )

    print(
        "\nQ-table saved to "
        "models/q_table.npy"
    )


# ============================================================
# 13. TRAINING GRAPH
# ============================================================

def plot_training_rewards(rewards):

    window = 100

    moving_average = []

    for i in range(len(rewards)):

        start = max(
            0,
            i - window + 1
        )

        average = np.mean(
            rewards[start:i + 1]
        )

        moving_average.append(
            average
        )

    plt.figure(
        figsize=(10, 5)
    )

    # Raw rewards
    plt.plot(
        rewards,
        alpha=0.3,
        label="Episode Reward"
    )

    # Moving average
    plt.plot(
        moving_average,
        label="100-Episode Average"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Reward"
    )

    plt.title(
        "CyberDefender-RL Training Performance"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# 14. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CYBERDEFENDER-RL")
    print("ADAPTIVE CYBERSECURITY RESPONSE")
    print("Q-LEARNING")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\nStarting training...\n")

    rewards = train(
        episodes=2000
    )

    # --------------------------------------------------------
    # RL EVALUATION
    # --------------------------------------------------------

    rl_reward, rl_response = evaluate_rl_agent(
        episodes=1000
    )

    # --------------------------------------------------------
    # RANDOM BASELINE
    # --------------------------------------------------------

    random_reward, random_response = evaluate_random_agent(
        episodes=1000
    )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    print("\nRL AGENT")
    print("-" * 30)

    print(
        f"Average Reward : "
        f"{rl_reward:.2f}"
    )

    print(
        f"Response Rate  : "
        f"{rl_response:.2f}%"
    )

    print("\nRANDOM BASELINE")
    print("-" * 30)

    print(
        f"Average Reward : "
        f"{random_reward:.2f}"
    )

    print(
        f"Response Rate  : "
        f"{random_response:.2f}%"
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    reward_difference = (
        rl_reward - random_reward
    )

    print("\nCOMPARISON")
    print("-" * 30)

    print(
        f"Reward Difference : "
        f"{reward_difference:.2f}"
    )

    # --------------------------------------------------------
    # LEARNED POLICY
    # --------------------------------------------------------

    show_q_table()

    show_policy_summary()

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    save_model()

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    plot_training_rewards(
        rewards
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CYBERDEFENDER-RL COMPLETE")
    print("=" * 60)