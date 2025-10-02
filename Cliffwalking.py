import numpy as np
import gymnasium as gym
from collections import defaultdict
import time
import os

def epsilon_greedy(Q, state, nA, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(nA)
    else:
        return np.argmax(Q[state])

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][best_next_action] - Q[state][action]
            )
            state = next_state
    return Q

def render_cliff(state, path=[]):
    nrows, ncols = 4, 12
    grid = [["."] * ncols for _ in range(nrows)]

    for c in range(1, 11):
        grid[3][c] = "C"

    grid[3][0] = "S"
    grid[3][11] = "G"

    for s in path:
        r, c = divmod(s, ncols)
        grid[r][c] = "*"

    r, c = divmod(state, ncols)
    grid[r][c] = "A"

    os.system("cls" if os.name == "nt" else "clear")
    for row in grid:
        print(" ".join(row))
    print()

def play(env, Q, epsilon=0.0, sleep_time=0.5, max_steps=100):
    state, _ = env.reset()
    done = False
    path = []
    steps = 0
    while not done and steps < max_steps:
        render_cliff(state, path)
        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        path.append(state)
        state = next_state
        done = terminated or truncated
        steps += 1
        time.sleep(sleep_time)
    render_cliff(state, path)
    print("Episodio terminado 🎉")

if __name__ == "__main__":
    env = gym.make("CliffWalking-v1")

    # 🔹 Antes de entrenar → tabla Q vacía = agente aleatorio
    print("Agente antes de entrenar (aleatorio):")
    Q_empty = defaultdict(lambda: np.zeros(env.action_space.n))
    play(env, Q_empty, epsilon=1.0, sleep_time=0.3)

    # 🔹 Entrenamiento
    Q = q_learning(env, num_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1)

    # 🔹 Después de entrenar → agente greedy con la política aprendida
    print("Agente después de entrenar (Q-learning):")
    play(env, Q, epsilon=0.0, sleep_time=0.3)
