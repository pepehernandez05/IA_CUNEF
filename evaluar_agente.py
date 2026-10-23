
import numpy as np
from basket_env import BasketballEnv
from wrappers.env_wrappers import RewardShapingWrapper
from agente import QLearningAgent

MODEL_PATH = "./best_model/basket_agent_q_table.npy"
NUM_EPISODES = 1000


def evaluate_agent():
    env = RewardShapingWrapper(BasketballEnv())
    agent = QLearningAgent(env)

    # Cargar el modelo entrenado
    agent.q_table = np.load(MODEL_PATH)
    agent.epsilon = 0.0

    successes = 0
    failures = 0
    truncations = 0

    for _ in range(NUM_EPISODES):
        obs, info = env.reset()

        done = False
        truncated = False

        while not done and not truncated:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        if info.get("scored", False):
            successes += 1
        elif truncated:
            truncations += 1
        else:
            failures += 1

    total = NUM_EPISODES

    print("\n--- RESULTADOS DE EVALUACIÓN ---")
    print(f"Lanzamientos evaluados: {total}")
    print(f"Canastas: {successes} ({successes / total:.1%})")
    print(f"Fallos: {failures} ({failures / total:.1%})")
    print(f"Truncamientos: {truncations} ({truncations / total:.1%})")

    env.close()


if __name__ == "__main__":
    evaluate_agent()
