import numpy as np
import os
# Imports Corregidos (sin "wrappers.")
from basket_env import BasketballEnv
from wrappers.env_wrappers import RewardShapingWrapper
from wrappers.log_utils import plot_logs

# --- Clase Agente Q-Learning ---
class QLearningAgent:
    """
    Agente Q-Learning Tabular con discretización de acción.
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999):
        # Parámetros de RL
        self.lr = learning_rate 
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.env = env
        self.num_states = env.observation_space.n 
        
        # Discretización de Acciones (5 ángulos x 5 velocidades = 25 acciones discretas)
        self.num_actions_discrete = 5 * 5 
        
        action_space = env.unwrapped.action_space
        self.action_angles = np.linspace(action_space.low[0], action_space.high[0], 5)
        self.action_speeds = np.linspace(action_space.low[1], action_space.high[1], 5)
        
        self.q_table = np.zeros((self.num_states, self.num_actions_discrete))

    def _get_continuous_action(self, discrete_action_index):
        """Mapea el índice de acción discreta (0-24) a (ángulo, velocidad) continuo."""
        angle_idx = discrete_action_index // 5
        speed_idx = discrete_action_index % 5
        
        return np.array([self.action_angles[angle_idx], self.action_speeds[speed_idx]], dtype=np.float32)

    def predict(self, obs, deterministic=False):
        """Selecciona una acción usando la política e-greedy."""
        state = obs
        
        if not deterministic and np.random.rand() < self.epsilon:
            # Explorar: elegir una acción discreta al azar
            action_discrete_index = np.random.randint(self.num_actions_discrete) 
        else:
            # Explotar: elegir la mejor acción de la tabla Q
            action_discrete_index = np.argmax(self.q_table[state, :])
            
        action_continuous = self._get_continuous_action(action_discrete_index)
        
        return action_continuous, action_discrete_index 

    def update_q_table(self, state, action_discrete_index, reward, next_state):
        """Aplica la fórmula de actualización de Q-Learning (Ecuación de Bellman)."""
        old_value = self.q_table[state, action_discrete_index]
        next_max = np.max(self.q_table[next_state, :])
        
        # Ecuación de Q-Learning: Q(s, a) = (1-lr)*Q(s, a) + lr * (r + gamma * max_a' Q(s', a'))
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        
        self.q_table[state, action_discrete_index] = new_value
        
    def decay_epsilon(self):
        """Reduce el factor de exploración."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Guarda la tabla Q."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path + "_q_table.npy", self.q_table)


# --- Bucle de Entrenamiento ---
def train_agent():
    # Crear entorno con wrapper de recompensa
    env = BasketballEnv()
    # La clase ObservationWrapper fue eliminada/omitida
    env = RewardShapingWrapper(env) 
    
    # Configuración de logs manual
    os.makedirs("./logs/", exist_ok=True)
    log_file = open("./logs/manual_log.csv", "w")
    log_file.write("r,l,t\n") # Encabezados: recompensa, longitud, timestep global

    agent = QLearningAgent(env) 

    total_timesteps = 50_000_000
    
    print(" Entrenando agente Q-Learning...")
    
    # Bucle de entrenamiento manual
    obs, info = env.reset()
    state = obs
    episode_reward = 0
    episode_length = 0
    
    for step in range(total_timesteps):
        # 1. Selección de acción
        action_continuous, action_discrete_index = agent.predict(state) 
        
        # 2. Paso en el entorno
        next_obs, reward, done, truncated, info = env.step(action_continuous)
        next_state = next_obs
        
        # 3. Actualización de Q-Table
        agent.update_q_table(state, action_discrete_index, reward, next_state)
        
        # 4. Decay de Epsilon
        agent.decay_epsilon()
        
        episode_reward += reward
        episode_length += 1

        if done or truncated:
            # Registro manual
            log_file.write(f"{episode_reward},{episode_length},{step}\n")
            log_file.flush()
            
            obs, info = env.reset()
            state = obs 
            episode_reward = 0
            episode_length = 0
            
        else:
            state = next_state
            
    log_file.close()

    agent.save("./best_model/basket_agent")
    print(" Entrenamiento completado.")

    plot_logs("./logs/")

if __name__ == "__main__":
    train_agent()