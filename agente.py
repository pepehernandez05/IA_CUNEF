import numpy as np # Para manejo de tablas Q y operaciones matemáticas
import os # Para manejo de directorios y guardado de modelos
# Los siguientes imports provienen de archivos locales/externos:
# Importa la clase base del entorno de baloncesto (que probablemente usa estados discretos)
from basket_env import BasketballEnv
# Importa un wrapper para modificar las recompensas del entorno base
from wrappers.env_wrappers import RewardShapingWrapper
# Importa una función para graficar los resultados del log
from wrappers.log_utils import plot_logs

# --- Clase Agente Q-Learning ---
class QLearningAgent:
    """
    Agente Q-Learning Tabular con discretización de acción.

    El agente aprende a seleccionar la mejor combinación discreta (ángulo, velocidad)
    a partir de una tabla Q.
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999):
        # Parámetros del algoritmo de Reinforcement Learning (RL)
        self.lr = learning_rate # Tasa de aprendizaje ($\alpha$)
        self.gamma = discount_factor # Factor de descuento ($\gamma$)
        self.epsilon = epsilon # Tasa de exploración inicial ($\epsilon$)
        self.epsilon_min = epsilon_min # Mínimo valor de exploración
        self.epsilon_decay = epsilon_decay # Tasa de decaimiento de $\epsilon$

        self.env = env
        # Se asume que el espacio de observación (estados) es discreto (ej. 100 estados)
        self.num_states = env.observation_space.n

        # --- Discretización de Acciones ---
        # El problema real es continuo (ángulo y velocidad). Se crea una rejilla discreta.
        # En este caso: 5 niveles de ángulo x 5 niveles de velocidad = 25 acciones discretas
        self.num_actions_discrete = 5 * 5

        # Obtiene los límites del espacio de acción real (continuo)
        action_space = env.unwrapped.action_space
        # Crea 5 puntos espaciados linealmente para el ángulo (ej: 0.0, 0.25, 0.5, 0.75, 1.0)
        self.action_angles = np.linspace(action_space.low[0], action_space.high[0], 5)
        # Crea 5 puntos espaciados linealmente para la velocidad
        self.action_speeds = np.linspace(action_space.low[1], action_space.high[1], 5)

        # Inicializa la Q-Tabla con dimensiones [Estados x Acciones Discretas]
        self.q_table = np.zeros((self.num_states, self.num_actions_discrete))

    def _get_continuous_action(self, discrete_action_index):
        """
        Mapea el índice de acción discreta (0-24) a (ángulo, velocidad) continuo.

        Ej: Si discrete_action_index = 6:
        angle_idx = 6 // 5 = 1 (Segundo nivel de ángulo)
        speed_idx = 6 % 5 = 1 (Segundo nivel de velocidad)

        Devuelve el par (ángulo, velocidad) real para el entorno.
        """
        angle_idx = discrete_action_index // 5
        speed_idx = discrete_action_index % 5

        return np.array([self.action_angles[angle_idx], self.action_speeds[speed_idx]], dtype=np.float32)

    def predict(self, obs, deterministic=False):
        """Selecciona una acción usando la política e-greedy (Exploración vs. Explotación)."""
        state = obs

        if not deterministic and np.random.rand() < self.epsilon:
            # Explorar: elegir una acción discreta al azar (índice 0 a 24)
            action_discrete_index = np.random.randint(self.num_actions_discrete)
        else:
            # Explotar: elegir la mejor acción discreta de la tabla Q para el estado actual
            action_discrete_index = np.argmax(self.q_table[state, :])

        # Convierte el índice discreto de la tabla Q a los valores continuos (ángulo, velocidad)
        action_continuous = self._get_continuous_action(action_discrete_index)

        return action_continuous, action_discrete_index

    def update_q_table(self, state, action_discrete_index, reward, next_state):
        """
        Aplica la fórmula de actualización de Q-Learning (Ecuación de Bellman).
        """
        old_value = self.q_table[state, action_discrete_index]
        # Encuentra el máximo valor Q para el siguiente estado (s')
        next_max = np.max(self.q_table[next_state, :])

        # Ecuación de Q-Learning (forma equivalente a la de los ejemplos anteriores):
        # Q(s, a) = (1-lr)*Q(s, a) + lr * (r + gamma * max_a' Q(s', a'))
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)

        self.q_table[state, action_discrete_index] = new_value

    def decay_epsilon(self):
        """Reduce el factor de exploración (epsilon) para fomentar la explotación."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Guarda la tabla Q en formato .npy."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path + "_q_table.npy", self.q_table)

# ----------------------------------------------------------------------
# --- Bucle de Entrenamiento ---

def train_agent():
    # 1. Configuración del Entorno
    env = BasketballEnv() # Entorno base (simulador de tiro libre, posiblemente continuo en acciones)
    # Se aplica el Wrapper de Reward Shaping: modifica las recompensas para guiar el aprendizaje
    env = RewardShapingWrapper(env)

    # 2. Configuración de Logs (Manual)
    os.makedirs("./logs/", exist_ok=True)
    log_file = open("./logs/manual_log.csv", "w")
    # Los logs registran la recompensa total del episodio, la longitud del episodio y el paso de tiempo global
    log_file.write("r,l,t\n")

    # 3. Inicialización del Agente
    agent = QLearningAgent(env)

    total_timesteps = 500_000 # El agente se entrenará durante medio millón de pasos (largo)

    print(" Entrenando agente Q-Learning...")

    # Variables de seguimiento
    obs, info = env.reset() # Reinicia el entorno y obtiene la observación inicial
    state = obs # La observación es el estado (asumiendo que son equivalentes)
    episode_reward = 0 # Recompensa acumulada en el episodio actual
    episode_length = 0 # Longitud del episodio actual

    # 4. Bucle principal de entrenamiento (interacción paso a paso)
    for step in range(total_timesteps):
        # 4a. Selección de acción (usa epsilon-greedy)
        # Devuelve la acción continua para el entorno y el índice discreto para la Q-Tabla
        action_continuous, action_discrete_index = agent.predict(state)

        # 4b. Paso en el entorno
        # Ejecuta la acción continua en el simulador
        next_obs, reward, done, truncated, info = env.step(action_continuous)
        next_state = next_obs

        # 4c. Actualización de Q-Table (la esencia del Q-Learning)
        agent.update_q_table(state, action_discrete_index, reward, next_state)

        # 4d. Decay de Epsilon
        agent.decay_epsilon()

        # 4e. Acumulación de métricas
        episode_reward += reward
        episode_length += 1

        # 4f. Fin del Episodio
        if done or truncated:
            # Registro de los resultados del episodio en el archivo CSV
            log_file.write(f"{episode_reward},{episode_length},{step}\n")
            log_file.flush() # Asegura que los datos se escriban inmediatamente

            # Reiniciar para el siguiente episodio
            obs, info = env.reset()
            state = obs
            episode_reward = 0
            episode_length = 0

        else:
            # Transición de estado
            state = next_state

    log_file.close()

    # 5. Guardado y Visualización
    agent.save("./best_model/basket_agent")
    print(" Entrenamiento completado.")

    # Genera los gráficos a partir del archivo de log
    plot_logs("./logs/")

if __name__ == "__main__":
    train_agent()