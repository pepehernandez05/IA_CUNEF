import gymnasium as gym # Biblioteca estándar para entornos de Reinforcement Learning (RL)
from gymnasium import spaces # Módulo para definir los espacios de acción y observación
import numpy as np # Para cálculos numéricos y manejo de arrays

# --- Constantes de Física ---
GRAVITY = 9.8  # Aceleración de la gravedad (m/s^2)
TIME_STEP = 0.1 # Paso de tiempo de simulación (segundos). Cuanto más pequeño, más precisa la simulación.

# --- Clase del Entorno ---
class BasketballEnv(gym.Env):
    """
    Entorno de lanzamiento de baloncesto simplificado para Q-Learning.

    Características clave:
    - Acción: Continua (Ángulo, Velocidad).
    - Observación (Estado): Discreta (Posición de la Canasta).
    - Dinámica: Movimiento Parabólico 2D.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # 1. Espacios de Acción y Observación

        # --- Espacio de Acción (Input del Agente) ---
        # Definido como una caja (Box) de valores continuos (np.float32)
        # Índice 0: Ángulo (rad) en el rango [30 grados ($\pi/6$), 60 grados ($\pi/3$)]
        # Índice 1: Velocidad (m/s) en el rango [5 m/s, 15 m/s]
        low_action = np.array([np.pi/6, 5], dtype=np.float32)
        high_action = np.array([np.pi/3, 15], dtype=np.float32)
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        # --- Discretización del Estado (Output del Entorno) ---
        # El Q-Learning requiere estados discretos para la Q-Tabla.

        # 10 posibles posiciones para la coordenada X de la canasta (e.g., 5.0m a 10.0m)
        self.RIM_X_POSITIONS = np.linspace(5.0, 10.0, 10)
        # 10 posibles posiciones para la coordenada Y de la canasta (e.g., 2.5m a 5.0m)
        self.RIM_Y_POSITIONS = np.linspace(2.5, 5.0, 10)

        # El número total de estados es 10 * 10 = 100
        self.NUM_STATES = len(self.RIM_X_POSITIONS) * len(self.RIM_Y_POSITIONS)
        # El espacio de observación es un conjunto discreto de 100 índices (0 a 99)
        self.observation_space = spaces.Discrete(self.NUM_STATES)

        # 2. Variables de Simulación
        self.pos = np.array([0.0, 0.0], dtype=np.float32) # Posición de la pelota [x, y]
        self.vel = np.array([0.0, 0.0], dtype=np.float32) # Velocidad de la pelota [vx, vy]

        self.rim_x = 0.0 # Posición actual de la canasta (X)
        self.rim_y = 0.0 # Posición actual de la canasta (Y)

        self.max_timesteps = 30 # Número máximo de pasos de simulación por lanzamiento
        self.current_timestep = 0



    def _get_obs(self):
        """
        Calcula el único índice de estado discreto basado en la posición actual de la canasta.
        Este es el estado que el agente Q-Learning utiliza para indexar la Q-Tabla.
        """
        # Encuentra el índice (0-9) del valor X más cercano a la posición de canasta actual
        x_idx = np.argmin(np.abs(self.RIM_X_POSITIONS - self.rim_x))

        # Encuentra el índice (0-9) del valor Y más cercano a la posición de canasta actual
        y_idx = np.argmin(np.abs(self.RIM_Y_POSITIONS - self.rim_y))

        # Combina los dos índices en un único índice de estado discreto (0 a 99)
        # La fórmula es esencial para mapear el par (X, Y) a un único entero.
        state_index = x_idx * len(self.RIM_Y_POSITIONS) + y_idx

        return int(state_index)


    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno, elige una nueva posición aleatoria para la canasta
        y devuelve el estado inicial.
        """
        super().reset(seed=seed)

        # Reiniciar la pelota
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.current_timestep = 0

        # Establecer una nueva posición aleatoria para la canasta, eligiendo de la rejilla discreta
        rim_x_new = self.np_random.choice(self.RIM_X_POSITIONS)
        rim_y_new = self.np_random.choice(self.RIM_Y_POSITIONS)

        self.rim_x = float(rim_x_new)
        self.rim_y = float(rim_y_new)

        print(f" Nueva canasta en: X={self.rim_x:.2f}, Y={self.rim_y:.2f}")

        # Devuelve el estado inicial (índice de canasta) y la información de la canasta
        return self._get_obs(), {"rim_x": self.rim_x, "rim_y": self.rim_y, "scored": False}


    def step(self, action):
        """
        Ejecuta un paso de simulación con la acción continua (ángulo, velocidad) provista por el agente.
        """
        angle, speed = action

        # 1. Inicializar la velocidad (solo en el primer paso)
        # Descompone la velocidad inicial en componentes X y Y
        if np.all(self.pos == 0) and np.all(self.vel == 0):
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            self.vel = np.array([vx, vy], dtype=np.float32)

        # 2. Simulación de la física (Integración de Euler simple)
        # Actualiza la posición: $x_{t+1} = x_t + v_x \cdot \Delta t$
        self.pos[0] += self.vel[0] * TIME_STEP
        self.pos[1] += self.vel[1] * TIME_STEP

        # Actualiza la velocidad vertical: $v_{y, t+1} = v_{y, t} - g \cdot \Delta t$
        self.vel[1] -= GRAVITY * TIME_STEP

        self.current_timestep += 1

        # 3. Cálculo de Recompensa y Terminación
        reward = -0.01  # Penalización por paso (alienta a encestar rápido)
        done = False # Indica la terminación por éxito o fallo grave
        truncated = self.current_timestep >= self.max_timesteps # Indica terminación por límite de tiempo
        info = {"scored": False}

        # --- Condición de Fallo: Tocar el suelo ---
        if self.pos[1] <= 0:
            done = True
            reward -= 1.0 # Gran penalización por fallo

        # --- Condición de Éxito: Encestar ---
        DISTANCE_TOLERANCE = 0.5 # Tolerancia de distancia para encestar (simplificación)

        # Si la pelota está verticalmente a la altura de la canasta (con un pequeño margen)
        if self.pos[1] >= (self.rim_y - 0.1) and self.pos[1] <= (self.rim_y + 0.1):
            distance_to_rim = np.abs(self.pos[0] - self.rim_x)

            # Si horizontalmente está dentro de la tolerancia
            if distance_to_rim <= DISTANCE_TOLERANCE:
                reward += 10.0 # Gran recompensa por éxito
                info["scored"] = True
                done = True # Termina el episodio al encestar

        # El estado observado (índice de canasta) es siempre el mismo para este episodio
        observation = self._get_obs()

        # Retorna los 5 elementos requeridos por la API de Gymnasium
        return observation, reward, done, truncated, info


    def render(self):
        # Implementación de renderizado (vacía aquí, se asume que se maneja externamente)
        pass

    def close(self):
        # Cierre y limpieza de recursos (vacía aquí)
        pass