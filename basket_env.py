import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- Constantes ---
GRAVITY = 9.8  # Aceleración de la gravedad (m/s^2)
TIME_STEP = 0.1 # Paso de tiempo de simulación (segundos)

# --- Clase del Entorno ---
class BasketballEnv(gym.Env):
    """
    Entorno de lanzamiento de baloncesto simplificado para Q-Learning.
    El estado se discretiza como la posición de la canasta (x, y) para el Q-Table.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        
        # 1. Espacios de Acción y Observación
        
        # Acción Continua (ángulo, velocidad)
        # Angulo (rad): [pi/6 (30 grados), pi/3 (60 grados)]
        # Velocidad (m/s): [5, 15]
        low_action = np.array([np.pi/6, 5], dtype=np.float32)
        high_action = np.array([np.pi/3, 15], dtype=np.float32)
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)
        
        # Observación Discreta: El estado es la canasta. Discretizamos el espacio.
        # Asumiendo 10 posibles posiciones en X y 10 en Y = 100 estados posibles.
        self.RIM_X_POSITIONS = np.linspace(5.0, 10.0, 10) # 10 posiciones de canasta en X
        self.RIM_Y_POSITIONS = np.linspace(2.5, 5.0, 10) # 10 posiciones de canasta en Y
        
        # El espacio de observación es el número total de estados discretos.
        self.NUM_STATES = len(self.RIM_X_POSITIONS) * len(self.RIM_Y_POSITIONS)
        self.observation_space = spaces.Discrete(self.NUM_STATES)

        # 2. Variables del Entorno
        self.pos = np.array([0.0, 0.0], dtype=np.float32) # Posición inicial de la pelota (x, y)
        self.vel = np.array([0.0, 0.0], dtype=np.float32) # Velocidad de la pelota (vx, vy)
        
        self.rim_x = 0.0 # Posición x de la canasta
        self.rim_y = 0.0 # Posición y de la canasta
        
        self.max_timesteps = 30 # Máximo de pasos para terminar el episodio
        self.current_timestep = 0


    
    def _get_obs(self):
        """
        Calcula el índice de estado discreto basado en la posición actual de la canasta.
        """
        # Encuentra el índice más cercano para la posición X de la canasta
        x_idx = np.argmin(np.abs(self.RIM_X_POSITIONS - self.rim_x))
        
        # Encuentra el índice más cercano para la posición Y de la canasta
        y_idx = np.argmin(np.abs(self.RIM_Y_POSITIONS - self.rim_y))
        
        # Combina los índices en un único índice de estado discreto
        # Formula: estado = x_idx * (Número de posiciones en Y) + y_idx
        state_index = x_idx * len(self.RIM_Y_POSITIONS) + y_idx
        
        return int(state_index)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reiniciar posición y velocidad de la pelota
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.current_timestep = 0
        
        # Establecer una nueva posición aleatoria para la canasta (dentro de los rangos predefinidos)
        # Esto asegura que el estado discreto devuelto por _get_obs será uno de los 100
        rim_x_new = self.np_random.choice(self.RIM_X_POSITIONS)
        rim_y_new = self.np_random.choice(self.RIM_Y_POSITIONS)
        
        self.rim_x = float(rim_x_new)
        self.rim_y = float(rim_y_new)
        
        print(f" Nueva canasta en: X={self.rim_x:.2f}, Y={self.rim_y:.2f}")

        # La línea con el error ahora funciona
        return self._get_obs(), {"rim_x": self.rim_x, "rim_y": self.rim_y, "scored": False} 


    def step(self, action):
        angle, speed = action
        
        # 1. Inicializar la velocidad
        # Solo se ejecuta en el primer paso del episodio (cuando la pelota está en [0, 0])
        if np.all(self.pos == 0) and np.all(self.vel == 0):
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            self.vel = np.array([vx, vy], dtype=np.float32)

        # 2. Simulación de la física (Movimiento Parabólico)
        self.pos[0] += self.vel[0] * TIME_STEP
        self.pos[1] += self.vel[1] * TIME_STEP
        
        # Actualizar la velocidad vertical debido a la gravedad
        self.vel[1] -= GRAVITY * TIME_STEP
        
        self.current_timestep += 1

        # 3. Cálculo de Recompensa y Terminación
        reward = -0.01  # Pequeña penalización por paso (animar a terminar rápido)
        done = False
        truncated = self.current_timestep >= self.max_timesteps
        info = {"scored": False}
        
        # Comprobar si tocó el suelo (Terminación)
        if self.pos[1] <= 0:
            done = True
            reward -= 1.0 # Gran penalización por tocar el suelo

        # Comprobar si encestó
        # Simplificamos: si la pelota está cerca de la canasta cuando está a su altura
        DISTANCE_TOLERANCE = 0.5 # Distancia de tolerancia para encestar (en metros)
        
        if self.pos[1] >= (self.rim_y - 0.1) and self.pos[1] <= (self.rim_y + 0.1): # Si está a la altura de la canasta
            distance_to_rim = np.abs(self.pos[0] - self.rim_x)
            
            if distance_to_rim <= DISTANCE_TOLERANCE:
                reward += 10.0 # Gran recompensa
                info["scored"] = True
                done = True # Termina el episodio al encestar
        
        # El estado de la canasta no cambia durante el episodio, por lo que el estado observado es el mismo.
        observation = self._get_obs() 
        
        return observation, reward, done, truncated, info


    def render(self):
        # El renderizado real se realiza en wrappers/render_utils.py
        pass 
        
    def close(self):
        # Cierre del entorno
        pass