import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import csv
import time

# Configuración del backend de Matplotlib para generar PNG sin interfaz gráfica
matplotlib.use('Agg') 

# --- 0. CONFIGURACIÓN Y UTILIDADES DE LOGGING ---

LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "manual_log.csv")
WINDOW_SIZE = 100 # Ventana para el promedio móvil de la gráfica

def setup_logging():
    """Crea el directorio de logs y el archivo CSV con encabezados."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # t = timestep, r = reward, q = max Q-value
        writer.writerow(['t', 'r', 'q'])

def plot_logs():
    """Carga el log y grafica la curva de aprendizaje."""
    if not os.path.exists(LOG_FILE):
        print(f"Archivo de log no encontrado en: {LOG_FILE}")
        return

    try:
        df = pd.read_csv(LOG_FILE)
        
        # Calcular el promedio móvil
        df['smoothed_reward'] = df['r'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        
        plt.figure(figsize=(10, 6))
        
        # Graficar la recompensa sin suavizar
        plt.plot(df['t'], df['r'], color='lightgray', alpha=0.5, label='Recompensa (Episodio)')
        
        # Graficar la recompensa suavizada (tendencia de aprendizaje)
        plt.plot(df['t'], df['smoothed_reward'], color='blue', linewidth=2, label=f'Recompensa Promedio Móvil ({WINDOW_SIZE} episodios)')
        
        plt.xlabel('Timesteps Globales')
        plt.ylabel('Recompensa del Episodio')
        plt.title('Curva de Aprendizaje del Agente Q-Learning (Tiros Libres)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        output_path = os.path.join(LOG_DIR, "curva_aprendizaje.png")
        plt.savefig(output_path)
        print(f"\n Gráfico de aprendizaje guardado en: {output_path}")
        plt.close() 

    except Exception as e:
        print(f"Error al procesar o graficar el archivo de log: {e}")

# --- 1. ENTORNOS Y WRAPPERS ---

class ManualEnvironment:
    """Clase base que simula el entorno de Tiro Libre."""
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.action_names = ["Fuerza Baja", "Fuerza Media", "Fuerza Alta"]

    def step(self, action_index):
        """Ejecuta la acción y devuelve (recompensa, resultado_texto)."""
        prob_success = self.probabilities[action_index]
        is_success = random.random() < prob_success
        reward = 1 if is_success else 0
        result_text = "ACIERTO" if is_success else "FALLO"
        return reward, result_text
    
class ActionMappingWrapper:
    """Wrapper para reasignar las acciones (simulando una capa de abstracción)."""
    def __init__(self, env, action_map):
        self.env = env
        self.action_map = action_map 

    def step(self, agent_action_index):
        # Mapea la acción del agente a la acción real del entorno
        actual_env_action = self.action_map[agent_action_index]
        return self.env.step(actual_env_action)

class LoggingWrapper:
    """
    Wrapper para registrar datos relevantes (timestep, recompensa, max Q-value) 
    en un archivo CSV después de cada paso (step).
    """
    def __init__(self, env, log_file):
        self.env = env
        self.log_file = log_file
        self.timestep = 0

    def step(self, agent_action_index, max_q_value):
        """Ejecuta el paso del entorno y luego registra los datos."""
        
        # 1. Ejecuta el paso del entorno envuelto
        reward, result_text = self.env.step(agent_action_index)
        
        # 2. Registra los datos
        self.timestep += 1
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.timestep, reward, max_q_value])

        return reward, result_text

# --- 2. IMPLEMENTACIÓN Y PARÁMETROS ---

# Parámetros del Entorno
# La acción 1 (Fuerza Media) es la óptima.
REAL_SUCCESS_PROBABILITIES = [0.4, 0.7, 0.5] 
N_ACTIONS = len(REAL_SUCCESS_PROBABILITIES)
ACTION_NAMES = ["Fuerza Baja", "Fuerza Media", "Fuerza Alta"]

# Parámetros de Q-Learning
EPISODES = 1000          
LEARNING_RATE = 0.1     
DISCOUNT_FACTOR = 0.9   
EXPLORATION_RATE = 1.0  
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.01 # Exploración más rápida para 1000 episodios

# Inicialización
Q_TABLE = np.zeros((1, N_ACTIONS)) 
exploration_rate = EXPLORATION_RATE

# Inicializar el entorno base
base_env = ManualEnvironment(REAL_SUCCESS_PROBABILITIES)

# Aplicar los wrappers
# Usamos un mapeo simple: 0->0, 1->1, 2->2
mapped_env = ActionMappingWrapper(base_env, action_map=[0, 1, 2]) 
# Creamos el wrapper de logging sobre el entorno mapeado
wrapped_env = LoggingWrapper(mapped_env, LOG_FILE)

# --- 3. BUCLE DE ENTRENAMIENTO ---

setup_logging()
print("Iniciando simulación de Tiros Libres...")
print(f"Se simularán {EPISODES} intentos.")
print("-" * 50)

for episode in range(EPISODES):
    state = 0
    
    # a) Estrategia de Selección de Acción
    is_exploring = random.random() < exploration_rate
    
    if is_exploring:
        action = random.randint(0, N_ACTIONS - 1)
        strategy = "Exploración"
    else:
        action = np.argmax(Q_TABLE[state, :])
        strategy = "Explotación"
    
    # b) Ejecutar la Acción ENVUELTA y Observar Recompensa
    future_max_q = np.max(Q_TABLE[state, :]) 
    reward, result_text = wrapped_env.step(action, future_max_q)
    
    # c) Q-Learning: Actualizar la Q-Tabla
    old_q_value = Q_TABLE[state, action]
    new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * future_max_q - old_q_value)
    Q_TABLE[state, action] = new_q_value
    
    # d) Disminuir la Tasa de Exploración
    exploration_rate = MIN_EXPLORATION_RATE + (EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY_RATE * episode)

    # e) Reporte SIMPLE (Formato solicitado)
    print(f"INTENTO {episode + 1} : {result_text}")
    
    if (episode + 1) % 100 == 0:
        print("-" * 50)
        
print("\nEntrenamiento finalizado.")

# --- 4. RESULTADOS Y VISUALIZACIÓN ---

# Generar el gráfico de la curva de aprendizaje usando los datos logueados
plot_logs() 

print("\n##  Resultados Finales de la Q-Tabla ##")
optimal_action_index = np.argmax(Q_TABLE[0])
optimal_action_name = ACTION_NAMES[optimal_action_index]
optimal_action_value = Q_TABLE[0, optimal_action_index]
