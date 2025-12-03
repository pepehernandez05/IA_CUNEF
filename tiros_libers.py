import numpy as np
import random
import matplotlib.pyplot as plt
import time
import matplotlib

# Configuración del backend de Matplotlib para evitar la advertencia 'non-interactive'
# Usamos 'Agg' para generar archivos de imagen sin depender de una GUI.
matplotlib.use('Agg') 

# --- 1. Definición del Entorno ---
# [Fuerza Baja, Fuerza Media, Fuerza Alta]
REAL_SUCCESS_PROBABILITIES = [0.4, 0.7, 0.5] 
N_ACTIONS = len(REAL_SUCCESS_PROBABILITIES)
N_STATES = 1

ACTION_NAMES = ["Fuerza Baja", "Fuerza Media", "Fuerza Alta"]

def get_reward(action_index):
    """Simula el resultado de un tiro libre dada la acción elegida."""
    prob_success = REAL_SUCCESS_PROBABILITIES[action_index]
    is_success = random.random() < prob_success
    # Devuelve la recompensa (1 o 0) y el resultado como texto
    return 1 if is_success else 0, "ACIERTO" if is_success else "FALLO"

# --- 2. Parámetros de Q-Learning ---
EPISODES = 2000          # Número de intentos para que el agente aprenda bien
LEARNING_RATE = 0.1      # Tasa de aprendizaje (alfa)
DISCOUNT_FACTOR = 0.9    # Factor de descuento (gamma)
EXPLORATION_RATE = 1.0   # Tasa de exploración inicial (epsilon)
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.005 # Tasa a la que disminuye la exploración

# --- 3. Inicialización de la Q-Tabla y el Historial ---
# La Q-Tabla es una matriz de [Estados x Acciones]
Q_TABLE = np.zeros((N_STATES, N_ACTIONS))

# Listas para el tracking del aprendizaje (para la gráfica)
rewards_per_episode = []
q_values_history = {action: [] for action in ACTION_NAMES} 

# --- 4. El Bucle de Entrenamiento (Learning Loop) ---
exploration_rate = EXPLORATION_RATE

print("Iniciando entrenamiento del agente (Tiros Libres)...")
print("-" * 50)

for episode in range(EPISODES):
    state = 0
    
    # a) Estrategia de Selección de Acción (Exploración vs. Explotación)
    is_exploring = random.random() < exploration_rate
    
    if is_exploring:
        action = random.randint(0, N_ACTIONS - 1)
        strategy = "Exploración"
    else:
        # Explotación: elige la acción con el valor Q más alto (greedy)
        action = np.argmax(Q_TABLE[state, :])
        strategy = "Explotación"
    
    # b) Ejecutar la Acción y Observar Recompensa
    reward, result_text = get_reward(action)
    
    # c) Q-Learning: Actualizar la Q-Tabla
    old_q_value = Q_TABLE[state, action]
    # En este caso simple, el estado siguiente es el mismo estado actual
    future_max_q = np.max(Q_TABLE[state, :])
    
    # Fórmula de Actualización de Q-Learning:
    new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * future_max_q - old_q_value)
    Q_TABLE[state, action] = new_q_value
    
    # d) Disminuir la Tasa de Exploración (Decay)
    exploration_rate = MIN_EXPLORATION_RATE + (EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY_RATE * episode)

    # Guardar datos para la gráfica
    rewards_per_episode.append(reward)
    for i, name in enumerate(ACTION_NAMES):
        q_values_history[name].append(Q_TABLE[state, i])

    # e) Reporte detallado por intento (mostramos el inicio y luego reportes periódicos)
    if (episode + 1) % 100 == 0 or episode < 5: 
        print(f"INTENTO {episode + 1}: {result_text}")
        print(f"  -> Acción: {ACTION_NAMES[action]} ({strategy}) | Recompensa: {reward} | Q-Valor Nuevo: {new_q_value:.4f}")
        print(f"  -> Tasa Exploración: {exploration_rate:.4f}")
        print("-" * 50)
        
    # time.sleep(0.001) # Descomentar para una pausa ligera durante el output

print("\nEntrenamiento finalizado.")

# --- 5. Resultados Finales ---

print("\n##  Resultados Finales de la Q-Tabla ##")
# La acción con el Q-valor más alto es la acción óptima aprendida
optimal_action_index = np.argmax(Q_TABLE[0])
optimal_action_name = ACTION_NAMES[optimal_action_index]
optimal_action_value = Q_TABLE[0, optimal_action_index]

print(f"  Q-Tabla (Estado 0): {Q_TABLE[0]}")
print("\n--- Conclusión ---")
print(f"La acción óptima aprendida es: **{optimal_action_name}**")
print(f"Con un Q-Valor final de: **{optimal_action_value:.4f}**")

# --- 6. Visualización de la Función de Aprendizaje con Matplotlib (Guardado en PNG) ---

## Gráfico 1: Porcentaje de Aciertos Promedio (Rolling Average)
WINDOW_SIZE = 50 
# Calcula el promedio móvil de las recompensas
rolling_avg_rewards = np.convolve(rewards_per_episode, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')

plt.figure(figsize=(12, 6))
plt.plot(range(len(rolling_avg_rewards)), rolling_avg_rewards, label='Porcentaje de Aciertos Promedio')
plt.axhline(y=max(REAL_SUCCESS_PROBABILITIES), color='r', linestyle='--', label='Probabilidad Máxima Real (0.7)')
plt.title('Función de Aprendizaje: Porcentaje de Aciertos Promedio')
plt.xlabel(f'Episodios (ventana de {WINDOW_SIZE})')
plt.ylabel('Porcentaje de Aciertos')
plt.grid(True)
plt.legend()
plt.savefig('porcentaje_aciertos.png') # Guarda el gráfico como archivo
plt.close()

## Gráfico 2: Evolución de los Q-Valores para cada acción
plt.figure(figsize=(12, 6))
for action_name, q_values in q_values_history.items():
    plt.plot(q_values, label=f'Q-Valor ({action_name})')

plt.title('Evolución de los Q-Valores por Acción')
plt.xlabel('Episodios')
plt.ylabel('Valor Q')
plt.grid(True)
plt.legend()
plt.savefig('evolucion_q_valores.png') # Guarda el gráfico como archivo
plt.close()

print("\n Gráficos generados")
print("   - porcentaje_aciertos.png")
print("   - evolucion_q_valores.png")