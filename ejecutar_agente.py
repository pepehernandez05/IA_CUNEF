# ejecutar_agente.py

import pygame # Biblioteca para crear juegos y visualizaciones (usada para renderizar)
import numpy as np # Para manejo de arrays y la Q-Tabla
import os # Para verificar la existencia del archivo del modelo
# Importaciones de los módulos personalizados (entorno, renderizado, wrappers y agente)
from basket_env import BasketballEnv
from wrappers.render_utils import PygameRenderer
from wrappers.env_wrappers import RewardShapingWrapper
from agente import QLearningAgent

def run_agent(model_path="./best_model/basket_agent_q_table.npy"):
    """
    Función principal para cargar y ejecutar al agente Q-Learning en modo de explotación.
    """
    pygame.init() # Inicializa todos los módulos de Pygame

    # 1. Configuración del Entorno
    env = BasketballEnv() # Inicializa el entorno base de baloncesto
    env = RewardShapingWrapper(env) # Aplica el wrapper para usar la misma configuración que en el entrenamiento

    # 2. Inicializar y Cargar el Modelo

    # Inicializa una instancia del agente, creando una Q-Tabla vacía por defecto
    model = QLearningAgent(env)

    # Cargar la tabla Q guardada desde el archivo .npy
    if os.path.exists(model_path):
        model.q_table = np.load(model_path)
        model.epsilon = 0.0 # Establece la exploración a cero (modo determinista / explotación)
        print(f"Tabla Q cargada desde {model_path}.")
    else:
        print(f" Archivo {model_path} no encontrado. Ejecutando agente sin entrenamiento.")

    # Reinicio inicial del entorno
    obs, info = env.reset()
    state = obs # El estado inicial es el índice de la canasta
    traj = [] # Lista para almacenar la trayectoria de la pelota (para renderizado)

    # 3. Inicializar el Renderizador
    renderer = PygameRenderer(env.unwrapped) # El renderer necesita el entorno sin wrappers

    # Bandera de control: Indica si el agente encestó en el paso anterior.
    # Se usa para forzar un reset y un cambio de canasta.
    scored_in_last_step = False

    # 4. Bucle Principal de Simulación
    running = True
    while running:
        # Manejo de Eventos de Pygame (Permite cerrar la ventana)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Lógica de Reset y Cambio de Canasta ---
        if scored_in_last_step:
            # Si acaba de encestar, reinicia el entorno (la canasta cambia de posición)
            obs, info = env.reset()
            state = obs
            traj = [] # Limpia la trayectoria para la nueva simulación
            scored_in_last_step = False # Desactiva la bandera
            continue # Salta al inicio del bucle para que el agente elija la acción para la nueva canasta

        # --- Selección de Acción ---
        # Predecir acción en modo determinista (deterministic=True)
        # El agente siempre elige la acción con el Q-Valor más alto (Explotación pura)
        action, _ = model.predict(state, deterministic=True)

        # --- Ejecución del Paso ---
        # Ejecuta la acción continua (ángulo, velocidad) en el entorno
        obs, reward, done, truncated, info = env.step(action)

        state = obs
        # Guarda la posición actual de la pelota para dibujar la trayectoria
        traj.append(env.unwrapped.pos.copy())

        # --- Lógica de Terminación del Episodio ---

        # 3. COMPROBACIÓN DE ENCESTE
        if info.get("scored", False):
            # Si encesta, activa la bandera. El reset se ejecuta en el siguiente ciclo.
            print(" Canasta. La canasta se moverá en el siguiente ciclo.")
            scored_in_last_step = True

        elif done or truncated:
            # Si el episodio termina por fallo (suelo) o timeout
            # Reiniciamos inmediatamente, manteniendo la misma canasta para que el agente lo intente de nuevo
            obs, info = env.reset()
            state = obs
            traj = []

        # Renderizar el estado actual del entorno y la trayectoria acumulada
        renderer.draw(traj)

    pygame.quit() # Cierra Pygame al terminar la simulación

if __name__ == "__main__":
    run_agent()