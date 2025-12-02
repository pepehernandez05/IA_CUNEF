# ejecutar_agente.py

import pygame
import numpy as np
import os
from basket_env import BasketballEnv
from wrappers.render_utils import PygameRenderer
from wrappers.env_wrappers import RewardShapingWrapper
from agente import QLearningAgent

def run_agent(model_path="./best_model/basket_agent_q_table.npy"):
    pygame.init()
    
    env = BasketballEnv()
    env = RewardShapingWrapper(env)

    # 1. Inicializar el agente (sin cambios)
    model = QLearningAgent(env) 
    
    # 2. Cargar la tabla Q guardada (sin cambios)
    if os.path.exists(model_path):
        model.q_table = np.load(model_path)
        model.epsilon = 0.0 
        print(f"Tabla Q cargada desde {model_path}.")
    else:
        print(f" Archivo {model_path} no encontrado. Ejecutando agente sin entrenamiento.")
        
    obs, info = env.reset() # Capturamos 'info' en el reset inicial
    state = obs
    traj = []

    renderer = PygameRenderer(env.unwrapped) 
    
    # NUEVO: Bandera para saber si el agente acaba de anotar
    scored_in_last_step = False 

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Si el agente acaba de anotar en el paso anterior, reiniciamos el entorno para cambiar la canasta
        if scored_in_last_step:
            obs, info = env.reset() # Esto genera una nueva posición de canasta
            state = obs
            traj = []
            scored_in_last_step = False # Resetear la bandera
            continue # Saltamos al inicio del bucle para que el agente piense en la nueva canasta

        # Predecir acción en modo determinista (explotación)
        action, _ = model.predict(state, deterministic=True) 
        
        # Ejecutar paso
        obs, reward, done, truncated, info = env.step(action)
        
        state = obs
        traj.append(env.unwrapped.pos.copy())
        
        # 3. COMPROBACIÓN DE ENCESTE
        if info.get("scored", False):
            # Si se encesta, la bandera se activa, pero el reset se hace en la siguiente iteración
            print(" Canasta, La canasta se moverá en el siguiente ciclo.")
            scored_in_last_step = True
            
        elif done or truncated:
            # Si el episodio termina por tocar el suelo o timeout, reiniciamos sin mover la canasta (se moverá al inicio del siguiente episodio)
            obs, info = env.reset()
            state = obs
            traj = []

        renderer.draw(traj)

    pygame.quit()

if __name__ == "__main__":
    run_agent()