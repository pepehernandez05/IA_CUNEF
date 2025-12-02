import gymnasium as gym
import numpy as np

# La clase debe estar aquí, a nivel superior del módulo
class RewardShapingWrapper(gym.RewardWrapper): 
    """
    Modifica la recompensa para ayudar al aprendizaje.
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # Ejemplo: aumentar un poco la recompensa
        return reward * 1.2