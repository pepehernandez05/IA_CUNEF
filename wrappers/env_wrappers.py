import gymnasium as gym # Importa la biblioteca base de entornos RL
import numpy as np # No se usa directamente aquí, pero es una buena práctica en scripts de RL

# La clase debe estar aquí, a nivel superior del módulo
# Hereda de gym.RewardWrapper, una clase base diseñada para modificar la señal de recompensa
class RewardShapingWrapper(gym.RewardWrapper):
    """
    Modifica la recompensa (Reward Shaping) para ayudar al aprendizaje.

    El Reward Shaping es una técnica para darle al agente señales adicionales o
    modificadas para que aprenda más rápido o evite un comportamiento no deseado.
    """
    def __init__(self, env):
        # Llama al constructor de la clase base (RewardWrapper), registrando el entorno envuelto
        super().__init__(env)

    def reward(self, reward):
        """
        Método obligatorio que se llama justo antes de que la recompensa se devuelva al agente.

        Parámetro:
            reward (float): La recompensa original devuelta por env.step().
        Retorno:
            float: La recompensa modificada.
        """
        # Ejemplo de Reward Shaping: Multiplica la recompensa original por 1.2.
        # Esto hace que las recompensas positivas (como encestar) sean más atractivas
        # y las penalizaciones negativas (como tocar el suelo) sean más fuertes.
        return reward * 1.2