# Q-Learning para el Tiro Libre de Baloncesto

Este proyecto implementa y visualiza un agente de **Reinforcement Learning (RL)** que aprende la combinación óptima de **ángulo y velocidad** para encestar un tiro libre en un entorno de simulación 2D.

El agente utiliza el algoritmo **Q-Learning tabular** y aborda el desafío de un espacio de acción continuo mediante la técnica de **discretización de acciones**. 

---

## Arquitectura del Proyecto

El sistema se compone de un entorno tipo **Gymnasium** y un agente de Q-Learning, complementado con *wrappers* para la gestión de recompensas y logs.

### Componentes Clave

| Archivo/Módulo | Rol Principal | Concepto de RL |
| :--- | :--- | :--- |
| `basket_env.py` | **Entorno (Gymnasium)**. Define la física parabólica 2D y un **espacio de observación discreto** (posición de la canasta). | Espacio de Observación Discreto |
| `agente.py` | **Agente Q-Learning Tabular**. Implementa el algoritmo Q-Learning y la política $\epsilon$-greedy. Contiene la lógica de **Discretización de Acciones**. | Q-Learning, Discretización |
| `wrappers/env_wrappers.py` | **`RewardShapingWrapper`**. Modifica la recompensa original del entorno (escalando por 1.2) para acelerar el aprendizaje. | Reward Shaping |
| `wrappers/log_utils.py` | **`plot_logs`**. Lee el log (`manual_log.csv`) y genera el gráfico de la **curva de aprendizaje suavizada** usando el promedio móvil de Pandas. | Monitorización de Aprendizaje |
| `ejecutar_agente.py` | **Visualizador / Inferencia**. Carga el modelo entrenado, lo ejecuta en modo de explotación y usa **Pygame** para el renderizado en tiempo real. | Inferencia, Pygame Renderizado |

---

## Cómo Empezar

### Requisitos

Necesitará Python 3.8+ y las siguientes librerías:

 ## pip install numpy pandas matplotlib gymnasium pygame

1. Entrenamiento del Agente

Para entrenar el modelo, ejecute el script del agente (que contiene el bucle de entrenamiento y logging):
## agente.py
El agente se entrenará durante 500,000 pasos (total_timesteps).

    Al finalizar, se guardará la Q-Tabla en ./best_model/basket_agent_q_table.npy y la curva de aprendizaje en ./logs/curva_aprendizaje.png.

2. Visualización de la Curva de Aprendizaje

Si desea regenerar o ver la curva de aprendizaje, puede ejecutar el módulo de logs (asumiendo que tiene datos en ./logs/manual_log.csv):
## python -c "from wrappers.log_utils import plot_logs; plot_logs('./logs/')"

3. Ejecución del Agente Entrenado (Inferencia)

Para ver al agente en acción, cargue la Q-Tabla guardada y ejecute el visualizador de Pygame:
## ejecutar_agente.py

El agente se ejecutará en modo determinista (ϵ=0.0), siempre seleccionando la acción con el valor Q más alto para la posición actual de la canasta.

Detalles Técnicos de Implementación:
1. Discretización de Acciones

El entorno tiene un espacio de acción continuo (ángulo y velocidad). El agente utiliza la discretización para crear una cuadrícula de 5×5=25 acciones posibles.

Esto permite que el Q-Learning tabular se aplique usando una Q-Tabla con dimensiones:
[100 Estados×25 Acciones]

    Estados (100): Mapean 10 posiciones X de la canasta × 10 posiciones Y de la canasta.

    Acciones (25): Mapean 5 niveles de ángulo × 5 niveles de velocidad.

2. Renderizado con Pygame

El módulo PygameRenderer es esencial para la visualización. Traduce las coordenadas físicas del entorno (metros) a píxeles en la pantalla, utilizando un factor de escala de 60 píxeles por metro y asegurando que el origen de las coordenadas se fije en el punto de lanzamiento (esquina inferior izquierda de la simulación).
