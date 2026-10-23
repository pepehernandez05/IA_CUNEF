# Q-Learning para el Tiro Libre de Baloncesto

Agente de **Reinforcement Learning** que aprende qué **ángulo y velocidad** de lanzamiento usar para encestar un tiro libre en una simulación 2D, con la canasta en posiciones variables.

Usa **Q-Learning tabular**. Como el espacio de acción es continuo (ángulo y velocidad), se resuelve con **discretización de acciones**: una rejilla de 15 × 15 = 225 acciones posibles.

Proyecto desarrollado en CUNEF Universidad (Ingeniería Informática) por José Ignacio Hernández Pablos y Rodrigo López Rodríguez.

---

## Estructura del proyecto

```
IA_CUNEF/
├── basket_env.py          # Entorno Gymnasium (física parabólica 2D)
├── agente.py              # Agente Q-Learning + bucle de entrenamiento
├── evaluar.py             # Evaluación: modelo vs. baseline aleatorio vs. techo de la rejilla
├── ejecutar_agente.py     # Inferencia con visualización en Pygame
├── requirements.txt
├── V2.py                  # <!-- COMPLETAR: qué cambia respecto a agente.py -->
├── tiros_libers.py        # <!-- COMPLETAR: qué hace -->
└── wrappers/
    ├── __init__.py        # (archivo vacío, necesario para los imports)
    ├── env_wrappers.py    # RewardShapingWrapper
    ├── log_utils.py       # plot_logs: curva de aprendizaje
    └── render_utils.py    # PygameRenderer
```

### Componentes

| Módulo | Rol |
|:---|:---|
| `basket_env.py` | Entorno tipo Gymnasium. Acción continua (ángulo, velocidad) y **observación discreta** de 100 estados, que codifican la posición de la canasta. Un episodio es un lanzamiento. |
| `agente.py` | `QLearningAgent`: Q-Tabla, política ε-greedy y discretización de acciones. `train_agent()` entrena, guarda el modelo y dibuja la curva de aprendizaje. |
| `wrappers/env_wrappers.py` | `RewardShapingWrapper`: penaliza los fallos según lo cerca que pasó la pelota del aro. |
| `wrappers/log_utils.py` | `plot_logs`: lee `logs/manual_log.csv` y dibuja recompensa y tasa de acierto con media móvil (Pandas). |
| `wrappers/render_utils.py` | `PygameRenderer`: convierte metros a píxeles (60 px/m) con el origen en el punto de lanzamiento. |
| `evaluar.py` | Mide la tasa de acierto del modelo, de una política aleatoria y el máximo alcanzable con la rejilla de acciones. |
| `ejecutar_agente.py` | Carga la Q-Tabla y ejecuta al agente en modo determinista (ε = 0) con Pygame. |

---

## Cómo empezar

### Requisitos

Python 3.8 o superior.

```bash
pip install -r requirements.txt
```

### 1. Entrenar el agente

```bash
python agente.py
```

Entrena durante 60.000 lanzamientos (unos segundos en un portátil). Al terminar guarda:

- la Q-Tabla en `./best_model/basket_agent_q_table.npy`
- el log en `./logs/manual_log.csv`
- la curva de aprendizaje en `./logs/curva_aprendizaje.png`

### 2. Evaluar el modelo

```bash
python evaluar.py
```

Lanza 500 tiros con la canasta en posiciones aleatorias y compara el modelo entrenado con una política aleatoria y con el techo teórico de la rejilla.

### 3. Ver al agente entrenado

```bash
python ejecutar_agente.py
```

Abre la ventana de Pygame y lanza tiros indefinidamente con una canasta nueva en cada uno, mostrando el marcador de aciertos. Si no encuentra el modelo, se ejecuta con una tabla vacía, sin entrenar.

### 4. Regenerar la curva de aprendizaje

```bash
python -c "from wrappers.log_utils import plot_logs; plot_logs('./logs/')"
```

---

## Detalles técnicos

### Entorno

| Elemento | Valor |
|:---|:---|
| Observación | `Discrete(100)`: 10 posiciones X (5,0–10,0 m) × 10 posiciones Y (2,5–5,0 m) de la canasta |
| Acción | `Box`: ángulo en [π/6, π/3] rad y velocidad en [5, 15] m/s. Solo cuenta la acción del primer paso. |
| Física | Parábola exacta con g = 9,8 m/s², muestreada cada Δt = 0,1 s (máximo 30 pasos) |
| Enceste | La pelota cruza hacia abajo la altura del aro a ≤ 0,5 m en horizontal. El punto de cruce se interpola entre dos pasos. |

### Recompensas

| Evento | Recompensa |
|:---|:---:|
| Enceste | +10,0 |
| Tocar el suelo | −1,0 |
| Cada paso | −0,01 |
| Fallo (reward shaping) | −0,1 × distancia mínima pelota–aro (m) |

### Discretización de acciones

Se toman 15 niveles de ángulo y 15 de velocidad, espaciados linealmente dentro de los límites del entorno. El índice discreto `k` (0–224) se convierte así:

```
ángulo    = niveles_ángulo[k // 15]
velocidad = niveles_velocidad[k % 15]
```

La Q-Tabla tiene dimensiones **100 estados × 225 acciones**.

### Algoritmo

Cada episodio es un único lanzamiento: no hay estado siguiente, así que la actualización no lleva término de descuento:

```
Q(s,a) ← Q(s,a) + α · (R − Q(s,a))
```

con R la recompensa total del lanzamiento. Política ε-greedy con desempate aleatorio entre acciones de igual valor.

| Parámetro | Valor |
|:---|:---:|
| Tasa de aprendizaje (α) | 0,2 |
| ε inicial / mínimo | 1,0 / 0,01 |
| Decaimiento de ε (por episodio) | 0,9998 |
| Episodios de entrenamiento | 60.000 |

---

## Resultados

Evaluación con 500 lanzamientos, canasta aleatoria y política determinista (ε = 0), repetida con 5 semillas de entrenamiento distintas:

| Métrica | Valor |
|:---|:---:|
| Acierto del modelo entrenado | 79,2 – 84,6 % (media 82,3 %) |
| Acierto de una política aleatoria | ≈ 1,6 % |
| Estados con al menos una acción ganadora en la rejilla | 83 de 100 |

En todas las semillas el modelo iguala el máximo que permite la rejilla de acciones, es decir, acierta en todos los estados en los que existe alguna acción que enceste. Los 17 estados restantes no tienen ninguna acción ganadora entre las 225; algunos son físicamente inalcanzables con los rangos de ángulo y velocidad (por ejemplo, una canasta muy alta y muy cercana). Una rejilla más fina (`N_ANGLES` y `N_SPEEDS` en `agente.py`) cubre más estados, pero necesita más episodios para converger.

## Trabajo futuro

- DQN u otro método para trabajar directamente con acciones continuas.
- Permitir cambiar la acción durante el vuelo (por ejemplo, un modelo con efecto o rebote en el aro).
- Evaluar con más semillas y reportar la desviación estándar entre entrenamientos.
e 60 píxeles por metro y asegurando que el origen de las coordenadas se fije en el punto de lanzamiento (esquina inferior izquierda de la simulación).
