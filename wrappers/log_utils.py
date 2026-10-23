import pandas as pd # Biblioteca para el análisis y manejo de datos (crucial para leer CSV y calcular el promedio móvil)
import matplotlib.pyplot as plt # Biblioteca para la generación de gráficos
import os # Para la manipulación de rutas de archivos

def plot_logs(log_dir):
    """
    Carga el log manual (CSV) generado durante el entrenamiento de RL y
    grafica la recompensa promedio por episodio, guardando el resultado como una imagen PNG.

    Args:
        log_dir (str): Directorio donde se espera encontrar el archivo 'manual_log.csv'.
    """
    # 1. Validación y Carga del Archivo
    log_path = os.path.join(log_dir, "manual_log.csv")

    if not os.path.exists(log_path):
        print(f" Archivo de log no encontrado en: {log_path}")
        return

    try:
        # Carga el archivo CSV en un DataFrame de Pandas
        df = pd.read_csv(log_path)

        # 2. Calcular el promedio móvil (Rolling Mean)
        # Esto es vital para suavizar el ruido en las recompensas por episodio y ver la tendencia real del aprendizaje.
        WINDOW_SIZE = 100 # Promedia cada 100 episodios
        # Calcula el promedio de la columna 'r' (recompensa) usando una ventana deslizante
        df['smoothed_reward'] = df['r'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

        # 3. Crear el gráfico
        plt.figure(figsize=(10, 6))

        # Graficar la recompensa sin suavizar (la recompensa cruda de cada episodio)
        plt.plot(df['t'], df['r'], color='lightgray', alpha=0.5, label='Recompensa (Episodio)')

        # Graficar la recompensa suavizada (la tendencia de aprendizaje)
        plt.plot(df['t'], df['smoothed_reward'], color='blue', linewidth=2, label=f'Recompensa Promedio Móvil ({WINDOW_SIZE} episodios)')

        # Configuración de Etiquetas y Título
        plt.xlabel('Timesteps Globales') # El eje X típicamente usa el paso de tiempo global ('t' en el CSV)
        plt.ylabel('Recompensa del Episodio')
        plt.title('Curva de Aprendizaje del Agente Q-Learning')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # 4. Guardar y Liberar Memoria
        output_path = os.path.join(log_dir, "curva_aprendizaje.png")
        plt.savefig(output_path) # Guarda el gráfico como una imagen PNG
        print(f" Gráfico de aprendizaje guardado en: {output_path}")
        # Cierra la figura para liberar la memoria utilizada por Matplotlib
        plt.close()

    except Exception as e:
        print(f" Error al procesar o graficar el archivo de log: {e}")

if __name__ == "__main__":
    # Ejecuta la función principal si el script se llama directamente
    plot_logs("./logs/")