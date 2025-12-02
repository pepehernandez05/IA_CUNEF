import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_logs(log_dir):
    """
    Carga el log manual (CSV) y grafica la recompensa promedio por episodio,
    guardando el resultado como una imagen PNG.
    
    Args:
        log_dir (str): Directorio donde se encuentra 'manual_log.csv'.
    """
    log_path = os.path.join(log_dir, "manual_log.csv")
    
    if not os.path.exists(log_path):
        print(f" Archivo de log no encontrado en: {log_path}")
        return

    try:
        # 1. Cargar datos
        df = pd.read_csv(log_path)
        
        # 2. Calcular el promedio móvil (Rolling Mean)
        WINDOW_SIZE = 100 # Promediar cada 100 episodios
        df['smoothed_reward'] = df['r'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        
        # 3. Crear el gráfico
        plt.figure(figsize=(10, 6))
        
        # Graficar la recompensa sin suavizar (opcional)
        plt.plot(df['t'], df['r'], color='lightgray', alpha=0.5, label='Recompensa (Episodio)')
        
        # Graficar la recompensa suavizada (la tendencia de aprendizaje)
        plt.plot(df['t'], df['smoothed_reward'], color='blue', linewidth=2, label=f'Recompensa Promedio Móvil ({WINDOW_SIZE} episodios)')
        
        plt.xlabel('Timesteps Globales')
        plt.ylabel('Recompensa del Episodio')
        plt.title('Curva de Aprendizaje del Agente Q-Learning')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 4. Guardar el gráfico en lugar de mostrarlo
        output_path = os.path.join(log_dir, "curva_aprendizaje.png")
        plt.savefig(output_path)
        print(f" Gráfico de aprendizaje guardado en: {output_path}")
        # Asegúrate de cerrar la figura para liberar memoria
        plt.close() 

    except Exception as e:
        print(f" Error al procesar o graficar el archivo de log: {e}")

if __name__ == "__main__":
    plot_logs("./logs/")