import pygame # Importa la biblioteca Pygame para gráficos y simulación

class PygameRenderer:
    """
    Maneja todo el renderizado en Pygame, dibujando el entorno de baloncesto
    y la trayectoria de la pelota.
    """
    def __init__(self, env, width=1000, height=600):
        # El entorno (BasketballEnv, sin wrappers) cuyos estados se van a dibujar
        self.env = env
        self.WIDTH = width     # Ancho de la ventana en píxeles
        self.HEIGHT = height   # Alto de la ventana en píxeles
        self.M2PX = 60         # Factor de conversión: Metros a Píxeles (60 píxeles por metro)
        self._init_pygame()

    def _init_pygame(self):
        """Inicializa Pygame y configura la ventana."""
        pygame.init()
        # Crea la ventana (surface) principal
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Agente RL - Tiro a Canasta")
        # Objeto para controlar la velocidad de la animación (FPS)
        self.clock = pygame.time.Clock()
        # Fuente para dibujar texto en pantalla
        self.font = pygame.font.SysFont(None, 26)

    def to_px(self, pos):
        """
        Convierte una posición de coordenadas físicas [x, y] (en metros)
        a coordenadas de píxeles (x_px, y_px) en la pantalla de Pygame.

        Nota: Pygame tiene el origen (0,0) en la esquina superior izquierda.
        El sistema convierte el origen a la esquina inferior izquierda (el punto de lanzamiento).
        """
        # Traslación en X: 100px de margen + coordenada X en metros * factor de escala
        x = int(100 + pos[0] * self.M2PX)
        # Traslación en Y: Altura total - 100px de margen - coordenada Y en metros * factor de escala
        y = int(self.HEIGHT - 100 - pos[1] * self.M2PX)
        return (x, y)

    def draw(self, traj):
        """
        Dibuja todos los elementos del entorno en el frame actual.

        Args:
            traj (list): Lista de puntos de la trayectoria (posiciones [x, y] en metros)
        """
        # Rellena la pantalla con un color de fondo (cielo azul claro)
        self.screen.fill((230, 240, 255))

        # Dibujar suelo (rectángulo grande en la parte inferior)
        pygame.draw.rect(self.screen, (60,160,60),
                         (0, self.HEIGHT-100, self.WIDTH, 100))

        # Dibujar pelota (círculo)
        # Usa self.env.pos, que es la posición actual de la pelota en metros
        pygame.draw.circle(self.screen, (255,165,0),
                           self.to_px(self.env.pos), 8)

        # Dibujar canasta (círculo rojo y delgado)
        # Usa self.env.rim_x y self.env.rim_y (la posición de la canasta)
        pygame.draw.circle(
            self.screen, (220,30,30),
            self.to_px([self.env.rim_x, self.env.rim_y]), 12, 3
        )

        # Dibujar trayectoria (líneas)
        if len(traj) > 1:
            # Convierte todos los puntos de la trayectoria (en metros) a píxeles
            pygame.draw.lines(
                self.screen, (80,80,200),
                False, [self.to_px(p) for p in traj], 2
            )

        # Dibujar texto de estado
        txt = self.font.render("Agente lanzando...", True, (0, 0, 0))
        self.screen.blit(txt, (20,20))

        # Actualiza la pantalla para mostrar lo que se ha dibujado
        pygame.display.flip()
        # Limita la velocidad de la animación a 15 frames por segundo (FPS)
        self.clock.tick(15) #Ajuste para una visualización más fluida/controlada