import pygame

class PygameRenderer:
    """
    Maneja todo el renderizado en Pygame.
    """
    def __init__(self, env, width=1000, height=600):
        self.env = env
        self.WIDTH = width
        self.HEIGHT = height
        self.M2PX = 60
        self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Agente RL - Tiro a Canasta")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 26)

    def to_px(self, pos):
        x = int(100 + pos[0] * self.M2PX)
        y = int(self.HEIGHT - 100 - pos[1] * self.M2PX)
        return (x, y)

    def draw(self, traj):
        self.screen.fill((230, 240, 255))

        # Dibujar suelo
        pygame.draw.rect(self.screen, (60,160,60),
                         (0, self.HEIGHT-100, self.WIDTH, 100))

        # Dibujar pelota
        pygame.draw.circle(self.screen, (255,165,0),
                           self.to_px(self.env.pos), 8)

        # Dibujar canasta
        pygame.draw.circle(
            self.screen, (220,30,30),
            self.to_px([self.env.rim_x, self.env.rim_y]), 12, 3
        )

        # Dibujar trayectoria
        if len(traj) > 1:
            pygame.draw.lines(
                self.screen, (80,80,200),
                False, [self.to_px(p) for p in traj], 2
            )

        txt = self.font.render("Agente lanzando...", True, (0, 0, 0))
        self.screen.blit(txt, (20,20))

        pygame.display.flip()
        self.clock.tick(15) # ⬅️ CAMBIO: Ajustado a 15 FPS