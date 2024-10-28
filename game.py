import pygame
import numpy as np
import sys
import math


class MeshPlayer:

    WIDTH, HEIGHT = 800, 600
    FPS = 60

    def __init__(self, caption=None, vertices=None, faces=None):

        # Initialize Pygame
        pygame.init()

        # Create the screen
        self.screen = pygame.display.set_mode((MeshPlayer.WIDTH, MeshPlayer.HEIGHT))

        if caption:
            pygame.display.set_caption(caption)

        self.vertices = vertices
        self.faces = faces

        self.angle_y = 0  # Initial rotation angle around Y axis

    def _control(self):
        # Handle input for rotation
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.angle_y -= 0.05  # Rotate left
        if keys[pygame.K_RIGHT]:
            self.angle_y += 0.05  # Rotate right

        rotation_matrix = np.array(
            [
                [math.cos(self.angle_y), 0, math.sin(self.angle_y)],
                [0, 1, 0],
                [-math.sin(self.angle_y), 0, math.cos(self.angle_y)],
            ]
        )

        self.vertices = np.dot(self.vertices, rotation_matrix)

        self.angle_y = 0

    def _update(self):
        """Randomly update the vertices of the cube."""
        for i in range(len(self.vertices)):
            # Randomly adjust each vertex position slightly
            self.vertices[i] += np.random.normal(0, 0.01, size=3)  # Small random change
        # return vertices

    # Projection parameters
    def _project(self, vertex):
        """Project 3D vertex to 2D space."""
        factor = 200  # Projection factor
        x, y, z = vertex
        x_proj = int(factor * x / (z + 5)) + MeshPlayer.WIDTH // 2
        y_proj = int(-factor * y / (z + 5)) + MeshPlayer.HEIGHT // 2
        return x_proj, y_proj

    def run(self):
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                    # Clear the screen
            self.screen.fill((0, 0, 0))  # Black background

            self._update()

            self._control()

            # Draw the faces of the cube
            for face in faces:
                polygon = [self._project(self.vertices[i]) for i in face]
                pygame.draw.polygon(
                    self.screen, (255, 255, 255), polygon, width=0
                )  # Draw the face outline

            pygame.display.flip()
            clock.tick(MeshPlayer.FPS)


if __name__ == "__main__":
    # main()

    # Define the vertices of a cube
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )

    # Define the faces of the cube (each face is defined by a list of vertex indices)
    faces = [
        [0, 1, 2],  # Back face
        [4, 5, 6],  # Front face
        [0, 1, 5],  # Bottom face
        [2, 3, 7],  # Top face
        [0, 3, 7],  # Left face
        [1, 2, 6],  # Right face
    ]

    mp = MeshPlayer(vertices=vertices, faces=faces)

    mp.run()
