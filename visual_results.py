import os
import sys
import math
import joblib

import pygame
import numpy as np


class MeshPlayer:

    WIDTH, HEIGHT = 800, 600
    FPS = 60

    def __init__(self, caption=None, vertices_frame=None, faces=None):

        # Initialize Pygame
        pygame.init()

        # Create the screen
        self.screen = pygame.display.set_mode((MeshPlayer.WIDTH, MeshPlayer.HEIGHT))

        if caption:
            pygame.display.set_caption(caption)

        self.vertices_frame = vertices_frame
        self.faces = faces

        self.vertices = self.vertices_frame[0]
        self.frame = 0

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

            self._control()

            # Draw the faces of the cube
            for face in self.faces:
                polygon = [self._project(self.vertices[i]) for i in face]
                pygame.draw.polygon(
                    self.screen, (255, 255, 255), polygon, width=0
                )  # Draw the face outline

            pygame.display.flip()

            self.frame += 1

            if self.frame >= len(self.vertices_frame):
                self.frame = 0

            self.vertices = self.vertices_frame[self.frame]

            clock.tick(MeshPlayer.FPS)


if __name__ == "__main__":

    video_name = "madfit1.mp4"

    output_pth = os.path.join(".", "output", video_name)

    results = joblib.load(os.path.join(output_pth, "wham_output.pkl"))
    faces = np.load(os.path.join(output_pth, "faces.npy"))

    all_vertices = results[0]["verts"]

    mp = MeshPlayer(vertices_frame=all_vertices, faces=faces)

    mp.run()
