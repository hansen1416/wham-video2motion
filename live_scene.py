import numpy as np
import pyrender
from pyrender import Mesh, Primitive
import trimesh
import time

# Create a simple mesh (e.g., a sphere)
sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
mesh: Mesh = pyrender.Mesh.from_trimesh(sphere)

# Create a Pyrender scene
scene = pyrender.Scene()
meshnode = scene.add(mesh)

# Create a viewer
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)


# Function to update vertices
def update_vertices(mesh: Mesh, time_step):
    mesh_primitive: Primitive = mesh.primitives[0]
    # Get current vertices
    vertices = mesh_primitive.positions

    # Example: modify vertices to create a wave effect
    wave_amplitude = 0.2
    wave_frequency = 5
    for i in range(len(vertices)):
        vertices[i][2] = wave_amplitude * np.sin(
            wave_frequency * (vertices[i][0] + time_step)
        )

    # Update the mesh with new vertices
    # mesh.primitives[0].positions = vertices
    mesh_primitive.positions = vertices
    # mesh_primitive.rebuild()


# Animation loop
start_time = time.time()

for i in range(1000):
    # Calculate time elapsed
    current_time = time.time()
    time_step = current_time - start_time

    viewer.render_lock.acquire()

    # Update the mesh vertices
    # update_vertices(mesh, time_step)

    # scene.set_pose(meshnode, pose=)

    # viewer.scene.add(mesh)

    # Render the scene

    viewer.render_lock.release()

    time.sleep(0.016)
