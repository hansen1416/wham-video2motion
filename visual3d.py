import open3d as o3d
import numpy as np
import time

print(o3d.__version__)

# Step 1: Create vertices and faces
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

faces = np.array([[0, 1, 2]])

# Step 2: Create a mesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)

# Step 3: Visualize the mesh
o3d.visualization.draw_geometries([mesh], window_name="Mesh Example")

# Step 4: Update vertices in a loop
for i in range(10):
    print(11111)
    # Update the vertices (e.g., move the triangle up)
    vertices[:, 2] += 0.1  # Move the triangle up in the z-direction
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Update the visualization
    o3d.visualization.draw_geometries([mesh], window_name="Mesh Example")

    # Pause for a moment to see the update
    time.sleep(0.5)
