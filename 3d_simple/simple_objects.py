import open3d as o3d
import numpy as np

# Create a sphere mesh
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
sphere.paint_uniform_color([0.1, 0.1, 0.7])  # Set color for the sphere

# Create a cube mesh
cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
cube.paint_uniform_color([0.7, 0.1, 0.1])  # Set color for the cube

# Move the cube to a different location
cube.translate([5.0, 0.0, 0.0])  # Translate the cube along the x-axis

# Create point cloud lines to connect the sphere and the cube
# Define the connection points (one point on the sphere and one point on the cube)
sphere_center = np.array([0.0, 0.0, 0.0])  # Center of the sphere
cube_center = np.array([5.0, 0.0, 0.0])    # Center of the cube


# Create an array of points aligned on a line
num_points = 100
x = np.linspace(0, 5, num_points)
y = np.linspace(0, 0, num_points)
z = np.zeros(num_points)
points = np.stack((x, y, z), axis=-1)

# Create colors for the points
colors = np.zeros((num_points, 3))
colors[:, 0] = np.linspace(0, 1, num_points)  # Red channel
colors[:, 1] = np.linspace(1, 0, num_points)  # Green channel
colors[:, 2] = 0  # Blue channel remains 0

# Create Open3D point clouds
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)


# Define lines connecting the sphere and cube
lines = [[0, 1]]
line_points = np.array([sphere_center, cube_center])

# Create LineSet object for connecting lines
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(line_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color for the line

# Create LineSet object for cube edges
cube_edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face edges
    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face edges
    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
]

# Get the vertices of the cube
cube_vertices = np.array(cube.vertices)

# Create LineSet object for the cube edges
edge_set = o3d.geometry.LineSet()
edge_set.points = o3d.utility.Vector3dVector(cube_vertices)
edge_set.lines = o3d.utility.Vector2iVector(cube_edges)
edge_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green color for the edges

# Create a Visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window()


# Add the sphere, cube, and lines to the visualizer
vis.add_geometry(sphere)
vis.add_geometry(cube)
vis.add_geometry(edge_set)
vis.add_geometry(point_cloud)

render_option = vis.get_render_option()
render_option.point_size = 5


# Run the visualizer
vis.run()
vis.destroy_window()
