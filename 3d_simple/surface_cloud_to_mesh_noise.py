import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

def add_noise_to_surface(z, noise_level=0.1):
    """
    Add random noise to the z-values of the surface.
    
    Parameters:
    - z: The original z-values of the surface.
    - noise_level: The standard deviation of the Gaussian noise to add.
    
    Returns:
    - The noisy z-values.
    """
    noise = np.random.normal(scale=noise_level, size=z.shape)
    return z + noise

def generate_point_cloud(x, y, z):
    # Stack the coordinates into a (num_points, 3) array
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    
    # Create the Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # point_cloud.colors = o3d.utility.Vector3dVector([0.0, 0.5, 0.0])
    
    return point_cloud

def create_mesh_from_surface(x, y, z):
    # Flatten and stack the coordinates
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    
    # Perform Delaunay triangulation on the (x, y) coordinates
    tri = Delaunay(np.vstack((x.flatten(), y.flatten())).T)
    
    # Create the Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
    
    return mesh

def visualize_mesh(mesh, point_cloud):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(point_cloud)

    # Add the mesh to the visualizer
    vis.add_geometry(mesh)
    
    # Customize render options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.0, 0.0, 0.0])  # Black background
    render_option.line_width = 1.0  # Set the width of the lines (edges)

    # Set the mesh to display edges
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()


    # Get rendering options for line set
    render_option.point_size = 5  # Adjust point size here

    # Create a mesh with edges
    mesh_edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_edges.paint_uniform_color([1.0, 0.0, 0.0])  # Color edges red

    # Add the edges to the visualizer
    vis.add_geometry(mesh_edges)
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


# Generate the surface data
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
x, y = np.meshgrid(x, y)
z = -2*(0.75*x)**4 + 2*(x)**2 + (0.8*y)**2

# Add noise to the z values
noise_level = 0.05  # Adjust the noise level here
z_noisy = add_noise_to_surface(z, noise_level)

# Generate the point cloud
point_cloud = generate_point_cloud(x, y, z_noisy)

# Create the mesh from the surface data
mesh = create_mesh_from_surface(x, y, z_noisy)

# Visualize the mesh
visualize_mesh(mesh,point_cloud)
