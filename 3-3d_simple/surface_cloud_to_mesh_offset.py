import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from multiprocessing import Process


def offset_mesh(mesh, offset_distance=1, color=[0.5, 0.5, 0.5]):
    # Create a new mesh for the offset surface
    offset_mesh = o3d.geometry.TriangleMesh()
    
    # Get vertices and normals
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    # Offset vertices
    offset_vertices = vertices + offset_distance * normals
    offset_mesh.vertices = o3d.utility.Vector3dVector(offset_vertices)
    
    # Copy the triangles from the original mesh
    offset_mesh.triangles = mesh.triangles
    
    # Compute normals for the offset mesh
    offset_mesh.compute_vertex_normals()
    
    # Color for the offset mesh
    offset_mesh.paint_uniform_color(color)
    
    return offset_mesh

def add_noise_to_surface(z, noise_level=0.1):
    noise = np.random.normal(scale=noise_level, size=z.shape)
    return z + noise

def create_mesh_from_surface(x, y, z):
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    tri = Delaunay(np.vstack((x.flatten(), y.flatten())).T)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
    return mesh

def smooth_mesh(mesh, smoothing_iterations=10):
    # Apply Laplacian smoothing
    mesh = mesh.filter_smooth_simple(number_of_iterations=smoothing_iterations)
    # Compute vertex normals
    mesh.compute_vertex_normals()
    return mesh




def visualize_meshes(meshes=[]):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    for mesh in meshes:
        vis.add_geometry(mesh)
    
    # Customize render options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.0, 0.0, 0.0])  # Black background
    render_option.mesh_show_back_face = True
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

# def visualize_mesh(mesh, title, color=[0.5, 0.5, 0.5]):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=title)

#     # Add mesh with specified color
#     mesh.paint_uniform_color(color)
#     vis.add_geometry(mesh)

#     # Optionally add edge lines
#     mesh_edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
#     mesh_edges.paint_uniform_color([1.0, 1.0, 1.0])  # Black edges
#     vis.add_geometry(mesh_edges)

#     # Set up the visualizer
#     render_option = vis.get_render_option()
#     render_option.background_color = np.array([0, 0, 0])  # White background

#     vis.run()
#     vis.destroy_window()

def generate_point_cloud(x, y, z):
    # Stack the coordinates into a (num_points, 3) array
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    
    # Create the Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # point_cloud.colors = o3d.utility.Vector3dVector([0.0, 0.5, 0.0])
    
    return point_cloud


def show_point_cloud():
    point_cloud = generate_point_cloud(x, y, z_noisy)
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(point_cloud)

    # Customize render options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.0, 0.0, 0.0])  # Black background

    # Get rendering options for line set
    render_option.point_size = 5  # Adjust point size here
   
    # Run the visualizer
    vis.run()
    vis.destroy_window()


# def show_smoothed_mesh():
#     visualize_mesh(smoothed_mesh, title="Smoothed Mesh", color=[0.0, 1.0, 0.0])



# Generate the surface data
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
x, y = np.meshgrid(x, y)
z = -2*(0.75*x)**4 + 2*(x)**2 + (0.8*y)**2

# Add noise to the z values
noise_level = 0.1  # Adjust the noise level here
z_noisy = add_noise_to_surface(z, noise_level)
point_cloud = generate_point_cloud(x, y, z_noisy)

# Create the mesh from the noisy surface
original_mesh = create_mesh_from_surface(x, y, z_noisy)

# Smooth the mesh
smoothing_iterations = 5  # Adjust the number of iterations here
smoothed_mesh = smooth_mesh(original_mesh, smoothing_iterations)

offset = -0.3
offset_surface_mesh = offset_mesh(smoothed_mesh, offset_distance=-offset, color=[0.5, 0.5, 0.5])

# Use multiprocessing to display the meshes simultaneously
if __name__ == '__main__':
    visualize_meshes([offset_surface_mesh,point_cloud])

