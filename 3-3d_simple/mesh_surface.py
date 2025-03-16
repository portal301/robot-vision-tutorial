import open3d as o3d
import numpy as np

def create_surface_mesh(x, y, z):
    # Create the mesh
    mesh = o3d.geometry.TriangleMesh()
    
    # Convert the mesh data to Open3D format
    vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Create the triangles
    triangles = []
    rows, cols = x.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Define two triangles per square
            triangles.append([i * cols + j, (i + 1) * cols + j, i * cols + (j + 1)])
            triangles.append([(i + 1) * cols + j, (i + 1) * cols + (j + 1), i * cols + (j + 1)])
    
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Compute vertex normals
    mesh.compute_vertex_normals()
    
    return mesh

def visualize_mesh(mesh):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the mesh to the visualizer
    vis.add_geometry(mesh)
    
    # Customize render options
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.background_color = np.array([0.0, 0.0, 0.0])  # Black background
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

# Generate the surface data
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
x, y = np.meshgrid(x, y)
z = -1.5*(0.70*x)**4 + 1.5*(x)**2 + (0.8*y)**2

# Create the mesh
surface_mesh = create_surface_mesh(x, y, z)

# Visualize the mesh
visualize_mesh(surface_mesh)
