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

# Generate the surface data
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
x, y = np.meshgrid(x, y)
z = -2*(0.75*x)**4 + 2*(x)**2 + (0.8*y)**2

# Create the reference surface mesh
reference_mesh = create_surface_mesh(x, y, z)

# Create the offset surface mesh
offset = 0.3
offset_surface_mesh1 = offset_mesh(reference_mesh, offset_distance=-offset, color=[0.5, 0.0, 0.0])
offset_surface_mesh2 = offset_mesh(reference_mesh, offset_distance=offset, color=[0.0, 0.5, 0.0])

# Visualize the original and offset meshes
visualize_meshes([reference_mesh, offset_surface_mesh1, offset_surface_mesh2])
