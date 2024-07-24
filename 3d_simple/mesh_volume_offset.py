import open3d as o3d
import numpy as np

def create_sphere_mesh(radius=1.0, resolution=20, color=None):
    # Create a mesh representing a sphere
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    mesh.compute_vertex_normals()
    
    # Set color if provided
    if color:
        mesh.paint_uniform_color(color)
    
    return mesh

def create_cube_mesh(size=1.0, color=None):
    # Create a mesh representing a cube
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    mesh.compute_vertex_normals()
    
    # Set color if provided
    if color:
        mesh.paint_uniform_color(color)
    
    return mesh

def offset_mesh(mesh, offset_distance):
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
    
    # Create a uniform color for the offset mesh
    offset_color = [0.5, 0.0, 0.0]  # Light gray color to simulate transparency
    offset_mesh.paint_uniform_color(offset_color)
    
    return offset_mesh

def visualize_meshes(original_meshes, offset_meshes):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add original meshes to the visualizer
    for mesh in original_meshes:
        vis.add_geometry(mesh)
    
    # Add offset meshes to the visualizer
    for mesh in offset_meshes:
        vis.add_geometry(mesh)
    
    # Get the render option to set visualization settings
    render_option = vis.get_render_option()
    
    # Set background color to highlight meshes
    render_option.background_color = np.array([0, 0, 0])  # Black background
    
    # Simulate transparency by adjusting colors and lighting
    render_option.mesh_show_back_face = True
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

# Create original meshes with colors
sphere_mesh = create_sphere_mesh(radius=1.0, resolution=20, color=[1, 0, 0])  # Red sphere
cube_mesh = create_cube_mesh(size=1.0, color=[0, 1, 0])  # Green cube

# Create offset meshes
offset_sphere_mesh = offset_mesh(sphere_mesh, offset_distance=0.2)
offset_cube_mesh = offset_mesh(cube_mesh, offset_distance=0.2)

# Visualize original and offset meshes
visualize_meshes([sphere_mesh, cube_mesh], [offset_sphere_mesh, offset_cube_mesh])
