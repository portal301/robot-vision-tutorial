import open3d as o3d
import numpy as np
import os
import sys

def generate_torus_path(r1,r2,r1_num_points,r2_num_points, center=[0,0,0], rotation_matrix=[np.eye(3)]):
    theta = np.linspace(-0.1*np.pi, 0.38*np.pi, r1_num_points).reshape(1,-1)
    phi = np.linspace(1*np.pi, 2*np.pi, r2_num_points).reshape(1,-1)

    x = r1*np.cos(theta)*np.ones_like(phi.T)+r2*np.cos(theta)*np.cos(phi.T)
    y = r1*np.sin(theta)*np.ones_like(phi.T)+r2*np.sin(theta)*np.cos(phi.T)
    z = np.ones_like(theta)*r2*np.sin(phi.T)

    x.shape=(1,-1)
    y.shape=(1,-1)
    z.shape=(1,-1)

    # differentiate along theta
    Vx_x = -r1*np.sin(theta)*np.ones_like(phi.T)-r2*np.sin(theta)*np.cos(phi.T)
    Vx_y = r1*np.cos(theta)*np.ones_like(phi.T)+r2*np.cos(theta)*np.cos(phi.T)
    Vx_z = np.zeros_like(theta)*np.zeros_like(phi.T)

    # differentiate along phi
    Vy_x = -np.cos(theta)*np.sin(phi.T)
    Vy_y = -np.sin(theta)*np.sin(phi.T)
    Vy_z = np.ones_like(theta)*np.cos(phi.T)

    # differentiate along r2
    Vz_x = np.cos(phi.T)*np.cos(theta)
    Vz_y = np.cos(phi.T)*np.sin(theta)
    Vz_z = np.ones_like(theta)*np.sin(phi.T)

    Vx_x.shape=(1,-1)
    Vx_y.shape=(1,-1)
    Vx_z.shape=(1,-1)

    Vy_x.shape=(1,-1)
    Vy_y.shape=(1,-1)
    Vy_z.shape=(1,-1)

    Vz_x.shape=(1,-1)
    Vz_y.shape=(1,-1)
    Vz_z.shape=(1,-1)

    # Vx = np.stack((Vx_x, Vx_y, Vx_z), axis=-1)
    Vy = np.stack((Vy_x, Vy_y, Vy_z), axis=-1)
    Vz = np.stack((Vz_x, Vz_y, Vz_z), axis=-1)

    Vx = np.cross(Vy, Vz)
    # Vy=np.cross(Vz, Vx)

    # normalize the vectors
    Vx = Vx/np.linalg.norm(Vx, axis=-1)[:,:,np.newaxis]
    Vy = Vy/np.linalg.norm(Vy, axis=-1)[:,:,np.newaxis]
    Vz = Vz/np.linalg.norm(Vz, axis=-1)[:,:,np.newaxis]

    rotation_matrices = np.stack((Vx, Vy, Vz), axis=-1)
    rotation_matrices.shape=(r2_num_points,r1_num_points,3,3)

    points = np.stack((x, y, z), axis=-1)
    points.shape=(r2_num_points,r1_num_points,3)

    dir = 1 # 1 for clockwise, -1 for counter-clockwise
    for i in range(r2_num_points):
        if dir == -1:
            # reverse the order of the points
            points[i] = points[i][::-1]
            rotation_matrices[i] = rotation_matrices[i][::-1]
        dir = -dir

    points.shape=(-1,3)
    rotation_matrices.shape=(-1,3,3)

    # Rotate 
    points = [np.dot(point, rotation_matrix.T) for point in points]
    rotation_matrices = [np.dot(rotation_matrix, rotation) for rotation in rotation_matrices]
    # Translate
    points = points + np.array(center)
    return points, rotation_matrices


def create_coordinate_frame(start, rotation_matrix):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
    coordinate_frame.rotate(rotation_matrix)
    coordinate_frame.translate(start)
    return coordinate_frame

def create_path_geometry(points, rotation_matrices):
    # Create Open3D point clouds
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)

    coordinate_frame_list = []
    for origin, rotation_matrix in zip(points, rotation_matrices):
        for i in range(3):
            coordinate_frame_list.append(create_coordinate_frame(origin, rotation_matrix))

    # Create LineSet object to draw lines between the points
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(points.shape[0]-1)])
    line_set.colors = o3d.utility.Vector3dVector([[0.9, 0.7, 0] for i in range(points.shape[0]-1)])  # Orange color for the lines

    return point_cloud, coordinate_frame_list, line_set

# This function is a fake implementation of a function that would return the parameters of a bended pipe
# We need to implement this function using AI/ML models, etc.
def get_bent_pipe_params_fake(pcd): 
    r1=480
    r2=120
    center=[210,-415,1380],
    theta_rad = np.radians(90)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad),  np.cos(theta_rad), 0],
        [0,                 0,                 1]
    ])

    return  {"r1": r1, "r2": r2, "center": center, "rotation_matrix": rotation_matrix}

# Create a Visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window()

# Load the point cloud from a PCD file
script_dir = os.path.dirname(os.path.abspath(__file__))

pcd = o3d.io.read_point_cloud(os.path.join(script_dir,"pcd", "data.pcd"))

# Check if the point cloud is loaded successfully
if pcd.is_empty():
    print("Failed to load point cloud from data.pcd")
    sys.exit(-1)
else:
    print("Successfully loaded point cloud from data.pcd")
    pcd.rotate(np.array([[1,0,0],[0,1,0],[0,0,-1]]))

# Visualize the point cloud
vis.add_geometry(pcd)

geometry_params = get_bent_pipe_params_fake(pcd) # fake implementation - AI/ML models, etc. will be used to get the parameters 
points, rotation_matrices = generate_torus_path(r1=geometry_params["r1"],r2=geometry_params["r2"],
                                                center=geometry_params["center"], rotation_matrix=geometry_params["rotation_matrix"],
                                                r1_num_points=20,r2_num_points=10)
point_cloud, coordinate_frame_list, line_set = create_path_geometry(points, rotation_matrices)

vis.add_geometry(point_cloud)
vis.add_geometry(line_set)
for coordinate_frame in coordinate_frame_list:
    vis.add_geometry(coordinate_frame)

render_option = vis.get_render_option()
render_option.point_size = 5

# Run the visualizer
vis.run()
vis.destroy_window()
