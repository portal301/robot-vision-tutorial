import open3d as o3d
import numpy as np


# Create an array of points aligned on a line
r1_num_points = 20
r2_num_points = 20
r1=20
r2=5
theta = np.linspace(0, 2*np.pi, r1_num_points).reshape(1,-1)
phi = np.linspace(0*np.pi, 2*np.pi, r2_num_points).reshape(1,-1)
thickness = 2

# print(theta)

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

Rot = np.stack((Vx, Vy, Vz), axis=-1)
points = np.stack((x, y, z), axis=-1)

Rot.shape=(r2_num_points,r1_num_points,3,3)
points.shape=(r2_num_points,r1_num_points,3)

dir = 1 # 1 for clockwise, -1 for counter-clockwise
for i in range(r2_num_points):
    if dir == -1:
        # reverse the order of the points
        points[i] = points[i][::-1]
        Rot[i] = Rot[i][::-1]
    dir = -dir

points.shape=(-1,3)
Rot.shape=(-1,3,3)

# Create Open3D point clouds
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)


# Create a Visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window()


# Add the sphere, cube, and lines to the visualizer
vis.add_geometry(point_cloud)


# Create LineSet object for the cube edges
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(points.shape[0]-1)])

vis.add_geometry(line_set)


def create_coordinate_frame(start, rotation_matrix):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    coordinate_frame.rotate(rotation_matrix)
    coordinate_frame.translate(start)
    return coordinate_frame

for origin, rotation_matrix in zip(points, Rot):
    for i in range(3):
        arrows = create_coordinate_frame(origin, rotation_matrix)
        vis.add_geometry(arrows)

render_option = vis.get_render_option()
render_option.point_size = 5


# Run the visualizer
vis.run()
vis.destroy_window()
