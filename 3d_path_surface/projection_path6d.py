import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

import cv2
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon, MultiPolygon

def generate_point_cloud(x, y, z):
    # Stack the coordinates into a (num_points, 3) array
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    
    # Create the Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # point_cloud.colors = o3d.utility.Vector3dVector([0.0, 0.5, 0.0])
    
    return point_cloud

def generate_mesh_from_surface(pcd):
    x = np.array(pcd.points)[:,0]
    y = np.array(pcd.points)[:,1]
    z = np.array(pcd.points)[:,2]
    # Flatten and stack the coordinates
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    
    # Perform Delaunay triangulation on the (x, y) coordinates
    tri = Delaunay(np.vstack((x.flatten(), y.flatten())).T)
    
    # Create the Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
    
    return mesh, tri

def visualize_3d(objects):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for obj in objects:
        if obj["type"] == "mesh":
            mesh = obj["data"]
            vis.add_geometry(mesh)
            # Set the mesh to display edges
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()


            # Create a mesh with edges
            # mesh_edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            # mesh_edges.paint_uniform_color([1.0, 0.0, 0.0])  # Color edges red

            # # Add the edges to the visualizer
            # vis.add_geometry(mesh_edges)

        elif obj["type"] == "pcd":
            pcd = obj["data"]
            vis.add_geometry(pcd)

            if "color" in obj:
                pcd.paint_uniform_color(obj["color"])

        elif obj["type"] == "path3d":
            path = obj["data"]
            lines = []
            for i in range(len(path)-2):
                lines.append([i,i+1])
            lineset = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(path),
                lines=o3d.utility.Vector2iVector(lines),
            )
            vis.add_geometry(lineset)

            if "color" in obj:
                lineset.paint_uniform_color(obj["color"])
            else:
                lineset.paint_uniform_color([0.0, 0.0, 1.0])  # Color edges blue
        elif obj["type"] == "path6d":
            path = obj["data"]
            path3d = path[:,:3]
            lines = []
            for i in range(len(path3d)-1):
                lines.append([i,i+1])
            # lines.append([len(path3d)-1,0])
            lineset = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(path3d),
                lines=o3d.utility.Vector2iVector(lines),
            )
            vis.add_geometry(lineset)

            for pos6d in path:
                size = 0.05

                # convert rotation vector to rotation matrix
                rvec = pos6d[3:6]
                r_mat = cv2.Rodrigues(rvec)[0]
                
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
                coordinate_frame.rotate(r_mat)
                coordinate_frame.translate(pos6d[:3])
                vis.add_geometry(coordinate_frame)


            if "color" in obj:
                lineset.paint_uniform_color(obj["color"])
            else:
                lineset.paint_uniform_color([0.0, 0.0, 1.0])
    


    # Customize render options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # Black background
    render_option.line_width = 1.0  # Set the width of the lines (edges)
    # Get rendering options for line set
    render_option.point_size = 5  # Adjust point size here

    # Run the visualizer
    vis.run()
    vis.destroy_window()




def find_intersection_points_metric(contours, gap):
    points = []
    for contour in contours:
        # print("contour")
        # for i in range(len(contour)):
        #     print(contour[i])
        # contour = np.array(contour).squeeze(axis=1)  # Remove unnecessary dimension
        max_y = np.max(np.array(contour)[:,1])#-gap*0.001
        min_y = np.min(np.array(contour)[:,1])#+gap*0.001
        # print(np.arange(min_y,max_y+gap,gap))
        for y in np.arange(min_y,max_y+gap,gap):
            for i in range(len(contour)):
                x1, y1 = contour[i]
                x2, y2 = contour[(i + 1) % len(contour)]  # Wrap around to close the contour
                # Check if the contour segment crosses the current horizontal line
                if (y1 <= y <= y2) or (y2 <= y <= y1):
                    # Calculate intersection point using linear interpolation
                    if y1 == y2:  # Horizontal segment, skip
                        continue
                    # x_intersection = int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                    x_intersection = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                    # print(x_intersection)
                    
                    is_new = True
                    for point in points:
                        if abs(point[0]-x_intersection) < 0.001 and abs(point[1]-y) < gap*0.001:
                            print("same point")
                            is_new = False
                            break
                    if is_new:
                        print("len: ",len(points), "x: ",x_intersection, "y: ",y)
                        # if (x_intersection, y) not in points:
                        points.append((x_intersection, y))
    return points
def rearrange_nodes(nodes):

    for i in range(len(nodes)-1):
        if nodes[i][1] == nodes[i+1][1] and nodes[i][0] > nodes[i+1][0]:
            temp = nodes[i]
            nodes[i] = nodes[i+1]
            nodes[i+1] = temp



    direction = 1 # 1 for right, -1 for left
    for i in range(len(nodes)-1):
        if nodes[i][1] == nodes[i+1][1]:
            if direction == -1:
                temp = nodes[i]
                nodes[i] = nodes[i+1]
                nodes[i+1] = temp
            direction *= -1
    return nodes


def add_intermediate_points_x_dir(points, gap):
    new_points = []

    for i in range(len(points) - 1):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[i + 1]

        # 현재 점을 추가
        new_points.append((x1, y1, z1))

        # x 차이와 y 차이 계산
        x_diff = abs(x2 - x1)

        # x 차이가 y 차이보다 클 경우, y 차이만큼 x에 새로운 점을 추가
        if x_diff > gap and x_diff != 0:  # y_diff가 0이면 추가할 필요 없음
            num_points_to_add = int(x_diff / gap)
            for j in range(1, num_points_to_add + 1):
                new_x = x1 + j * (x2 - x1) / num_points_to_add
                new_y = y1 + j * (y2 - y1) / num_points_to_add
                new_z = z1 + j * (z2 - z1) / num_points_to_add
                # new_points.append((int(new_x), int(new_y)))
                new_points.append((new_x, new_y, new_z))

    # 마지막 점 추가
    # new_points.append(points[-1])

    return new_points

def generate_affine_path(pcd, waypoint_interval=0.1):
    pcd_projected = np.array(pcd.points)
    affine_plane_z = np.max(pcd_projected[:, 2])+0.1
    
    for i in range(len(pcd_projected)):
        pcd_projected[i][2] = affine_plane_z

    # Convert 3d to 2d
    points_2d = np.array(pcd_projected)[:, :2]
    alpha_shape = alphashape.alphashape(points_2d, 0.1)
    # Plot the alpha shape
    # plt.scatter(points_2d[:, 0], points_2d[:, 1], label='Points')
    plt.plot(*alpha_shape.exterior.xy, 'r-', label='Alpha Shape')

    # Initialize a list to store contour line segments
    contour_lines = []

    # Check if the alpha shape is a Polygon or MultiPolygon
    if isinstance(alpha_shape, Polygon):
        # Get the exterior boundary
        coords = list(alpha_shape.exterior.coords)
        contour_lines = [coords[i] for i in range(len(coords) - 1)]

    elif isinstance(alpha_shape, MultiPolygon):
        # Iterate through each polygon
        for polygon in alpha_shape:
            coords = list(polygon.exterior.coords)
            contour_lines.extend([coords[i] for i in range(len(coords) - 1)])

    intersection_points = find_intersection_points_metric([contour_lines],0.1)
    intersection_points = rearrange_nodes(intersection_points)
    intersection_points = np.array(intersection_points)

    # display the primitive path
    plt.plot(intersection_points[:,0],intersection_points[:,1])
    plt.show()

    affine_path = np.hstack((intersection_points, affine_plane_z*np.ones((intersection_points.shape[0], 1))))
    affine_path = add_intermediate_points_x_dir(affine_path, waypoint_interval)
    return affine_path

def build_6d_path(path_3d, mesh):
    mesh.compute_triangle_normals()

    # Extract triangles and vertices as NumPy arrays
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Calculate the centers of each triangle
    triangle_centers = np.mean(vertices[triangles], axis=1)

    kd_tree = cKDTree(triangle_centers)

    distances, closest_triangle_indices = kd_tree.query(path_3d, k=1)

    print("closest triangle indices: ",closest_triangle_indices)

    # Extract triangle normals as a NumPy array
    triangle_normals = np.asarray(mesh.triangle_normals)

    # Define the length of the normal vectors for visualization
    normal_length = 0.2  # Adjust this value as needed

    # Calculate the end points of the normals
    normals_end = triangle_centers + normal_length * triangle_normals

    path_6d = []
    for i, pos in enumerate(path_3d):
        mat = np.eye(4)
        mat[:3,3] = pos

        z_vec = triangle_normals[closest_triangle_indices[i]]
        if i == 0:
            x_vec = path_3d[i+1] - path_3d[i]
        elif i == len(path_3d)-1:
            x_vec = path_3d[i] - path_3d[i-1]
        else:
            x_vec = path_3d[i+1] - path_3d[i-1]

        x_vec /= np.linalg.norm(x_vec)
        y_vec = np.cross(z_vec,x_vec)
        y_vec /= np.linalg.norm(y_vec)
        rmat = np.vstack((x_vec,y_vec,z_vec)).T
        rvec = cv2.Rodrigues(rmat)[0]

        # mat[:3,:3] = cv2.Rodrigues(triangle_normals[closest_triangle_indices[i]])[0]
        # path_6d.append(mat)

        path_6d.append(np.hstack((pos, rvec.flatten())))
        # path_6d.append(np.hstack((pos, np.rad2deg([0,0,45]))))
    path_6d = np.array(path_6d)
    return path_6d


def create_test_cloud():
    # Generate the surface data
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, y)
    z = 1*(0.75*y)**3 + 0.5*(y)**2 -1.4*(x)**4 + 1.4*(x)**2

    for xi in range(len(x)):
        for yi in range(len(y)):
            if xi > len(x)*1/3 and xi<len(x)*2/3 and yi > len(y)*1/3 and yi < len(y)*2/3:
                z[xi][yi] = 0
            if z[xi][yi] > 0.7:
                z[xi][yi] = 0.7

    # Generate the point cloud
    point_cloud = generate_point_cloud(x, y, z)

    return point_cloud


def project_path_to_mesh(affine_path, mesh):
    # convert faces to points
    points = np.array(mesh.vertices)
    mesh_points = points[:,:2]
    mesh_z = points[:,2]

    path_2d = np.array(affine_path)[:, :2]
    # Use griddata to interpolate z-values at the path's x, y locations
    path_z = griddata(mesh_points, mesh_z, path_2d, method='linear')

    # Step 4: Handle Any NaN Values from Interpolation
    # Replace NaNs (if any) using nearest-neighbor interpolation
    nan_indices = np.isnan(path_z)
    if np.any(nan_indices):
        path_z[nan_indices] = griddata(mesh_points, mesh_z, path_2d[nan_indices], method='nearest')

    # Combine x, y, z to form the 3D path
    path_3d = np.column_stack((path_2d, path_z))
    return path_3d

if __name__ == "__main__":
    point_cloud = create_test_cloud()
    mesh, faces = generate_mesh_from_surface(point_cloud)
    affine_path = generate_affine_path(point_cloud, waypoint_interval=0.05)
    path_3d = project_path_to_mesh(affine_path, mesh)   
    path_6d = build_6d_path(path_3d, mesh)

    visualize_3d([
        {"type":"mesh","data":mesh},
        # {"type":"pcd","data":point_cloud},
        # {"type":"path3d","data":affine_path},
        # {"type":"path3d","data":path_3d,"color":[0,1,0]}
        {"type":"path6d","data":path_6d,"color":[0,1,0]}
        ])
