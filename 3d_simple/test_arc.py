import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_arrows(centers_and_directions):
    # Create a new figure
    fig = plt.figure(facecolor='#111')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    # Set the facecolor of the 3D axis to black
    ax.set_facecolor('white')
    
    # Set the color of the surfaces for each plane to black
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    # Set the color of the labels on the axes to white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')

    # Set the color of the grid lines to black
    ax.xaxis._axinfo['grid'].update(color = 'white')
    ax.yaxis._axinfo['grid'].update(color = 'white')
    ax.zaxis._axinfo['grid'].update(color = 'white')

    # Set the color of the tick marks and tick labels to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    ax.w_xaxis.pane.set_edgecolor('white')
    ax.w_yaxis.pane.set_edgecolor('white')
    ax.w_zaxis.pane.set_edgecolor('white')

    # Set the alpha parameter of the Axes3D object to 0
    ax.patch.set_alpha(0)
    # Extract center coordinates and direction vectors
    centers = centers_and_directions[:, :3]
    directions = centers_and_directions[:, 3:]
    # Draw the arrows
    for center, direction in zip(centers, directions):
        ax.quiver(center[0], center[1], center[2], direction[0], direction[1], direction[2], 
                  length=0.12, normalize=True, color='#5080DF',
                  arrow_length_ratio=0.5, pivot='tail' # tail, middle, tip
                  )
    # Draw small dots at the centers of the arrows
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='red', s=10)
    # Draw the polyline connecting the centers of the arrows

    angles = np.linspace(0, 2*np.pi, 100)
    radius = 0.65

    primitive = [
        radius * np.cos(angles),
        radius * np.sin(angles),
        0.15 * np.ones(angles.shape[0])
    ]
    # ax.plot(primitive[0], primitive[1], primitive[2], color='#00FF00')
    # ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color='g')
    # Set the limits of the plot
    max_range = np.array([centers[:,0].max()-centers[:,0].min(), centers[:,1].max()-centers[:,1].min(), centers[:,2].max()-centers[:,2].min()]).max() / 2.0
    mid_x = (centers[:,0].max()+centers[:,0].min()) * 0.5
    mid_y = (centers[:,1].max()+centers[:,1].min()) * 0.5
    mid_z = (centers[:,2].max()+centers[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # Label the axes
    ax.set_xlabel('z [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    # Show the plot
    plt.show()


def circular_pattern(radius, num_points, z_coordinate):
    # Generate angles evenly spaced around the circle
    angles = np.linspace(0.9*np.pi, 1.4*np.pi, num_points)
    # Compute x, y coordinates using parametric equations
    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)
    # Create an n by 6 array filled with zeros
    centers_and_directions = np.zeros((num_points, 6))

    # Create an empty array to store the centers with random error
    centers = np.zeros((num_points, 3))
    direction_vectors = np.zeros((num_points, 3))
    position_error_scale = 0.015
    direction_error_scale = 0.3
    # position_error_scale = 0.0
    # direction_error_scale = 0.0
    phi = 45/180*np.pi  # Adjust this as needed
    # Add random error to each element in the position data
    for i in range(num_points):
        x_error = np.random.uniform(-position_error_scale, position_error_scale)
        y_error = np.random.uniform(-position_error_scale, position_error_scale)
        z_error = np.random.uniform(-position_error_scale, position_error_scale)
        centers[i] = [x_coords[i] + x_error, y_coords[i] + y_error, z_coordinate + z_error]

        theta_error = angles[i] + np.random.uniform(-direction_error_scale, direction_error_scale)
        phi_error = phi + np.random.uniform(-direction_error_scale, direction_error_scale)
        direction_vectors[i] = np.array([-np.cos(theta_error)*np.sin(phi_error), -np.sin(theta_error)*np.sin(phi_error), -np.cos(phi_error)])  # Adjust this as needed

    centers_and_directions[:, :3] = centers
    # Assign x, y, z coordinates to the array
    # Compute direction vectors (for simplicity, using the same vector [0, 0, 1] for all points)
    # Assign direction vectors to the last three columns of the array
    centers_and_directions[:, 3:] = direction_vectors

    return centers_and_directions

# Example usage
radius = 0.65  # Radius of the circular pattern
num_points = 50  # Number of points
z_coordinate = 0.15  # Z-coordinate for each center
centers_and_directions = circular_pattern(radius, num_points, z_coordinate)

plot_arrows(centers_and_directions)
