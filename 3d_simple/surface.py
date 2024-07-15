import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def calculate_normals(x, y, z):
#     # Calculate gradients
#     dz_dx, dz_dy = np.gradient(z, x[0, :], y[:, 0])
#     # Calculate the normal vectors
#     normals = np.dstack((-dz_dx, -dz_dy, np.ones_like(z)))
#     # Normalize the normal vectors
#     norm = np.linalg.norm(normals, axis=2)
#     normals /= norm[..., np.newaxis]
#     return normals

def calculate_normals(x, y, z):
    # Calculate gradients
    # dz_dx, dz_dy = np.gradient(z, x[:, 0], y[0, :])
    dz_dy, dz_dx = np.gradient(z, y[:, 0], x[0, :])
    print(dz_dx.shape, dz_dy.shape, z.shape)
    # Calculate the normal vectors
    normals = np.dstack((-dz_dx, -dz_dy, np.ones_like(z)))
    # Normalize the normal vectors
    print("normals", normals.shape)
    print(normals)
    norm = np.linalg.norm(normals, axis=2)
    print("norm:",  norm[..., np.newaxis])


    # norm = np.linalg.norm(normals)
    
    normals /= norm[..., np.newaxis]
    print("normals")
    print(normals)
    print(normals[:,:,0]**2+ normals[:,:,1]**2+ normals[:,:,2]**2)
    return normals


def plot_surface_with_offset(x, y, z, offset_distance):
    # Calculate the normals
    normals = calculate_normals(x, y, z)
    
    # Create the offset surface
    x_offset = x + offset_distance * normals[:, :, 0]
    y_offset = y + offset_distance * normals[:, :, 1]
    z_offset = z + offset_distance * normals[:, :, 2]

    # Create the plot
    fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')  # Set background color to black

    # Plot the original surface
    surface = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', antialiased=True)

    # Plot the offset surface
    offset_surface = ax.plot_surface(x_offset, y_offset, z_offset, cmap='plasma', edgecolor='none', antialiased=True, alpha=0.5)

    # Customize the color bars
    color_bar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, pad=0.1)
    color_bar.set_label('Height (Original)', color='white')
    color_bar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(color_bar.ax.axes, 'yticklabels'), color='white')

    color_bar_offset = fig.colorbar(offset_surface, ax=ax, shrink=0.5, aspect=5, pad=0.05)
    color_bar_offset.set_label('Height (Offset)', color='white')
    color_bar_offset.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(color_bar_offset.ax.axes, 'yticklabels'), color='white')

    # Add labels
    ax.set_xlabel('X axis', color='white')
    ax.set_ylabel('Y axis', color='white')
    ax.set_zlabel('Z axis', color='white')

    # Set tick label colors
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


    # Show the plot
    plt.show()



def plot_surface_with_offset_path(x, y, z, offset_distance, path):
    # Calculate the normals
    normals = calculate_normals(x, y, z)
    
    # Create the offset surface
    x_offset = x + offset_distance * normals[:, :, 0]
    y_offset = y + offset_distance * normals[:, :, 1]
    z_offset = z + offset_distance * normals[:, :, 2]

    # Create the plot
    fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')  # Set background color to black

    # Plot the original surface
    surface = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', antialiased=True)

    # Plot the offset surface
    offset_surface = ax.plot_surface(x_offset, y_offset, z_offset, cmap='plasma', edgecolor='none', antialiased=True, alpha=0.5)

    # Customize the color bars
    color_bar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, pad=0.1)
    color_bar.set_label('Height (Original)', color='white')
    color_bar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(color_bar.ax.axes, 'yticklabels'), color='white')

    color_bar_offset = fig.colorbar(offset_surface, ax=ax, shrink=0.5, aspect=5, pad=0.05)
    color_bar_offset.set_label('Height (Offset)', color='white')
    color_bar_offset.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(color_bar_offset.ax.axes, 'yticklabels'), color='white')


    ax.plot(path[0], path[1], path[2], color='#FF0000')


    # Add labels
    ax.set_xlabel('X axis', color='white')
    ax.set_ylabel('Y axis', color='white')
    ax.set_zlabel('Z axis', color='white')

    # Set tick label colors
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


    # Show the plot
    plt.show()


# Example usage
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
x, y = np.meshgrid(x, y)
z = -(0.5*x)**2+(0.5*y)**2

path_2d = np.linspace([-1,-1,5], [1,1,5], 50)


print(path_2d)

projection_matrix = np.array([[1,0,0],[0,1,0],[0,0,0]])

def z_dir_projection(path, surface):
    for point in path:
        print(point)
        print(np.dot(point, surface))

# path_3d = np.dot(path_2d, projection_matrix)
# print(path_3d)


offset_distance = 0.8
# plot_surface_with_offset(x, y, z, offset_distance)
plot_surface_with_offset_path(x, y, z, offset_distance, path_2d)
