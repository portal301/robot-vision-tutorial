import open3d as o3d
import numpy as np
import os
from PIL import Image


# Load the lena image
image_path = os.path.join('image','lena.png')
image = Image.open(image_path)
image = image.convert('RGB')  # Ensure image is in RGB format

# Convert image to numpy array
image_np = np.array(image)

# Get the dimensions of the image
height, width, _ = image_np.shape

# Create a grid of (x, y) coordinates
x = np.arange(0, width)
y = np.arange(0, height)
x, y = np.meshgrid(x, y)

# Flatten the arrays
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)  # z = 0 for all points

# Get the color values for each point
colors = image_np[y, x] / 255.0  # Normalize to [0, 1]

# Create point cloud
points = np.stack((x, height - y - 1, z), axis=-1)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualization
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

# Add point cloud and line set to visualizer
visualizer.add_geometry(point_cloud)
# visualizer.add_geometry(line_set)

# Get rendering options for line set
render_options = visualizer.get_render_option()
render_options.point_size = 5  # Adjust point size here

# Run the visualizer
visualizer.run()
visualizer.destroy_window()
