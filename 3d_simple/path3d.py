import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image
image_path = 'testImage1.png'
image = cv2.imread(image_path)

# Convert the image to RGB (Matplotlib uses RGB, OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the dimensions of the image
height, width, _ = image_rgb.shape

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create X, Y coordinates
x = np.linspace(0, width, width)
y = np.linspace(0, height, height)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)

# Plot the image on the z=0 plane using plot_surface
ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=image_rgb / 255, shade=False)

# Set the labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([0, width])
ax.set_ylim([0, height])
ax.set_zlim([0, 1])

# Adjust the view angle
ax.view_init(elev=90, azim=-90)

# Hide the z-axis
ax.set_zticks([])

# Show the plot
plt.show()
