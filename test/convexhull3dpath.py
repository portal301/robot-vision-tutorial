import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to draw parallel lines
def draw_parallel_lines(image, gap):
    height, width = image.shape[:2]
    lines_image = image.copy()

    for y in range(0, height, gap):
        cv2.line(lines_image, (0, y), (width, y), (128, 128, 128), 1)
    
    return lines_image

def find_intersection_points(contours, gap, height, width):
    points = []

    for y in range(0, height, gap):
        for contour in contours:
            contour = contour.squeeze(axis=1)  # Remove unnecessary dimension

            for i in range(len(contour)):
                x1, y1 = contour[i]
                x2, y2 = contour[(i + 1) % len(contour)]  # Wrap around to close the contour

                # Check if the contour segment crosses the current horizontal line
                if (y1 <= y <= y2) or (y2 <= y <= y1):
                    # Calculate intersection point using linear interpolation
                    if y1 == y2:  # Horizontal segment, skip
                        continue
                    x_intersection = int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                    points.append((x_intersection, y))

    return points

def rearrange_nodes(nodes):
    direction = 1 # 1 for right, -1 for left
    for i in range(len(nodes)-1):
        if nodes[i][1] == nodes[i+1][1]:
            if direction == -1:
                temp = nodes[i]
                nodes[i] = nodes[i+1]
                nodes[i+1] = temp
            direction *= -1
    return nodes
            
def draw_path(nodes, image):
    image_path = image.copy()
    for i in range(len(nodes)-1):
        cv2.line(image_path, nodes[i], nodes[i+1], (0, 0, 255), 2)
    return image_path




# Step 1: Load the image
image_path = 'testImage2.png'
# image_path = 'gratingImage.png'
image = cv2.imread(image_path)

# Step 2: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Convert to black and white (binary) image
_, bw_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Invert the binary image if necessary to make the object white and background black
bw_image = cv2.bitwise_not(bw_image)

# Step 6: Create the offset contours by dilating the binary image
kernel = np.ones((20, 20), np.uint8)  # Define a square kernel with a size of 10 pixels
dilated_image = cv2.dilate(bw_image, kernel, iterations=1)

# Step 4: Extract contours
contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Find the convex hull for each contour
convex_hulls = [cv2.convexHull(contour) for contour in contours]

# Step 6: Create a mask from the convex hulls
# hull_mask = np.zeros_like(bw_image)
# cv2.drawContours(hull_mask, convex_hulls, -1, (255), thickness=cv2.FILLED)

# Step 7: Dilate the mask to create the offset contour
# kernel = np.ones((20, 20), np.uint8)  # Define a square kernel with a size of 10 pixels
# dilated_hull_mask = cv2.dilate(hull_mask, kernel, iterations=1)

# Step 8: Extract the offset contours from the dilated mask
# offset_contours, _ = cv2.findContours(dilated_hull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Number of original contours:", len(convex_hulls[0]))
# print("Number of offset contours:", len(offset_contours[0]))

# Step 9: Draw the convex hulls and the offset contours on the image
convex_hull_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)  # Convert binary image to BGR
cv2.drawContours(convex_hull_image, convex_hulls, -1, (0, 255, 0), 2)  # Draw convex hulls in green
# cv2.drawContours(convex_hull_image, offset_contours, -1, (255, 0, 0), 2)  # Draw offset contours in red

# Step 10: Draw parallel lines on the image with the offset contour
gap = 22
lines_image = draw_parallel_lines(convex_hull_image, gap)

# Step 11: Find intersection points of lines with offset contours
height, width = bw_image.shape
intersection_points = find_intersection_points(convex_hulls, gap, height, width)

np_points = np.array(intersection_points)
np_z_column = np.ones((len(np_points), 1))
np_points = np.append(np_points, np_z_column, axis=1)
# rearrange_nodes(intersection_points)

print(np_points)


# Mark intersection points on the image
for point in intersection_points:
    cv2.circle(lines_image, point, 3, (0, 0, 255), -1)  # Red color for intersection points


rearrange_nodes(intersection_points)
path_image = draw_path(intersection_points, convex_hull_image)


# Display the original, convex hull, and lines images
# plt.figure(figsize=[15, 5])
# plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(232), plt.imshow(cv2.cvtColor(convex_hull_image, cv2.COLOR_BGR2RGB)), plt.title('Convex Hulls and Offset Contours')
# plt.subplot(233), plt.imshow(cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB)), plt.title('Parallel Lines with Intersections')
# plt.subplot(234), plt.imshow(cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB)), plt.title('Path')
# plt.show()


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
ax.set_zlim([0, 4])

# Adjust the view angle
ax.view_init(elev=90, azim=-90)

# Hide the z-axis
# ax.set_zticks([])
ax.plot(np_points[:,0], np_points[:,1], np_points[:,2], '#00FF00')

# Show the plot
plt.show()