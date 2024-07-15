import cv2
import os
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
                    if (x_intersection, y) not in points:
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

if __name__ == '__main__':
    # Load the image
    image_path = 'diagram_bg_white.png'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image = cv2.imread(os.path.join(script_dir,"image",image_path))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to black and white (binary) image
    _, bw_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the binary image if necessary to make the object white and background black
    bw_image = cv2.bitwise_not(bw_image)

    # Create the offset contours by dilating the binary image
    kernel = np.ones((20, 20), np.uint8)  # Define a square kernel with a size of 10 pixels
    dilated_image = cv2.dilate(bw_image, kernel, iterations=1)

    # Extract contours
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the convex hull for each contour
    convex_hulls = [cv2.convexHull(contour) for contour in contours]

    # Draw the convex hulls and the offset contours on the image
    convex_hull_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)  # Convert binary image to BGR
    cv2.drawContours(convex_hull_image, convex_hulls, -1, (0, 255, 0), 4)  # Draw convex hulls in green

    # Draw parallel lines on the image with the offset contour
    gap = 20
    lines_image = draw_parallel_lines(convex_hull_image, gap)

    # Find intersection points of lines with offset contours
    height, width = bw_image.shape
    intersection_points = find_intersection_points(convex_hulls, gap, height, width)

    # Mark intersection points on the image
    for point in intersection_points:
        cv2.circle(lines_image, point, 3, (0, 0, 255), -1)  # Red color for intersection points

    rearrange_nodes(intersection_points)
    path_image = draw_path(intersection_points, convex_hull_image)

    # Display the original, convex hull, and lines images
    plt.figure(figsize=[10, 10])
    plt.subplot(221), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(222), plt.imshow(cv2.cvtColor(convex_hull_image, cv2.COLOR_BGR2RGB)), plt.title('Convex Hulls and Offset Contours')
    plt.subplot(223), plt.imshow(cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB)), plt.title('Parallel Lines with Intersections')
    plt.subplot(224), plt.imshow(cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB)), plt.title('Path')
    plt.show()
