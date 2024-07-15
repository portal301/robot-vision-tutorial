import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to draw parallel lines
def draw_parallel_lines(image, gap):
    height, width = image.shape[:2]
    lines_image = image.copy()
    linewidth = max(image.shape)//200

    for y in range(0, height, gap):
        cv2.line(lines_image, (0, y), (width, y), (128, 128, 128), linewidth)
    
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
    linewidth = max(image.shape)//200
    for i in range(len(nodes)-1):
        cv2.line(image_path, nodes[i], nodes[i+1], (0, 0, 255), linewidth)
    return image_path

def get_convexhull(bw_image, offset=0):

    # Create the offset contours by dilating the binary image
    kernel = np.ones((offset, offset), np.uint8)  # Define a square kernel with a size of 10 pixels
    dilated_image = cv2.dilate(bw_image, kernel, iterations=1)

    # Extract contours
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the convex hull for each contour
    convex_hulls = [cv2.convexHull(contour) for contour in contours]
    return convex_hulls

def draw_contour(image, contour):
    image_contour = image.copy()
    linewidth = max(image.shape)//200

    image_contour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert binary image to BGR
    cv2.drawContours(image_contour, contour, -1, (0, 255, 0), linewidth)  # Draw convex hulls in green
    return image_contour

def filter_convex_hulls(convex_hulls, min_size=50):
    filtered_hulls = []
    for hull in convex_hulls:
        # Calculate the bounding rectangle for each convex hull
        x, y, w, h = cv2.boundingRect(hull)
        # Calculate the diameter as the maximum dimension of the bounding rectangle
        diameter = max(w, h)
        if diameter >= min_size:
            filtered_hulls.append(hull)
    return filtered_hulls
    

if __name__ == '__main__':
    # Load the image
    # image_path = 'diagram_bg_white.png'
    image_path = "realSample.jpg"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    image = cv2.imread(os.path.join(script_dir,"image",image_path))
    print("image size: ", image.shape)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert to black and white (binary) image
    _, bw_image_otsu = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # OTSU sets the threshold automatically
    _, bw_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    # Invert the binary image if necessary to make the object white and background black
    bw_image_otsu = cv2.bitwise_not(bw_image_otsu)
    bw_image = cv2.bitwise_not(bw_image)
    
    gap = max(image.shape)//40
    convex_hulls_otus = get_convexhull(bw_image_otsu, offset=gap)
    convex_hulls = get_convexhull(bw_image, offset=gap)
    convex_hulls_filtered = filter_convex_hulls(convex_hulls, min_size=100)

    # Draw the convex hulls and the offset contours on the image
    convex_hull_image_otsu=draw_contour(bw_image_otsu, convex_hulls_otus)
    convex_hull_image=draw_contour(bw_image, convex_hulls)
    convex_hull_image_filtered=draw_contour(bw_image, convex_hulls_filtered)

    # Draw parallel lines on the image with the offset contour
    lines_image_otsu = draw_parallel_lines(convex_hull_image_otsu, gap)
    lines_image = draw_parallel_lines(convex_hull_image, gap)
    lines_image_filtered = draw_parallel_lines(convex_hull_image_filtered, gap)

    # Find intersection points of lines with offset contours
    height, width = bw_image.shape
    intersection_points_otsu = find_intersection_points(convex_hulls_otus, gap, height, width)
    intersection_points = find_intersection_points(convex_hulls, gap, height, width)
    intersection_points_filtered = find_intersection_points(convex_hulls_filtered, gap, height, width)


    # Mark intersection points on the image
    for point in intersection_points_otsu:
        cv2.circle(lines_image_otsu, point, gap//5, (0, 0, 255), -1)
    rearrange_nodes(intersection_points_otsu)

    for point in intersection_points:
        cv2.circle(lines_image, point, gap//5, (0, 0, 255), -1)  # Red color for intersection points
    rearrange_nodes(intersection_points)

    for point in intersection_points_filtered:
        cv2.circle(lines_image_filtered, point, gap//5, (0, 0, 255), -1)
    rearrange_nodes(intersection_points_filtered)

    path_image_otsu = draw_path(intersection_points_otsu, convex_hull_image_otsu)
    path_image = draw_path(intersection_points, convex_hull_image)
    path_image_filtered = draw_path(intersection_points_filtered, convex_hull_image_filtered)

    # Display the original, convex hull, and lines images
    plt.figure(figsize=[20, 10])
    plt.subplot(3,4,1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(3,4,2), plt.imshow(cv2.cvtColor(convex_hull_image_otsu, cv2.COLOR_BGR2RGB)), plt.title('Convex Hulls and Offset Contours')
    plt.subplot(3,4,3), plt.imshow(cv2.cvtColor(lines_image_otsu, cv2.COLOR_BGR2RGB)), plt.title('Parallel Lines with Intersections')
    plt.subplot(3,4,4), plt.imshow(cv2.cvtColor(path_image_otsu, cv2.COLOR_BGR2RGB)), plt.title('Path')
    plt.subplot(3,4,5), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(3,4,6), plt.imshow(cv2.cvtColor(convex_hull_image, cv2.COLOR_BGR2RGB)), plt.title('Convex Hulls and Offset Contours')
    plt.subplot(3,4,7), plt.imshow(cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB)), plt.title('Parallel Lines with Intersections')
    plt.subplot(3,4,8), plt.imshow(cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB)), plt.title('Path')
    plt.subplot(3,4,9), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(3,4,10), plt.imshow(cv2.cvtColor(convex_hull_image_filtered, cv2.COLOR_BGR2RGB)), plt.title('Convex Hulls and Offset Contours')
    plt.subplot(3,4,11), plt.imshow(cv2.cvtColor(lines_image_filtered, cv2.COLOR_BGR2RGB)), plt.title('Parallel Lines with Intersections')
    plt.subplot(3,4,12), plt.imshow(cv2.cvtColor(path_image_filtered, cv2.COLOR_BGR2RGB)), plt.title('Path')
    plt.show()
