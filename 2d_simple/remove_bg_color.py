import cv2
import os
import numpy as np


def convert_green_to_color(img, conversion_color=[0, 0, 0]):

    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 210, 210])
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    converted_img = img.copy()
    # Change green pixels to conversion_color
    converted_img[mask != 0] = conversion_color
    
    return converted_img


def convert_to_binary(img, threshold=128):
    # Apply a binary threshold to convert the image to black and white
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img
    
def invert_black_and_white(img):
    # Invert the image
    inverted_img = cv2.bitwise_not(img)
    return inverted_img

if __name__ == "__main__":
    img = cv2.imread(os.path.join("image","grating_original.png"))
    img_green2black = convert_green_to_color(img, conversion_color=[0, 0, 0])   # convert green to black
    img_green2white = convert_green_to_color(img, conversion_color=[255, 255, 255])  # convert green to white
    img_binary = convert_to_binary(img_green2black)
    inverted_img = invert_black_and_white(img_binary)

    cv2.imwrite(os.path.join("image","grating_green2black.png"), img_green2black)
    cv2.imwrite(os.path.join("image","grating_green2white.png"), img_green2white)
    cv2.imwrite(os.path.join("image","grating_binary.png"), img_binary)
    cv2.imwrite(os.path.join("image","grating_inverted.png"), inverted_img)
