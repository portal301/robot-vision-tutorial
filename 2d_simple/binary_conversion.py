import cv2
import os

def convert_to_binary(image_path, output_path, threshold=128):
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply a binary threshold to convert the image to black and white
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Save the result
    cv2.imwrite(output_path, binary_img)

if __name__ == "__main__":
    # Convert lena.png to a binary image and save as binary_lena.png
    # join path with the /image folder
    
    convert_to_binary( os.path.join("image","lena.png"),  os.path.join("image","lena_binary.png"))
