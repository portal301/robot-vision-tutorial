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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir,"image")
    
    convert_to_binary( os.path.join(image_dir,"lena.png"),  os.path.join(image_dir,"lena_binary.png"))
