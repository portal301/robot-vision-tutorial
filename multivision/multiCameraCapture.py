import cv2
import threading
import os
import time
import numpy as np
class MultiCameraCapture:
    def __init__(self):
        pass

    def scan_available_cameras(self, max_cameras=10):
        available_cameras = []
        for camera_index in range(max_cameras):
            try:
                cap = cv2.VideoCapture(camera_index)  
                if cap.isOpened():
                    print(f"cap info: {cap}")
                    available_cameras.append(camera_index)
                    cap.release()
            except:
                print(f"Camera {camera_index} is not available.")
        return available_cameras

    def load_cameras(self, camera_indices, save_folder="./image"):
        self.camera_indices = camera_indices
        self.captures = [cv2.VideoCapture(i) for i in camera_indices]
        self.save_folder = save_folder

        # Ensure save folder exists
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def capture_image(self, cap, camera_index, warmup_frames=5, image_size=(640, 480)):
        # Discard initial frames to allow the camera to stabilize
        for _ in range(warmup_frames):
            cap.read()

        # Set the camera to a high resolution (e.g., 1920x1080 or higher depending on your camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])

        # Capture the actual image
        ret, frame = cap.read()
        if ret:
            # Save the captured image
            image_path = os.path.join(self.save_folder, f"cam_{camera_index}_image.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image from Camera {camera_index} saved to {image_path}")
        else:
            print(f"Failed to capture from Camera {camera_index}")

    def capture_from_all(self):
        threads = []
        for i, cap in enumerate(self.captures):
            # Create and start a new thread for each camera
            t = threading.Thread(target=self.capture_image, args=(cap, self.camera_indices[i]))
            threads.append(t)
            t.start()

        # Join all threads (wait for all to finish)
        for t in threads:
            t.join()

    def capture_calibration_images(self, camera_index, num_images=10, save_folder="./image", warmup_frames=5, interval=3, image_size=(640, 480)):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        cap = cv2.VideoCapture(camera_index)

        # Query the current resolution (the default is usually 640x480)
        default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"default image size: {int(default_width)}x{int(default_height)}")

        # Set the camera to a high resolution (e.g., 1920x1080 or higher depending on your camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])

        # Check if the resolution was set successfully
        width_px = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_px = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Set resolution: {int(width_px)}x{int(height_px)}")

        for _ in range(warmup_frames):
            cap.read()

        for i in range(num_images):
            time_start = time.time()
            time_elasped_int = -1
            while time_elasped_int < interval:
                time_elasped = time.time() - time_start
                if time_elasped_int < int(time_elasped):
                    time_elasped_int = int(time_elasped)
                    print(f"Capturing in {interval-time_elasped_int} seconds...", end="\r")

                ret, frame = cap.read()
                if ret:

                    if time_elasped_int == interval:
                        # Save the captured image
                        image_path = os.path.join(save_folder, f"{i}.jpg")
                        cv2.imwrite(image_path, frame)
                        print(f"Image saved to {image_path}")
                        frame = np.zeros((image_size[0], image_size[1], 3), np.uint8)

                    resized_frame = cv2.resize(frame, (640, 480))
                    cv2.imshow("Camera", resized_frame)
                    # Break the loop on key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                else:
                    print(f"Failed to capture from Camera {camera_index}")
                    break



        # for i in range(num_images):
        #     for j in range(interval):
        #         print(f"Capturing in {interval-j} seconds...", end="\r")
        #         time.sleep(1)
        #     # Capture the actual image
        #     ret, frame = cap.read()
        #     if ret:
        #         cv2.imshow("Camera", frame)

        #         # Save the captured image
        #         image_path = os.path.join(save_folder, f"{i}.jpg")
        #         cv2.imwrite(image_path, frame)
        #         print(f"Image saved to {image_path}")
        #     else:
        #         print(f"Failed to capture from Camera {camera_index}")
        #         break


# # Open the camera (0 is the default camera index, change if needed)
# cam = cv2.VideoCapture(0)

# # Query the current resolution (the default is usually 640x480)
# default_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
# default_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Default resolution: {int(default_width)}x{int(default_height)}")

# # Set the camera to a high resolution (e.g., 1920x1080 or higher depending on your camera)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# # Check if the resolution was set successfully
# width_px = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
# height_px = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Set resolution: {int(width_px)}x{int(height_px)}")

# # Capture a frame
# ret, frame = cam.read()

# # Check if the frame was successfully captured
# if ret:
#     # Display the captured frame
#     cv2.imshow("Full Resolution Image", frame)
#     cv2.waitKey(0)  # Wait for any key press to close the window
# else:
#     print("Failed to capture the image")

# # Release the camera and close all windows
# cam.release()
# cv2.destroyAllWindows()





    def release_cameras(self):
        for cap in self.captures:
            cap.release()