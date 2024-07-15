# pip install qrcode[pil]
import qrcode
from PIL import Image

# Data to encode
data = "https://affine-robotics.com"

# Create a QR Code object
qr = qrcode.QRCode(
    version=1,  # controls the size of the QR Code, try different values to get different sizes
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # controls the error correction used for the QR Code
    box_size=10,  # controls how many pixels each “box” of the QR code is
    border=4,  # controls how many boxes thick the border should be
)

# Add data to the QR Code object
qr.add_data(data)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image(fill_color="black", back_color="white")

# Save the image
img.save("qrcode_affine_robotics.png")

# Optionally, display the QR code image
img.show()
