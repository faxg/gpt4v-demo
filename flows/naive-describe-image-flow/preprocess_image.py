from promptflow import tool
from promptflow.contracts.multimedia import Image

import cv2
import numpy as np


from io import BytesIO
import base64
from PIL import Image as PILImage, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from PIL.ImageFilter import Kernel

@tool
def preprocess_image_tool(input_image: Image) -> Image:
    # Convert base64 string to a PIL Image
    input_data = input_image.to_base64()
    input_pil_image = PILImage.open(BytesIO(base64.b64decode(input_data)))

    # Convert the image to grayscale
    img_gray = input_pil_image.convert('L')

    # Enhance the contrast of the image
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(2.0)  # Increase contrast

    # Apply a median filter for noise reduction
    img_denoised = img_contrast.filter(ImageFilter.MedianFilter(size=3))


    # Sharpen the image
    img_sharpened = img_denoised.filter(ImageFilter.SHARPEN)
    
    # Convert image to black and white with strong contrast
    img_bw = img_sharpened.point(lambda x: 0 if x < 128 else 255, '1')

    # Dilation operation
    # Define a dilation kernel
    #dilation_kernel = Kernel((3, 3), [1, 1, 1, 1, 1, 1, 1, 1, 1], 1, 0)
    #img_dilated = img_bw.filter(dilation_kernel)


    # Invert the colors of the image
    img_inverted = ImageOps.invert(img_bw)


    # Convert OpenCV Image back to PIL Image
    img_result = img_inverted



    #
    ## Now draw bounding boxes on top
    #
    # Create a draw object
    draw = ImageDraw.Draw(img_result)
    # Define the coordinates of the bounding box
    # (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
    x1, y1, x2, y2 = 50, 50, 100, 100
    # Draw the bounding box
    draw.rectangle([x1, y1, x2, y2], outline="blue")






    # Save the processed image to a BytesIO object
    byte_io = BytesIO()
    img_result.save(byte_io, format='JPEG')
    byte_io.seek(0)

    # Convert BytesIO to bytes
    processed_bytes = byte_io.getvalue()

    # Create a new Image object with the processed image
    result = Image(value=processed_bytes, mime_type="image/jpeg")

    return result