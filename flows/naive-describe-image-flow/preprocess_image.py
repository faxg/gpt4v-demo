from promptflow import tool
from promptflow.contracts.multimedia import Image

from io import BytesIO
import base64
import cv2
import numpy as np
from PIL import Image as PILImage


@tool
def preprocess_image_tool(input_image: Image) -> Image:
    # Convert base64 string to a PIL Image
    input_data = input_image.to_base64()
    input_pil_image = PILImage.open(BytesIO(base64.b64decode(input_data)))

    # Convert PIL Image to OpenCV format
    input_cv_image = np.array(input_pil_image)
    # Check if the image has 3 channels (color image)
    if len(input_cv_image.shape) == 3:
        # Reverse the channels
        input_cv_image = input_cv_image[:, :, ::-1].copy() 

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(input_cv_image, cv2.COLOR_BGR2GRAY)
    else:
        # If the image is already grayscale, no need to convert
        img_gray = input_cv_image

    # Normalize the image
    img_normalized = img_gray #cv2.normalize(img_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_contrast = clahe.apply(img_normalized)

    # Convert the processed image back to PIL format
    img_pil = PILImage.fromarray(cv2.cvtColor(img_contrast, cv2.COLOR_BGR2RGB))

    # Save the processed image to a BytesIO object
    byte_io = BytesIO()
    img_pil.save(byte_io, format='JPEG')
    byte_io.seek(0)

    # Convert BytesIO to bytes
    processed_bytes = byte_io.getvalue()

    # Create a new Image object with the processed image
    result = input_image #Image(value=processed_bytes, mime_type="image/jpeg")

    return result