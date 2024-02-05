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
    input_cv_image = input_cv_image[:, :, ::-1].copy() 

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(input_cv_image, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    img_normalized = cv2.normalize(img_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_contrast = clahe.apply(np.array(img_normalized, dtype = np.uint8))

    # Convert the processed image back to PIL format
    img_pil = PILImage.fromarray(cv2.cvtColor(img_contrast, cv2.COLOR_BGR2RGB))

    # Save the processed image to a BytesIO object
    byte_io = BytesIO()
    img_pil.save(byte_io, format='JPEG')
    byte_io.seek(0)

    # Convert BytesIO to bytes
    processed_bytes = byte_io.getvalue()

    # Create a new Image object with the processed image
    result = Image(value=processed_bytes, mime_type=input_image._mime_type)

    return result