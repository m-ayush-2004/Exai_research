import numpy as np
from PIL import Image
import io

def preprocess_image(input_data, target_size=(180, 180)):
    """
    Preprocess an image from a file path, numpy array, or file-like object.
    Returns a batch of shape (1, H, W, 3) with normalized RGB data.
    """
    try:
        if isinstance(input_data, str):
            # Input is file path
            img = Image.open(input_data).convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            return img_batch

        elif isinstance(input_data, np.ndarray):
            # Input is numpy array
            if input_data.dtype != np.uint8:
                input_data = input_data.astype(np.uint8)
            img = Image.fromarray(input_data)
            img = img.convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            return img_batch

        else:
            # Assume input_data is a file-like object (e.g., uploaded file)
            image_bytes = input_data.read()
            img = Image.open(io.BytesIO(image_bytes)).convert('L')  # Grayscale
            img = img.resize(target_size)
            img_normalized = np.array(img) / 255.0
            image = img_normalized.reshape(target_size[0], target_size[1], 1)
            single_image_rgb = np.repeat(image, 3, axis=-1)  # Grayscale to RGB
            single_image_rgb = np.expand_dims(single_image_rgb, axis=0)  # Batch dimension
            return single_image_rgb

    except Exception as e:
        raise RuntimeError(f"Image preprocessing failed: {e}")
