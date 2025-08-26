from deep_learning_functions.FetchModels import *
import numpy as np
from deep_learning_functions.PlottingFunctions import fetch_existing_plot_filepaths
from deep_learning_functions.ExplainerClass import generate_lime_heatmap_and_explanation
from deep_learning_functions.FetchModels import load_model_from_weights
from deep_learning_functions.PreprocessingClasss import preprocess_image
from deep_learning_functions.LoadConfig import *


def predict_image(model, input_data, class_names=None, target_size=(180, 180)):
    """
    Predicts the class of an input image using the provided model.

    Args:
        model: Trained Keras model
        input_data: File path (str) or image array (numpy array)
        class_names: List of class names corresponding to model output indices
        target_size: Tuple (height, width) for resizing

    Returns:
        dict: {
            'predicted_class_index': int,
            'predicted_class_name': str (or None if class_names not provided),
            'confidence': float (0-1),
            'class_probabilities': np.array of probabilities,
            'class_probability_dict': dict of {class_name: probability} (if class_names provided)
        }
    """
    try:
        # Handle different input types
        if isinstance(input_data, str):
            # Input is file path
            img = Image.open(input_data).convert('RGB')
        elif isinstance(input_data, np.ndarray):
            # Input is numpy array
            if input_data.dtype != np.uint8:
                input_data = input_data.astype(np.uint8)
            img = Image.fromarray(input_data)
        else:
            raise TypeError("input_data must be file path (str) or image array (numpy array)")

        # Preprocess image
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Make prediction
        probabilities = model.predict(img_batch)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        # Prepare results
        result = {
            'predicted_class_index': predicted_index,
            'predicted_class_name': None,
            'confidence': confidence,
            'class_probabilities': probabilities,
            'class_probability_dict': None
        }

        # Add class names if provided
        if class_names:
            result['predicted_class_name'] = class_names[predicted_index]
            result['class_probability_dict'] = {
                class_name: float(prob)
                for class_name, prob in zip(class_names, probabilities)
            }

        return result

    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        return {
            'error': f"Prediction failed: {error_type} - {error_details}",
            'input_type': str(type(input_data))
        }
