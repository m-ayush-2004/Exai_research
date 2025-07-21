import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries, quickshift
from io import BytesIO
import shap
from lime import lime_image
import tensorflow as tf
from skimage.measure import find_contours
import seaborn as sns
import os
# from deep_learning_functions.CompiledClass import predict_image
from deep_learning_functions.LoadConfig import load_model_config

import os
import re

def get_next_plot_number(directory):
    """
    Finds all filenames in the directory matching "plot-{number}.png" and returns the next available increasing number.

    Args:
        directory (str): Path to the directory to search.

    Returns:
        int: The next plot number to use (i.e., 1 greater than the current max; returns 1 if none exists).
    """
    # List all files in the directory
    files = os.listdir(directory)
    # Regex pattern to match plot-X.png where X is a number
    pattern = re.compile(r'plot-(\d+)\.png$')
    numbers = []
    for filename in files:
        match = pattern.match(filename)
        if match:
            numbers.append(int(match.group(1)))
    if numbers:
        return max(numbers) + 1
    else:
        return 1

# Example usage:
# Suppose your directory contains: plot-1.png, plot-2.png, plot-5.png,
# then get_next_plot_number('your/plots/dir') will return 6.
# Generate LIME heatmap and explanation for a given image
def generate_lime_heatmap_and_explanation(model, model_name , disease_name, image, target_label=1, num_segments_to_select=3, save_path='front end/static/shap_plots/dl_res.png', shap_path='front end/static/shap/' , lime_path='front end/static/lime/'):
    os.makedirs(shap_path, exist_ok=True)
    os.makedirs(lime_path, exist_ok=True)
    explainer = lime_image.LimeImageExplainer(kernel_width=0.10, random_state=42)
    # Use quickshift to find superpixels
    superpixels = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)+10

    def model_predict_proba(image_array):
        # image_array_resized = tf.image.resize(image_array, (180, 180))  # Resize to match model input
        # predictions = model.predict(image_array_resized, verbose=0)
        # # print("\033[92m[INFO::]\033[0mpredictions :",predictions)
        # # predictions = predictions['dense_7']
        # # print("\033[92m[INFO::]\033[0mpredictions after extractions:",predictions)
        # if np.array(predictions).ndim == 1:
        #     return np.stack([predictions, 1 - predictions], axis=1)
        # Ensure input is a batch (add batch dimension if needed)
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)
        # Convert to Tensor
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        # Resize to model input shape
        image_tensor_resized = tf.image.resize(image_tensor, (180, 180))
        # Run prediction on GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        device = '/GPU:0' if gpus else '/CPU:0'
        # print(device)
        with tf.device(device):
            predictions = model.predict(image_tensor_resized, verbose=0)
        # Handle binary classification output shape
        if np.array(predictions).ndim == 1:
            return np.stack([1 - predictions, predictions], axis=1)
        return predictions

    explanation = explainer.explain_instance(
        image,
        model_predict_proba,
        top_labels=0,
        hide_color=1,
        distance_metric='cosine', 
        num_samples=150,
        segmentation_fn=lambda x: slic(x, n_segments=np.unique(superpixels).shape[0])
    )
    
    # Extract segment weights for all segments
    num_segments = np.max(explanation.segments) + 1  # Total number of segments
    segment_weights = np.zeros(num_segments)  # Initialize weights array

    # Check if the target_label is in the explanation's top labels
    if target_label :
        for segment_id, weight in explanation.local_exp[target_label]:
            segment_weights[segment_id] += np.abs(weight)
        # Determine dynamic max_weight based on the desired number of segments
        sorted_weights = np.sort(segment_weights)
        if num_segments_to_select > len(sorted_weights):
            dynamic_max_weight = 0  # If we want more segments than available, set to zero
        else:
            print(sorted_weights)
            print(sorted_weights[-num_segments_to_select])
            dynamic_max_weight = sorted_weights[-num_segments_to_select]  # Get the weight at the position
        
        print(f"Dynamic max_weight threshold: {dynamic_max_weight}")
        if(dynamic_max_weight<0):
            temp, mask = explanation.get_image_and_mask(
                label=target_label,
                positive_only=True,
                hide_rest=True,
                num_features=100,  # Use dynamic max_weight here
                min_weight=dynamic_max_weight# Use dynamic max_weight here
            )
        else:
            temp, mask = explanation.get_image_and_mask(
                label=target_label,
                # negative_only=True,
                # positive_only= False,
                hide_rest=True,
                num_features=100,
                min_weight=dynamic_max_weight# Use dynamic max_weight here
            )
        norm_weights = (segment_weights - np.min(segment_weights)) / (np.max(segment_weights) - np.min(segment_weights))
        cmap = sns.color_palette("viridis", as_cmap=True)
        
        segment_info = []
        
        for segment_id in range(num_segments):
            mask_segment = (explanation.segments == segment_id)
            if np.any(mask_segment):
                # Find contours of the current segment
                contours = find_contours(mask_segment.astype(float), 0.5)  # Find contours at a constant value
                
                # Get color from the colormap based on normalized weight
                color = cmap(norm_weights[segment_id])
                
                # Store segment information (ID, weight, coordinates)
                segment_info.append({
                    'id': segment_id,
                    'weight': segment_weights[segment_id],
                    'contours': contours,
                    'color': f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.5)'  # RGBA format for Plotly
                })

        # print(segment_weights)
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 5, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 5, 2)
        heatmap_with_boundaries = mark_boundaries(temp, mask, color=(0, 1, 0), mode='thick')
        plt.imshow(heatmap_with_boundaries)
        plt.title("Produced Heatmap")
        plt.axis("off")

        plt.subplot(1, 5, 3)
        heatmap_with_boundaries = mark_boundaries(image, explanation.segments)
        plt.imshow(heatmap_with_boundaries)
        plt.title("Heatmap with Boundaries")
        plt.axis("off")

        plt.subplot(1, 5, 4)
        plt.imshow(mask, cmap='gray')
        plt.title("Explanation Mask")
        plt.axis("off")

        # Plot all segments with distinct colors
        segmented_image = np.zeros((*image.shape[:2], 3), dtype=np.float32)  # Initialize segmented image with zeros and RGB channels

        for segment_id in range(num_segments):
            mask_segment = (explanation.segments == segment_id)  # Create mask for the current segment
            if np.any(mask_segment):
                color = cmap(norm_weights[segment_id])  # Get the RGB color from the colormap
                segmented_image[mask_segment] = np.array(color[:3])  # Assign the RGB color to the mask segment

        plt.subplot(1, 5, 5)
        plt.imshow(segmented_image)
        plt.title("Colored Segments")
        plt.axis("off")
        # Save the figure to the specified path
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
       
        nex = get_next_plot_number(lime_path)
        plt.savefig(os.path.join(lime_path, f"plot-{nex}.png"))

        plt.savefig(save_path)
        print(f"Heatmap saved at: {save_path}")
        print(f"Lime Explanation Heatmap saved at: {lime_path}")
        
        # plt.show()
        
    else:
        print(f"Label {target_label} not found in top labels.")
    # print(segment_info)
    # predictions = predict_image(model, image, class_names=load_model_config()[model_name][disease_name]["class_name"], target_size=(180, 180))
    # print(predictions)
    lime_path=os.path.join(lime_path, f"plot-{nex}.png")
    # Build masker and SHAP explainer
    masker = shap.maskers.Image('blur(15,15)', image.shape)
    explainer = shap.Explainer(model, masker)
    shap_values = explainer(np.expand_dims(image, 0))
    sv = shap_values.values

    # Automatically determine number of classes
    if sv.ndim == 5:
        n_classes = sv.shape[-1]
    elif sv.ndim == 4:
        n_classes = sv.shape[-1] if sv.shape[-1] > 1 else 1
    else:
        n_classes = 1

    # Prepare plot grid: Original + one for each class
    n_cols = n_classes + 1
    plt.figure(figsize=(5 * n_cols, 8))

    # 1. Original image (panel 0)
    plt.subplot(1, n_cols, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')

    # 2. For each class, show SHAP overlay (panels 1...N)
    for i in range(n_classes):
        # Get the SHAP class attributions for this class
        if sv.ndim == 5:
            shap_map = sv[0, :, :, :, i]
        elif sv.ndim == 4 and n_classes > 1:
            shap_map = sv[0, :, :, i]
        elif sv.ndim == 4 and sv.shape[-1] in (1, 3):
            shap_map = sv[0]
        elif sv.ndim == 3:
            shap_map = sv[0]
        else:
            shap_map = sv[0]
        # Convert to grayscale (mean across RGB) if needed
        if shap_map.ndim == 3 and shap_map.shape[-1] in [1, 3]:
            shap_heatmap = shap_map.mean(axis=-1)
        else:
            shap_heatmap = shap_map
        # Normalize to [0,1] for display
        if shap_heatmap.max() > shap_heatmap.min():
            shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min())
        plt.subplot(1, n_cols, i + 2)
        plt.imshow(image, alpha=0.50)
        plt.imshow(shap_heatmap, cmap='RdBu_r', alpha=0.65)
        plt.title(f"Class: {i}")
        plt.axis('off')

    plt.tight_layout()
    next_num = get_next_plot_number(shap_path)
    save_file = os.path.join(shap_path, f"plot-{next_num}.png")
    plt.savefig(save_file)
    plt.close()
    print(f"SHAP explanation grid saved at: {save_path}")
    print(f"SHAP Explanation Heatmap saved at: {save_file}")
    return segment_info, save_path, save_file, lime_path
