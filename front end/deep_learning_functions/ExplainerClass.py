import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries, quickshift
from io import BytesIO
from lime import lime_image
import tensorflow as tf
from skimage.measure import find_contours
import seaborn as sns
import os
# from deep_learning_functions.CompiledClass import predict_image
from deep_learning_functions.LoadConfig import load_model_config
# Generate LIME heatmap and explanation for a given image
def generate_lime_heatmap_and_explanation(model, model_name , disease_name, image, target_label=1, num_segments_to_select=3, save_path='front end/static/shap_plots/dl_res.png'):
    explainer = lime_image.LimeImageExplainer(kernel_width=0.10, random_state=42)
    # Use quickshift to find superpixels
    superpixels = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)+10

    def model_predict_proba(image_array):
        image_array_resized = tf.image.resize(image_array, (180, 180))  # Resize to match model input
        predictions = model.predict(image_array_resized, verbose=0)
        # print("\033[92m[INFO::]\033[0mpredictions :",predictions)
        # predictions = predictions['dense_7']
        # print("\033[92m[INFO::]\033[0mpredictions after extractions:",predictions)
        if np.array(predictions).ndim == 1:
            return np.stack([predictions, 1 - predictions], axis=1)
        return predictions

    explanation = explainer.explain_instance(
        image,
        model_predict_proba,
        top_labels=0,
        hide_color=1,
        distance_metric='cosine', 
        num_samples=500,
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
            dynamic_max_weight = sorted_weights[-num_segments_to_select]  # Get the weight at the position
        
        print(f"Dynamic max_weight threshold: {dynamic_max_weight}")
        if(dynamic_max_weight<0):
            temp, mask = explanation.get_image_and_mask(
                label=target_label,
                positive_only=True,
                hide_rest=True,
                num_features=100,  # Use dynamic max_weight here
            )
        else:
            temp, mask = explanation.get_image_and_mask(
                label=target_label,
                # negative_only=True,
                # positive_only= False,
                hide_rest=True,
                num_features=100,
                # min_weight=dynamic_max_weight,# Use dynamic max_weight here
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
        
        plt.savefig(save_path)
        print(f"Heatmap saved at: {save_path}")
        
        # plt.show()
        
    else:
        print(f"Label {target_label} not found in top labels.")
    # print(segment_info)
    # predictions = predict_image(model, image, class_names=load_model_config()[model_name][disease_name]["class_name"], target_size=(180, 180))
    # print(predictions)
    return segment_info, save_path
