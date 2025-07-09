import os
import numpy as np
import cv2, json
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.applications import InceptionV3, ResNet50V2, NASNetLarge
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from keras.models import load_model
from lime import lime_image
from skimage.measure import find_contours
import seaborn as sns
from skimage.segmentation import slic, mark_boundaries, quickshift


# Load available models from config file
def load_model_config2(config_path='front end/config/config2.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('models', {})
    return {}

    
def generate_vibrant_colors(num_colors):
    """Generate a list of vibrant colors."""
    colors = plt.cm.get_cmap("hsv", num_colors)  # Use HSV colormap for vibrant colors
    return [colors(i)[:3] for i in range(num_colors)]  # Return RGB values

from keras.layers import TFSMLayer

# Load model from weights directory
def load_model_from_weights(model_name, disease_name, config_path='front end/config/config2.json'):
    models = load_model_config2(config_path)
    if model_name in models and disease_name in models[model_name]:
        weights_path = models[model_name][disease_name]
        model = TFSMLayer(weights_path, call_endpoint='serving_default')
        return model
    else:
        raise ValueError(f"Model or disease not found in configuration: {model_name}, {disease_name}")

# Preprocess image for prediction
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

# Generate LIME heatmap and explanation for a given image
def generate_lime_heatmap_and_explanation(model, image, target_label=1, num_segments_to_select=8, save_path='front end/static/shap_plots/dl_res.png'):
    explainer = lime_image.LimeImageExplainer(kernel_width=0.10, random_state=42)
    # Use quickshift to find superpixels
    superpixels = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

    def model_predict_proba(image_array):
        image_array_resized = tf.image.resize(image_array, (180, 180))  # Resize to match model input
        predictions = model.predict(image_array_resized, verbose=0)
        # print("\033[92m[INFO::]\033[0mpredictions :",predictions)
        # predictions = predictions['dense_7']
        # print("\033[92m[INFO::]\033[0mpredictions after extractions:",predictions)
        if np.array(predictions).ndim == 1:
            return np.stack([1 - predictions, predictions], axis=1)
        return predictions

    explanation = explainer.explain_instance(
        image,
        model_predict_proba,
        top_labels=2,
        hide_color=1,
        distance_metric='cosine', 
        num_samples=150,
        segmentation_fn=lambda x: slic(x, n_segments=np.unique(superpixels).shape[0])
    )
    
    # Extract segment weights for all segments
    num_segments = np.max(explanation.segments) + 1  # Total number of segments
    segment_weights = np.zeros(num_segments)  # Initialize weights array
    
    # Check if the target_label is in the explanation's top labels
    if target_label in explanation.top_labels:
        for segment_id, weight in explanation.local_exp[target_label]:
            segment_weights[segment_id] += np.abs(weight)
        # Determine dynamic max_weight based on the desired number of segments
        sorted_weights = np.sort(segment_weights)
        if num_segments_to_select > len(sorted_weights):
            dynamic_max_weight = 0  # If we want more segments than available, set to zero
        else:
            dynamic_max_weight = sorted_weights[-num_segments_to_select]  # Get the weight at the position
        
        print(f"Dynamic max_weight threshold: {dynamic_max_weight:.4f}")
        
        temp, mask = explanation.get_image_and_mask(
            label=target_label,
            positive_only=True,
            hide_rest=False,
            num_features=100,
            min_weight=dynamic_max_weight  # Use dynamic max_weight here
        )
        norm_weights = (segment_weights - np.min(segment_weights)) / (np.max(segment_weights) - np.min(segment_weights))
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        
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
        
        plt.show()
        
    else:
        print(f"Label {target_label} not found in top labels.")
    # print(segment_info)
    predictions = predict_image(model, image, class_names=labels, target_size=(180, 180))
    print(predictions)
    return segment_info, save_path

def compute_and_generate_plots(model, model_name, disease_name, train_time, history, test_generator, y_test, label_map):
    """
    Computes predictions, metrics, and generates/saves all plots and report.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
    from sklearn.preprocessing import label_binarize

    base_dir = f'/content/drive/MyDrive/plots/{model_name}/{disease_name}'
    os.makedirs(base_dir, exist_ok=True)

    print("\033[92m[INFO::]\033[0m\n🔍 Evaluating model...")
    y_pred_probs = model.predict(test_generator)
    y_true = y_test
    target_names = list(label_map.keys())
    num_classes = len(target_names)

    if num_classes == 2:
        if y_pred_probs.ndim == 1 or y_pred_probs.shape[1] == 1:
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()
            y_pred_probs = np.column_stack([1 - y_pred_probs, y_pred_probs])
        else:
            y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)

    if y_pred_probs.shape[1] != num_classes:
        raise ValueError(f"Model output shape {y_pred_probs.shape[1]} does not match classes {num_classes}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    sensitivity = recall

    cm = confusion_matrix(y_true, y_pred)
    specificities, fnrs, npvs = [], [], []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        fnr = FN / (TP + FN) if (TP + FN) != 0 else 0
        npv = TN / (TN + FN) if (TN + FN) != 0 else 0
        specificities.append(specificity)
        fnrs.append(fnr)
        npvs.append(npv)
    specificity = np.mean(specificities)
    fnr = np.mean(fnrs)
    npv = np.mean(npvs)

    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    auc_scores = []
    fpr_dict = {}
    tpr_dict = {}
    if num_classes == 2:
        print(y_true_bin.shape)
        for i in range(2):
            one_hot = np.eye(2)[y_true]
            fpr, tpr, _ = roc_curve(one_hot[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
        mean_auc = np.mean(auc_scores)
    else:
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
        mean_auc = np.mean(auc_scores)

    # 1. Training History
    try:
      plt.figure(figsize=(14, 5))
      plt.subplot(1, 2, 1)
      plt.plot(history['accuracy'], color='#4e79a7', label='Train Accuracy')
      plt.plot(history['val_accuracy'], color='#f28e2b', label='Validation Accuracy')
      plt.title('Accuracy History', fontsize=14)
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(history['loss'], color='#e15759', label='Train Loss')
      plt.plot(history['val_loss'], color='#76b7b2', label='Validation Loss')
      plt.title('Loss History', fontsize=14)
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.tight_layout()
      plt.savefig(os.path.join(base_dir, 'training_history.png'))
      plt.show()
    except:
      print("\033[92m[INFO::]\033[0m\n\n[INFO]:: Can't plot the accuracy and loss history")
    # 2. Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))
    plt.show()

    # 3. ROC Curves
    plt.figure(figsize=(8, 6))
    for i in range(len(fpr_dict)):
        plt.plot(fpr_dict[i], tpr_dict[i], label=f'{target_names[i]} (AUC={auc_scores[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title(f'ROC Curves ({num_classes}-Class)', fontsize=16)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'roc_curves.png'))
    plt.show()

    # 4. Metrics Summary
    metrics_list = [
        ('Accuracy', accuracy),
        ('Precision', precision),
        ('Recall/Sensitivity', recall),
        ('F1-Score', f1),
        ('Specificity', specificity),
        ('False Negative Rate', fnr),
        ('Negative Predictive Value', npv),
        ('AUC', mean_auc),
        ('Training Time', train_time)
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')
    table_data = [[m[0], f'{m[1]:.4f}'] for m in metrics_list]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                     cellLoc='center', loc='center',
                     colColours=['#4e79a7', '#4e79a7'],
                     colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title('Performance Metrics Summary', fontsize=16)
    plt.savefig(os.path.join(base_dir, 'metrics_summary.png'))
    plt.show()

    # 5. Classification Report
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\033[92m[INFO::]\033[0m\n📋 Detailed Classification Report:")
    print(report)
    with open(os.path.join(base_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'fnr': fnr,
        'npv': npv,
        'auc': mean_auc,
        'training_time': train_time,
        'status': 'computed_new'
    }


def predict_image(model, input_data, labels, class_names=None, target_size=(180, 180)):
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

def ensure_training_history_plot(model_name, disease_name):
    """
    Ensures the training history plot exists. If not, loads history from JSON and creates the plot.
    """
    base_dir = f'/content/drive/MyDrive/plots/{model_name}/{disease_name}'
    os.makedirs(base_dir, exist_ok=True)
    history_plot_path = os.path.join(base_dir, 'training_history.png')
    history_json_path = f'/content/drive/MyDrive/weights/{model_name}/{disease_name}/history.json'

    # Check if plot exists
    if os.path.exists(history_plot_path):
        print(f"\033[92m[INFO::]\033[0m✅ Training history plot already exists: {history_plot_path}")
        return

    # If not, try to load history and plot
    if os.path.exists(history_json_path):
        print(f"\033[92m[INFO::]\033[0m📂 Loading training history from {history_json_path}")
        with open(history_json_path, 'r') as f:
            history = json.load(f)
        # Plot
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], color='#4e79a7', label='Train Accuracy')
        plt.plot(history['val_accuracy'], color='#f28e2b', label='Validation Accuracy')
        plt.title('Accuracy History', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], color='#e15759', label='Train Loss')
        plt.plot(history['val_loss'], color='#76b7b2', label='Validation Loss')
        plt.title('Loss History', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(history_plot_path)
        plt.show()
        print(f"\033[92m[INFO::]\033[0m✅ Training history plot generated and saved: {history_plot_path}")
    else:
        print(f"\033[92m[INFO::]\033[0m❌ Neither plot nor history JSON found for {model_name}/{disease_name}.")

def fetch_existing_plot_filepaths(model_name, disease_name):
    """
    Returns a dictionary of existing plot/report file paths for the given model and disease.
    Only includes files that actually exist.
    """
    base_dir = f'../Datasets/plots/{model_name}/{disease_name}'
    required_files = [
        'training_history.png',
        'confusion_matrix.png',
        'roc_curves.png',
        'metrics_summary.png',
        'classification_report.txt'
    ]
    existing_files = {}
    for file in required_files:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            existing_files[file] = file_path
    return existing_files

def evaluate_and_visualize(model, model_name, disease_name, train_time=None, history=None, test_generator=None, y_test=None, label_map=None, fetch_existing=False):
    """
    Evaluates model and generates comprehensive visualizations and metrics.
    Skips computation if plots exist and fetch_existing=True.
    """
    ensure_training_history_plot(model_name,disease_name)
    if(fetch_existing):
      return fetch_existing_plot_filepaths(model_name,disease_name)
    else:
      return compute_and_generate_plots(model, model_name, disease_name, train_time, history, test_generator, y_test, label_map)


@app.route('/predict_dl', methods=['POST'])
def predict_dl():
    if request.method == 'POST':
        file = request.files.get('image_file')
        disease_name = request.form.get('disease_name')  # Get selected model name from form
        model_name = request.form.get('model_name')  # Get selected model name from form
        
        if file and model_name:
            print("File uploaded:", file.filename)  # Debugging line
            
            # Preprocess the uploaded image
            image_array = preprocess_image(file)
            
            # Load the selected model
            model = load_model_from_weights(model_name)
            print(image_array.shape)
            # Make prediction using the loaded model
            input_layer = keras.Input(shape=image_array.shape[1:])
            output = model(input_layer)
            model = keras.Model(inputs=input_layer, outputs=output)

            prediction_result = model.predict(image_array)
            print(prediction_result)
            # Generate LIME explanation for the prediction
            segment_info,path = generate_lime_heatmap_and_explanation(model, image_array[0],num_segments_to_select=10)
            # Create a Plotly figure with segments and hover information
            # fig = go.Figure()

            # for segment in segment_info:
            #     for contour in segment['contours']:
            #         fig.add_trace(go.Scatter(
            #             x=contour[:, 1],  # X-coordinates
            #             y=contour[:, 0],  # Y-coordinates (inverted)
            #             mode='lines',
            #             line=dict(color=segment['color'], width=2),
            #             hoverinfo='text',
            #             text=f'Segment ID: {segment["id"]}<br>Weight: {segment["weight"]:.4f}',
            #             showlegend=False
            #         ))

            # fig.update_layout(title='LIME Segmentation Visualization', xaxis_title='X', yaxis_title='Y')
            fig = generate_clinical_visualization(segment_info, image_array)
            graph_json = fig
            images= fetch_existing_plot_filepaths(model_name,disease_name)
            # Render results in a new template (you'll need to create this template)
            return render_template('dl_model/result.html', heatmap_path ='/shap_plots/dl_res.png',graph_json=graph_json, model_plots=images)

    return render_template('deep_learning.html')  # Redirect back to deep learning page if something goes wrong
