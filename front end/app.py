from flask import Flask, render_template, request, redirect, url_for,flash,jsonify,send_from_directory
from disease_scraper import scrape_disease_info
from custom_model_manager import *
from hospital_loader import *
import os
import pandas as pd 
import numpy as np 
import xgboost as xgb
import pickle
import json
import matplotlib
import keras
import os
import requests
import time
from test11 import *
app = Flask(__name__)
import plotly.io as pio
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import zipfile
from sklearn.metrics import (accuracy_score, auc, classification_report,
                            confusion_matrix, f1_score, precision_score,
                            recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from skimage.metrics import normalized_root_mse, structural_similarity
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import (Concatenate, Conv2D, Dense, Dropout,
                                     Flatten, GlobalAveragePooling2D, Input,
                                     MaxPooling2D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Resizing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# TensorFlow Pipeline models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import MobileNetV3Large
pio.renderers.default = 'browser'

# Configurations
CONFIG_FILE_PATH = 'front end/config/config.json'
MODELS_DIR = 'Back end/chk_pts/'
SHAP_VALUES_DIR = 'Back end/shap_values/'
SHAP_PLOT_DIR = 'front end/static/shap_plots/'
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'Back end\ml_data'


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
matplotlib.use('Agg')  # Set backend for non-interactive plotting

@app.route('/')
def index():
    return render_template('index/index.html')

@app.route('/news')
def news():
    # Assuming you fetch data from an API
    news_articles = [{"title": "Health News 1", "description": "Details about news 1", "url": "#"}]
    return render_template('news.html', news_articles=news_articles)

@app.route('/hospitals', methods=['GET', 'POST'])
def hospitals():
    hospitals_data = []
    
    if request.method == 'POST':
        location = request.form['location']
        lat_lon = get_latitude_longitude(location)
        
        if lat_lon:
            latitude, longitude = lat_lon
            hospitals_data = fetch_nearby_hospitals(latitude, longitude)
    
    return render_template('location/hospitals.html', hospitals=hospitals_data)

@app.route('/search')
def search():
    query = request.args.get('query')
    # Web scraping for disease info using your scraping function
    disease_info = scrape_disease_info(query)
    return render_template('search_results.html', query=query, info=disease_info)

@app.route('/models')
def models():
    return render_template('index/predict.html')

# Define available models and their parameters

def preprocess_data(data,target_column):
    data.dropna()
    # Encode categorical features
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes
    # Split dataset into features (X) and target (y)
    y = data[target_column]


    # Get unique classes and their counts
    class_counts = y.value_counts()

    # Identify minimum and maximum class counts
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()

    # Balance classes by reducing larger classes if necessary
    balanced_data = []

    for cls in class_counts.index:
        cls_data = data[data[target_column] == cls]

        if len(cls_data) > min_class_count * 2:
            # Downsample larger classes to double the size of the minimum class
            cls_data = resample(cls_data, replace=False, n_samples=min_class_count * 2, random_state=42)

        balanced_data.append(cls_data)
        print(len(balanced_data))

    # Combine balanced classes into a single DataFrame
    balanced_data = pd.concat(balanced_data)
    return balanced_data

def load_model_config():
    config_path = os.path.join('front end', 'config', 'config.json')
    with open(config_path, 'r') as file:
        models_config = json.load(file)
    print("Loaded models config:", models_config)  # Add this line for debugging
    return models_config

@app.route('/basic_models', methods=['GET', 'POST'])
def basic_models_route():
    basic_models = load_model_config()  # Reload each time to ensure updated configuration

    if request.method == 'POST':
        selected_model = request.form.get('selected_model')
        
        if selected_model in basic_models:
            inputs = {param: request.form.get(param) for param in basic_models[selected_model]["features"]}
            # Redirect to prediction page with model and inputs
            return redirect(url_for('predict', model=selected_model, **inputs))
        else:
            flash("Selected model not found in configuration.", "danger")

    return render_template('basic_model/basic_models.html', models=basic_models)


@app.route('/add_custom_model', methods=['GET', 'POST'])
def add_custom_model():
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        target_column = request.form.get('target_column')
        file = request.files['file']

        if model_name and target_column and file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            print(UPLOAD_FOLDER+ file.filename)
            try :
                file.save(file_path)
                data = pd.read_csv(file_path)
                balanced_data= preprocess_data(data, target_column)
                model_path = train_and_save_model(balanced_data, target_column, model_name)
                flash(f"Model '{model_name}' created and saved at {model_path}", 'success')
                return redirect(url_for('model_visualizations', model_name=model_name, file_path= file_path))
            except Exception as e:
                error_message = str(e) or "An unexpected error occurred."
                return render_template('basic_model/add_custom_model.html', 
                             error_message=error_message)
        else:
            flash("Please provide all required fields.", 'warning')
            return render_template('basic_model/add_custom_model.html')
    elif request.method == 'GET':
        return render_template('basic_model/add_custom_model.html')

@app.route('/model_visualizations/<model_name>/<file_path>')
def model_visualizations(model_name, file_path):
    config_data = load_model_config()
    if model_name not in config_data:
        flash("Model not found.", "danger")
        return redirect(url_for('basic_models_route'))

    # Load the data and generate visualizations
    data = pd.read_csv(file_path)
    target_column = config_data[model_name]['target_column']
    data = preprocess_data(data, target_column)
    
    # Generate visualizations and insights
    fig_cm, report_summary = create_visualizations(model_name, data, target_column)
    
    shap_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_shap_summary.png")
    corr_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_correlation_matrix.png")
    pair_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_pair_plot.png")
    
    # Convert figures to JSON for Plotly to render in HTML
    fig_cm_json = fig_cm.to_json()

    return render_template('basic_model/model_visualizations.html', 
                           fig_cm=fig_cm_json,
                           shap_plot_path=shap_plot_path,
                           corr=corr_plot_path,
                           pair_plot_path=pair_plot_path,
                           report_summary=report_summary)  # Pass report summary to template

# Dictionary to store loaded models
models = {}

# Function to load available models from a specified directory
def load_model(models_dir, model_name):
    model_path = os.path.join(models_dir, model_name+".json")
    models[model_name] = xgb.XGBClassifier()
    models[model_name].load_model(model_path)


@app.route('/predict', methods=['GET'])
def predict():
    # Get model choice from form
    model_choice = request.args.get('model')  # Changed to use query parameters
    
    # Check if the selected model is available
    if model_choice in load_model_config().keys():
        # Collect inputs from query parameters
        inputs = {key: request.args.get(key) for key in request.args if key != 'model'}

        prediction_result, force_plot_path = run_model(model_choice, inputs)
        load_model('Back end/chk_pts', model_choice)  # Ensure the model is loaded
        shap_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_choice}_shap_summary.png")
        corr_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_choice}_correlation_matrix.png")
        pair_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_choice}_pair_plot.png")
    
        return render_template('basic_model/predict_results.html', 
                    result=prediction_result, 
                    model_name=model_choice, 
                    force_plot=force_plot_path,
                           shap_plot_path=shap_plot_path,
                           corr=corr_plot_path,
                           pair_plot_path=pair_plot_path)
    else:
        return "Model not found", 400

























# Load available models from config file
def load_model_config2(config_path='front end\config\config2.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['models']
    return {}

# Save model configuration to the config file
def save_model_config2(models, config_path='front end\config\config2.json'):
    with open(config_path, 'w') as f:
        json.dump({"models": models}, f, indent=4)

@app.route('/add_custom_dl_model', methods=['GET', 'POST'])
def add_custom_dl_model():
    error_message = None
    success_message = None
    
    if request.method == 'POST':
        if 'model_name' not in request.form or 'weights_folder' not in request.files:
            error_message = "Model name and weights folder are required."
        else:
            model_name = request.form['model_name']
            weights_files = request.files.getlist('weights_folder')

            if not weights_files or all(f.filename == '' for f in weights_files):
                error_message = "No selected files."
            else:
                # Create a directory to save weights if it doesn't exist
                weights_dir = os.path.join('Back end/weights', model_name)
                os.makedirs(weights_dir, exist_ok=True)

                # Save the uploaded weights file(s)
                for file in weights_files:
                    file_path = os.path.join(weights_dir, file.filename)
                    file.save(file_path)

                # Load existing models
                models = load_model_config2()
                
                # Update the model's weight path
                models[model_name] = weights_dir  # Save the directory path

                # Save updated models back to config file
                save_model_config2(models)

                success_message = f"Weights for {model_name} uploaded successfully."

    # Render the form for GET requests or show messages for POST requests
    return render_template('dl_model/add_custom_model.html', error_message=error_message, success_message=success_message)

@app.route('/deep_learning', methods=['GET', 'POST'])
def deep_learning():
    if request.method == 'POST':
        file = request.files.get('image_file')
        model_name = request.form.get('model_name')  # Get selected model name from form
        
        if file and model_name:
            print("File uploaded:", file.filename)  # Debugging line
            
            # Run deep learning prediction with the selected model
            # prediction_result = run_deep_learning_model(file, model_name)
            # return render_template('dl_results.html', prediction=prediction_result)

    # Load available models for selection in the HTML form
    models = load_model_config2()
    
    return render_template('dl_model/deep_learning.html', models=models, diseases={
    "Lung-Adenocarcinoma": ["resnet50", "inceptionv3"],
    "Brain-Tumor": ["resnet50", "inceptionv3"],
    "malaria": ["cnn", "densenet"]
}
)

def create_medical_lime_plot(segment_info, image_shape, medical_image, top_k=15, min_weight_threshold=0.01):
    """
    Enhanced version for medical imaging visualization with proper RGB handling
    """
    # Extract height and width from image shape
    height, width = image_shape[:2]
    
    # Filter and sort segments
    important_segments = sorted(
        [seg for seg in segment_info if seg['weight'] >= min_weight_threshold],
        key=lambda x: x['weight'], reverse=True
    )[:top_k]

    # Create figure with dark background for better contrast
    fig = go.Figure()
    
    # Medical imaging color palette (high-contrast colors)
    colors = px.colors.qualitative.Vivid + px.colors.qualitative.Dark24
    
    # Add each segment with enhanced visibility
    for idx, seg in enumerate(important_segments):
        contour_color = colors[idx % len(colors)]
        
        # Filter and validate contours
        valid_contours = [
            cnt for cnt in seg['contours']
            if len(cnt) > 10 and cv2.contourArea(cnt) > 50
        ]
        
        for cnt in valid_contours:
            # Convert contour points to plotly coordinates
            y_values = height - cnt[:, 0]  # Proper Y-axis flip for medical view
            
            fig.add_trace(go.Scatter(
                x=cnt[:, 1],
                y=y_values,
                mode='lines',
                line=dict(color=contour_color, width=4),  # Thicker lines
                fill='toself',
                fillcolor=contour_color.replace('rgb', 'rgba').replace(')', ',0.5)'),  # More opaque fill
                name=f'Region {idx+1}',
                hovertemplate=(
                    f'<b>Region {idx+1}</b><br>'
                    f'Clinical Significance: {seg["weight"]:.4f}<br>'
                    f'Size: {cv2.contourArea(cnt):.0f} px<extra></extra>'
                )
            ))
    
    # Add original RGB image as background
    fig.add_layout_image(
        dict(
            source="data:image/png;base64," + image_to_base64(medical_image),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=width,
            sizey=height,
            sizing="stretch",  # Maintain aspect ratio
            opacity=0.85,  # More visible background
            layer="below"
        )
    )
    
    # Medical-grade layout configuration
    fig.update_layout(
        title={
            'text': 'AI Explanation of Clinical Findings',
            'x': 0.5,
            'font': {'size': 24, 'color': 'white', 'family': 'Arial Black'}
        },
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, width],
            scaleanchor='y',
            scaleratio=1  # Ensure 1:1 pixel aspect ratio
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, height],
            scaleanchor='x',
            scaleratio=1
        ),
        plot_bgcolor='rgba(0,0,0,0.9)',
        paper_bgcolor='rgba(0,0,0,0.9)',
        legend=dict(
            title='Pathological Regions',
            font=dict(color='white', size=14),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=20, r=20, t=100, b=20),
        width=800,
        height=800*(height/width)  # Maintain original aspect ratio
    )
    
    return fig

# Helper functions
def hex_to_rgba(hex, alpha):
    """Convert hex color to rgba string"""
    h = hex.lstrip('#')
    return f'rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})'

def image_to_base64(img_array):
    """Convert medical image array to base64 string"""
    # Remove singleton dimensions and ensure proper data type
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    
    # Handle different array shapes:
    if img_array.ndim == 4:  # Batch dimension (N, H, W, C)
        img_array = img_array[0]  # Take first image in batch
    elif img_array.ndim == 3 and img_array.shape[0] == 1:  # (1, H, W)
        img_array = img_array[0]
    
    # Add channel dimension if missing (for grayscale)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_array)
    
    # Save to base64
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Usage
def generate_clinical_visualization(segment_info, medical_image):
    """Main entry point for medical visualization"""
    fig = create_medical_lime_plot(
        segment_info=segment_info,
        image_shape=medical_image.shape,
        medical_image=medical_image,
        top_k=10,
        min_weight_threshold=0.05
    )
    return fig.to_json()

# @app.route('/predict_dl', methods=['POST'])
# def predict_dl():
#     if request.method == 'POST':
#         file = request.files.get('image_file')
#         model_name = request.form.get('model_name')  # Get selected model name from form
        
#         if file and model_name:
#             print("File uploaded:", file.filename)  # Debugging line
            
#             # Preprocess the uploaded image
#             image_array = preprocess_image(file)
            
#             # Load the selected model
#             model = load_model_from_weights(model_name)
#             print(image_array.shape)
#             # Make prediction using the loaded model
#             input_layer = keras.Input(shape=image_array.shape[1:])
#             output = model(input_layer)
#             model = keras.Model(inputs=input_layer, outputs=output)

#             prediction_result = model.predict(image_array)
#             print(prediction_result)
#             # Generate LIME explanation for the prediction
#             segment_info,path = generate_lime_heatmap_and_explanation(model, image_array[0],num_segments_to_select=10)
#             # Create a Plotly figure with segments and hover information
#             # fig = go.Figure()

#             # for segment in segment_info:
#             #     for contour in segment['contours']:
#             #         fig.add_trace(go.Scatter(
#             #             x=contour[:, 1],  # X-coordinates
#             #             y=contour[:, 0],  # Y-coordinates (inverted)
#             #             mode='lines',
#             #             line=dict(color=segment['color'], width=2),
#             #             hoverinfo='text',
#             #             text=f'Segment ID: {segment["id"]}<br>Weight: {segment["weight"]:.4f}',
#             #             showlegend=False
#             #         ))

#             # fig.update_layout(title='LIME Segmentation Visualization', xaxis_title='X', yaxis_title='Y')
#             fig = generate_clinical_visualization(segment_info, image_array)
#             graph_json = fig
#             # Render results in a new template (you'll need to create this template)
#             return render_template('dl_model/result.html', heatmap_path ='/shap_plots/dl_res.png',graph_json=graph_json)

#     return render_template('deep_learning.html')  # Redirect back to deep learning page if something goes wrong

def build_multiscale_cnn_model(input_shape=(180, 180, 3),target_size=(180, 180), num_classes=2):
    """
    Builds and compiles a multi-scale CNN model for image classification

    Args:
        input_shape: Tuple of (height, width, channels) for input images
        target_size: Tuple (height, width) to resize images to
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    # Build the model
    inputs = Input(shape=input_shape)
    x = Resizing(target_size[0], target_size[1], interpolation='bilinear')(inputs)
    # CNN Backbone
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Multi-Scale Head
    b1 = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    b1 = MaxPooling2D((2, 2))(b1)

    b2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    b2 = MaxPooling2D((2, 2))(b2)

    b3 = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    b3 = MaxPooling2D((2, 2))(b3)

    # Feature fusion
    concat = Concatenate()([b1, b2, b3])
    flat = Flatten()(concat)

    # Classifier head
    dense1 = Dense(512, activation='relu')(flat)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(256, activation='relu')(drop1)
    drop2 = Dropout(0.5)(dense2)
    output = Dense(num_classes, activation='softmax')(drop2)

    # Build model
    model = Model(inputs=inputs, outputs=output)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_inceptionv3_model(input_shape=(180, 180, 3), target_size=(244, 244), num_classes=2, freeze_base=True):
    """
    Builds and compiles an InceptionV3-based model for image classification with resizing layer

    Args:
        input_shape: Tuple of (height, width, channels) for input images (can be None for variable size)
        target_size: Tuple (height, width) to resize images to
        num_classes: Number of output classes
        freeze_base: Whether to freeze pre-trained layers (default: True)

    Returns:
        Compiled Keras model
    """
    # Input layer with variable size support
    inputs = Input(shape=input_shape)

    # Resizing layer to standardize input size
    x = Resizing(target_size[0], target_size[1], interpolation='bilinear')(inputs)

    # Load pre-trained InceptionV3 base model
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(target_size[0], target_size[1], 3),
        pooling=None
    )

    # Freeze base model layers if requested
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    # Pass resized input through base model
    x = base_model(x)

    # Add custom head
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    # Build full model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_resnet_model_with_resize(input_shape=(180, 180, 3), target_size=(224, 224), num_classes=2, freeze_base=True):
    """
    Builds and compiles a ResNet50-based model with an initial resizing layer.

    Args:
        input_shape: Tuple of (height, width, channels) for input images (can be None for variable size)
        target_size: Tuple (height, width) to resize images to
        num_classes: Number of output classes
        freeze_base: Whether to freeze pre-trained layers (default: True)
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = Resizing(target_size[0], target_size[1], interpolation='bilinear')(inputs)

    # Load pre-trained ResNet50 base model
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(target_size[0], target_size[1], 3),
        pooling=None
    )
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_model(model_name, input_shape=(180, 180, 3), num_classes=2):
    """
    Initializes and returns a model based on the given name.
    Currently supports only 'multi-header-cnn'.

    Args:
        model_name (str): Name of the model to initialize
        input_shape (tuple): Shape of input images
        num_classes (int): Number of output classes

    Returns:
        tuple: (model, plot_file_path)
        - model: Compiled Keras model
        - plot_file_path: Path to saved model architecture plot (or None if failed)
    """
    if model_name == 'multi-header-cnn':
      model = build_multiscale_cnn_model(num_classes=num_classes)
    elif model_name == 'resnet50':
      model = build_resnet_model_with_resize(num_classes=num_classes)
    elif model_name == 'inceptionv3':
      model = build_inceptionv3_model(num_classes=num_classes)
    elif model_name == 'mobilenet':
      model = build_multiscale_cnn_model()
    else:
        raise ValueError(f"Model '{model_name}' is not implemented. Only 'multi-header-cnn' is available.")

    # Attempt to generate model plot with simplified settings
    plot_file_path = f"{model_name}_architecture.png"
    try:
        # Generate plot with improved settings

        # Post-process image for better readability
        try:
            img = Image.open(plot_file_path)
            width, height = img.size
            new_height = int(height * 1.5)  # Add whitespace
            new_img = Image.new('RGB', (width, new_height), 'white')
            new_img.paste(img, (0, 0))
            new_img.save(plot_file_path)
        except Exception as img_error:
            print(f"\033[92m[INFO::]\033[0mImage post-processing failed: {img_error}")
    except Exception as e:
        print(f"\033[92m[INFO::]\033[0mFailed to generate model plot: {str(e)}")
        print("\033[92m[INFO::]\033[0mUsing text summary instead")
        model.summary()
    # finally:
        # Create a simple manual visualization
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5,
                 "Model Architecture Visualization\n"
                 "--------------------------------\n"
                 "Input -> Conv Blocks -> Multi-Scale Heads -> Concatenate -> Dense Layers -> Output\n"
                 f"Input Shape: {input_shape}\n"
                 f"Output Classes: {num_classes}",
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.title(f"{model_name} Architecture", fontsize=16)
        manual_plot_path = f"{model_name}_architecture_fallback.png"
        plt.savefig(manual_plot_path)
        plt.close()
        plot_file_path = manual_plot_path
        print(f"\033[92m[INFO::]\033[0mCreated simplified architecture diagram at: {plot_file_path}")

    return model, plot_file_path

# Load available models from config file
def load_model_config2(config_path='front end/config/config2.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('models', {})
    return {}

def train_and_save_model(model, model_name, disease_name, epochs=30, fetch_existing=True, train_generator= None, val_generator = None):
    """
    Trains and saves a model with history and training time, or loads existing model

    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        model_name: Name for saving model
        disease_name: Name of disease for file organization
        epochs: Maximum training epochs
        fetch_existing: Whether to load existing model if available

    Returns:
        tuple: (trained_model, training_history, training_time)
    """
    # Create save paths
    base_dir = f"C:/Users/Asus/Desktop/Git Codes/Explainable_ai_Dashboard/Datasets/weights/{model_name}/{disease_name}"
    os.makedirs(base_dir, exist_ok=True)

    model_path = f"{base_dir}/{disease_name}.h5"
    history_path = f"{base_dir}/history.json"
    time_path = f"{base_dir}/training_time.txt"

    # Check for existing model
    # if fetch_existing and os.path.exists(model_path):
    if fetch_existing:
        print(f"\033[92m[INFO::]\033[0m🚀 Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Try to load history
        history = {}
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = {}

        # Try to load training time
        training_time = 0
        if os.path.exists(time_path):
            with open(time_path, 'r') as f:
                training_time = float(f.read())

        return model, history, training_time

    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\033[92m[INFO::]\033[0mUsing {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("\033[92m[INFO::]\033[0mUsing CPU")

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Train model
    print(f"\033[92m[INFO::]\033[0m🔥 Training new model for {disease_name} using {model_name}")
    start_time = time.time()
    history_obj = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"\033[92m[INFO::]\033[0m\n✅ Training completed in {training_time:.2f} seconds")

    # Convert history to serializable format
    history = {k: [float(num) for num in v] for k, v in history_obj.history.items()}

    # Save model and metadata
    model.save(model_path)
    with open(history_path, 'w') as f:
        json.dump(history, f)
    with open(time_path, 'w') as f:
        f.write(str(training_time))

    print(f"\033[92m[INFO::]\033[0m💾 Model and training data saved to {base_dir}")

    return model, history, training_time

def generate_vibrant_colors(num_colors):
    """Generate a list of vibrant colors."""
    colors = plt.cm.get_cmap("hsv", num_colors)  # Use HSV colormap for vibrant colors
    return [colors(i)[:3] for i in range(num_colors)]  # Return RGB values

from keras.layers import TFSMLayer

# Load model from weights directory
def load_model_from_weights(model_name, disease_name, config_path='front end/config/config2.json'):
    models = load_model_config2(config_path)
    print(list(os.listdir("./Datasets")))
    print(list(os.listdir("../")))
    
    if model_name in models and disease_name in models[model_name]:
        weights_path = models[model_name][disease_name]["weights"]
        model, plot_path = get_model(model_name,
                                 input_shape=(180, 180, 3),
                                 num_classes=models[model_name][disease_name]["num_class"])
        model,history,training_time = train_and_save_model(
            model=model,
            model_name=model_name,
            disease_name=disease_name,
            epochs=1,
            fetch_existing=True
        )
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
def generate_lime_heatmap_and_explanation(model, model_name , disease_name, image, target_label=1, num_segments_to_select=8, save_path='front end/static/shap_plots/dl_res.png'):
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
    predictions = predict_image(model, image, class_names=load_model_config2()[model_name][disease_name]["class_name"], target_size=(180, 180))
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
    base_dir = f'Datasets/plots/{model_name}/{disease_name}'
    base_dir2 = f'plots/{model_name}/{disease_name}'
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
            file_path2 = os.path.join(base_dir2, file)
            existing_files[file.split('.')[0]]=file_path2
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
            model = load_model_from_weights(model_name,disease_name)
            print(image_array.shape)
            # Make prediction using the loaded model
            input_layer = keras.Input(shape=image_array.shape[1:])
            output = model(input_layer)
            model = keras.Model(inputs=input_layer, outputs=output)

            prediction_result = model.predict(image_array)
            print(prediction_result)
            # Generate LIME explanation for the prediction
            segment_info,path = generate_lime_heatmap_and_explanation(model=model, image=image_array[0],num_segments_to_select=10, model_name=model_name, disease_name=disease_name)
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
            # fig = generate_clinical_visualization(segment_info, image_array)
            # graph_json = fig
            images= fetch_existing_plot_filepaths(model_name,disease_name)
            print(images)
            # Render results in a new template (you'll need to create this template)
            return render_template('dl_model/result.html', heatmap_path ='/shap_plots/dl_res.png',graph_json={}, model_plots=images)

    return render_template('deep_learning.html')  # Redirect back to deep learning page if something goes wrong

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    print(filename)
    plot_dir = r'C:\Users\Asus\Desktop\Git Codes\Explainable_ai_Dashboard\Datasets'
    print(plot_dir)
    return send_from_directory(plot_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)