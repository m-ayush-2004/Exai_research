from flask import Flask, render_template, request, redirect, url_for,flash,jsonify
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
from test11 import *
app = Flask(__name__)
import plotly.io as pio
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























import plotly.graph_objects as go


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
    
    return render_template('dl_model/deep_learning.html', models=models)

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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

from PIL import Image
from io import BytesIO
import base64
import cv2

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


@app.route('/predict_dl', methods=['POST'])
def predict_dl():
    if request.method == 'POST':
        file = request.files.get('image_file')
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
            # Render results in a new template (you'll need to create this template)
            return render_template('dl_model/result.html', heatmap_path ='/shap_plots/dl_res.png',graph_json=graph_json)

    return render_template('deep_learning.html')  # Redirect back to deep learning page if something goes wrong



if __name__ == '__main__':
    app.run()