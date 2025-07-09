import os
import json
import shap
import xgboost as xgb
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.metrics import f1_score, mean_squared_error
# Configurations
CONFIG_FILE_PATH = 'front end/config/config.json'
MODELS_DIR = 'Back end/chk_pts/'
SHAP_VALUES_DIR = 'back end/shap_values/'
SHAP_EXPLAINER_DIR = 'back end/shap_explainer/'
SHAP_PLOT_DIR = 'front end/static/shap_plots/'

MAX_SHAP_SAMPLES = 200  # Set maximum number of samples for SHAP

def update_config_file(model_name, target_column, feature_columns):
    config_data = {}
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as file:
            config_data = json.load(file)
    config_data[model_name] = {'target_column': target_column, 'features': feature_columns}
    with open(CONFIG_FILE_PATH, 'w') as file:
        json.dump(config_data, file, indent=4)

def train_and_save_model(data, target_column, model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.json")
    shap_values_path = os.path.join(SHAP_VALUES_DIR, f"{model_name}_shap_values.npz")
    shap_explainer_path = os.path.join(SHAP_EXPLAINER_DIR, f"{model_name}_explainer.pkl")  # Path for saving the explainer
    
    # Check if model and SHAP values already exist
    if os.path.exists(model_path) and os.path.exists(shap_values_path):
        print(f"Model and SHAP values for '{model_name}' already exist. Skipping training.")
        return model_path
    print(data)
    # Separate features and target from balanced data
    x_balanced = data.drop(columns=[target_column])
    y_balanced = data[target_column]

    # Apply SMOTE to create a balanced dataset with equal number of instances for both classes
    smote = SMOTE(n_jobs=-1)
    try:
        # Fit SMOTE only if there is an imbalance after downsampling
        x_resampled, y_resampled = smote.fit_resample(x_balanced, y_balanced)
    except:
        print("Smote failed")
        x_resampled= x_balanced
        y_resampled= y_balanced
    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False)
    model.fit(x_resampled, y_resampled)

    # Save model
    model.save_model(model_path)

    # Generate and save SHAP values with sampling
    shap_values, x_sampled = generate_shap_values(model, x_resampled)
    
    np.savez_compressed(shap_values_path, shap_values=shap_values.values, base_values=shap_values.base_values, features=x_sampled, feature_names=x_balanced.columns.tolist())

    # Save the explainer (you can save it as a pickle)
    with open(shap_explainer_path, 'wb') as f:
        pickle.dump(shap.Explainer(model), f)

    # Update config file
    update_config_file(model_name, target_column, list(x_balanced.columns))
    
    return model_path


def generate_shap_values(model, x):
    # Limit dataset to 3,000 samples for SHAP analysis
    if len(x) > MAX_SHAP_SAMPLES:
        x_sampled = x.sample(n=MAX_SHAP_SAMPLES, random_state=42)
    else:
        x_sampled = x

    # Run SHAP analysis on the sampled data
    explainer = shap.Explainer(model.predict, x_sampled)
    shap_values = explainer(x_sampled)
    return shap_values, x_sampled

def create_visualizations(model_name, data, target_column):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.json")
    shap_values_path = os.path.join(SHAP_VALUES_DIR, f"{model_name}_shap_values.npz")
    
    # Encode categorical features
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes

    # Separate features and target from balanced data
    x_balanced = data.drop(columns=[target_column])
    y_balanced = data[target_column]

    # Load the saved model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Check if SHAP values already exist
    if not os.path.exists(shap_values_path):
        print(f"SHAP values for '{model_name}' do not exist. Please run training first.")
        return None
    else:
        # Load SHAP values and the feature sample
        shap_data = np.load(shap_values_path, allow_pickle=True)
        
        shap_values = shap_data['shap_values']
        base_values = shap_data['base_values']
        x_sampled = shap_data['features']

        # Generate SHAP summary plot with the sampled features 
        plt.figure()
        shap.violin_plot(shap_values, features=pd.DataFrame(x_sampled), show=False,  feature_names=x_balanced.columns.tolist())
        plt.savefig(os.path.join(SHAP_PLOT_DIR, f"{model_name}_shap_summary.png"), bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    # Generate confusion matrix
    y_pred = model.predict(x_balanced)
    conf_matrix = confusion_matrix(y_balanced, y_pred)
    fig_cm = px.imshow(conf_matrix,
                        labels={'x': 'Predicted', 'y': 'Actual'},
                        color_continuous_scale='Blues')

    # Calculate accuracy score
    accuracy_score = (y_balanced == y_pred).mean()
    
    # Check if pair plot already exists; if not, create it.
    pair_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_pair_plot.png")
    if not os.path.exists(pair_plot_path):
        print("Generating pair plot...")
        plt.figure(figsize=(10, 8))
        sns.pairplot(data.sample(n=min(1000,len(data))), hue=target_column)  # Sample to reduce time
        plt.savefig(pair_plot_path, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        print(f"Pair plot for '{model_name}' already exists. Skipping generation.")

    # Calculate and plot the correlation matrix using Seaborn
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    # Get the absolute values of correlations with respect to the target column
    target_correlations = correlation_matrix[target_column].abs()

    # Get the top 3 features correlated with the target column
    top_features = target_correlations.nlargest(4).index[1:]  # Exclude the target itself
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, xticklabels=data.columns, yticklabels=data.columns)
    
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(SHAP_PLOT_DIR, f"{model_name}_correlation_matrix.png"), bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    # Prepare statements regarding feature importance and dataset distribution
# Calculate accuracy score and other metrics
    accuracy_score = (y_balanced == y_pred).mean()
    f1_score_value = f1_score(y_balanced, y_pred)
    rmse_score = np.sqrt(mean_squared_error(y_balanced, y_pred))
    
    # Prepare human-readable statements
    insights_statements = ( f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; padding: 20px; border-radius: 8px;">
        <h1 style="color: #2c3e50;">Model Performance Insights</h1>
        
        <div style="margin-bottom: 20px;">
            <p><strong>Model Accuracy:</strong> <span style="font-weight:bold;">{accuracy_score:.2f}</span></p>
            <p><strong>F1 Score:</strong> <span style="font-weight:bold;">{f1_score_value:.2f}</span></p>
            <p><strong>RMSE Score:</strong> <span style="font-weight:bold;">{rmse_score:.2f}</span></p>
            <p>The dataset was preprocessed using SMOTE to balance it based on the lesser number of instances for binary classification.</p>
        </div>

        <div style="margin-top: 20px; padding: 10px; border-left: 5px solid #3498db; background-color: #ecf9ff;">
            <h2>Feature Correlation Analysis</h2>
            <p>The correlation matrix indicates that the top three highly correlated features are:</p>
            <ul>
                <li>{top_features[0]}</li>
                <li>{top_features[1]}</li>
                <li>{top_features[2]}</li>
            </ul>
        </div>

        <div style="margin-top: 20px; padding: 10px; border-left: 5px solid #3498db; background-color: #ecf9ff;">
            <h2>SHAP Analysis</h2>
            <p>SHAP analysis confirms that these features are significantly influencing the model's predictions:</p>
            <ul>
                <li>{x_balanced.columns[np.argsort(np.abs(shap_values).mean(axis=0))[-3:]][-1]}</li>
                <li>{x_balanced.columns[np.argsort(np.abs(shap_values).mean(axis=0))[-3:]][-2]}</li>
                <li>{x_balanced.columns[np.argsort(np.abs(shap_values).mean(axis=0))[-3:]][-3]}</li>
            </ul>
        </div>

        <div style="margin-top: 20px;">
            <h2>Further Reading</h2>
            <p>For more information on SHAP and LIME methodologies for interpreting models, you can visit:</p>
            <ul>
                <li><a href="https://shap.readthedocs.io/en/latest/" style="color: #3498db;">SHAP Documentation</a></li>
                <li><a href="https://github.com/marcotcr/lime" style="color: #3498db;">LIME Documentation</a></li>
            </ul>
        </div>
    </div>
    """
    )

    return fig_cm, insights_statements  # Returning Plotly figures and summary report

def run_model(selected_model, inputs):
    model_path = os.path.join(MODELS_DIR, f"{selected_model}.json")
    shap_explainer_path = os.path.join(SHAP_EXPLAINER_DIR, f"{selected_model}_explainer.pkl")  # Path for SHAP explainer

    # Load the model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Prepare input data for prediction
    input_array = np.array([list(inputs.values())], dtype=float)  # Reshape inputs for prediction
    print(input_array)
    # Get predictions
    predictions = model.predict(input_array)

    # Generate SHAP force plot and save it as an image file
    force_plot_path = os.path.join(SHAP_PLOT_DIR, f"{selected_model}_force_plot.png")
    
    # Load existing SHAP explainer
    with open(shap_explainer_path, 'rb') as f:
        explainer = pickle.load(f)

    # Calculate SHAP values for the input using loaded explainer
    input_shap_values = explainer(input_array)

    # Create a force plot using matplotlib
    shap.force_plot(explainer.expected_value, input_shap_values.values, input_array, matplotlib=True, feature_names=inputs.keys() )
    
    # Save the figure
    plt.savefig(force_plot_path)
    plt.close()  # Close the figure to free memory
    return predictions[0], force_plot_path  # Return prediction and force plot HTML