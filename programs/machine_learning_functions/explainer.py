# The `import` statements at the beginning of the Python script are used to import various libraries
# and modules that provide specific functionalities required for the data analysis and machine
# learning tasks. Here is a brief explanation of each import statement:
import shap
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
import xgboost as xgb
import seaborn as sns
import plotly.express as px
import pandas as pd

# The lines you provided are setting up environment variables for paths related to model weights, SHAP
# values, SHAP explainer, and SHAP plot directories. Here is a breakdown of each variable:
ML_MODEL_WEIGHTS_PATH = os.getenv('ML_WEIGHTS_PATH')
ML_SHAP_VALUES_DIR = os.getenv('ML_SHAP_VALUES_DIR')
ML_SHAP_EXPLAINER_DIR = os.getenv('ML_SHAP_EXPLAINER_DIR')
ML_SHAP_PLOT_DIR = os.getenv('ML_SHAP_PLOT_DIR')
MAX_SHAP_SAMPLES = 200

def generate_shap_values(model, x):
    """
    The function `generate_shap_values` takes a model and input data, limits the dataset to 3,000
    samples for SHAP analysis, runs SHAP analysis on the sampled data, and returns the SHAP values along
    with the sampled data.
    
    Args:
      model: The `model` parameter in the `generate_shap_values` function is typically a machine
    learning model that has already been trained on a dataset. This model is used to generate
    predictions for the input data `x` and then SHAP (SHapley Additive exPlanations) values are
    calculated
      x: It seems like you were about to provide some information about the parameter `x`, but the
    information is missing. Could you please provide more details about the parameter `x` so that I can
    assist you further with the `generate_shap_values` function?
    
    Returns:
      The function `generate_shap_values` returns the SHAP values calculated for the sampled data
    `x_sampled` and the sampled data itself `x_sampled`.
    """
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
    """
    The function `create_visualizations` generates various visualizations and insights for a machine
    learning model, including SHAP analysis, correlation matrix, and model performance metrics.
    
    Args:
      model_name: The `model_name` parameter is a string that represents the name or identifier of the
    model being used for creating visualizations and generating insights. It is used to construct file
    paths for model weights, SHAP values, and saving the generated plots with the model name as part of
    the file name.
      data: The `data` parameter in the `create_visualizations` function represents the dataset that
    will be used to create visualizations and analyze the model's performance. It contains the features
    and the target column necessary for training and evaluating the model. The dataset should be in a
    format that can be used by the
      target_column: The `target_column` parameter in the `create_visualizations` function refers to the
    column in your dataset that contains the target variable you are trying to predict. This column is
    typically the one that your machine learning model will be trained to predict based on the other
    features in the dataset. It is important
    
    Returns:
      The function `create_visualizations` returns two main outputs:
    1. `fig_cm`: A confusion matrix plot generated using Plotly, representing the model's predictions
    compared to the actual target values.
    2. `insights_statements`: A formatted HTML string containing various insights and analysis regarding
    the model's performance, feature correlations, SHAP analysis, and additional resources for further
    reading.
    """
    MODEL_PATH = os.path.join(ML_MODEL_WEIGHTS_PATH, f"{model_name}.json")
    SHAP_PATH = os.path.join(ML_SHAP_VALUES_DIR, f"{model_name}.npz")
    PLOT_DIR = os.path.join(ML_SHAP_PLOT_DIR, f"{model_name}/")
    os.makedirs(PLOT_DIR, exist_ok=True)
    # Encode categorical features
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes

    # Separate features and target from balanced data
    x_balanced = data.drop(columns=[target_column])
    y_balanced = data[target_column]

    # Load the saved model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Check if SHAP values already exist
    if not os.path.exists(SHAP_PATH):
        print(f"\033[31m[INFO::]\033[00m SHAP values for '{model_name}' do not exist. Please run training first.")
        return None
    else:
        # Load SHAP values and the feature sample
        shap_data = np.load(SHAP_PATH, allow_pickle=True)
        
        shap_values = shap_data['shap_values']
        x_sampled = shap_data['features']

        # Generate SHAP summary plot with the sampled features 
        plt.figure()
        shap.violin_plot(shap_values, features=pd.DataFrame(x_sampled), show=False,  feature_names=x_balanced.columns.tolist())
        plt.savefig(os.path.join(ML_SHAP_PLOT_DIR, f"{model_name}/shap_summary.png"), bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    # Generate confusion matrix
    y_pred = model.predict(x_balanced)
    conf_matrix = confusion_matrix(y_balanced, y_pred)
    fig_cm = px.imshow(conf_matrix,
                        labels={'x': 'Predicted', 'y': 'Actual'},
                        color_continuous_scale='Blues')
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(ML_SHAP_PLOT_DIR, f"{model_name}/confusion_matrix.png"), bbox_inches='tight')
    
    plt.close()
    # Calculate accuracy score
    accuracy_score = (y_balanced == y_pred).mean()
    
    # Check if pair plot already exists; if not, create it.
    pair_plot_path = os.path.join(ML_SHAP_PLOT_DIR, f"{model_name}/pair_plot.png")
    if not os.path.exists(pair_plot_path):
        print(f"\033[32m[INFO::]\033[00mGenerating pair plots for {model_name}")
        plt.figure(figsize=(10, 8))
        sns.pairplot(data.sample(n=min(1000,len(data))), hue=target_column)  # Sample to reduce time
        plt.savefig(pair_plot_path, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        print(f"\033[32m[INFO::]\033[00mPair plot for '{model_name}' already exists. \033[32mSkipping generation\033[00m].")

    # Calculate and plot the correlation matrix using Seaborn
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    # Get the absolute values of correlations with respect to the target column
    target_correlations = correlation_matrix[target_column].abs()

    # Get the top 3 features correlated with the target column
    top_features = target_correlations.nlargest(4).index[1:]  # Exclude the target itself
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, xticklabels=data.columns, yticklabels=data.columns)
    
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(ML_SHAP_PLOT_DIR, f"{model_name}/correlation_matrix.png"), bbox_inches='tight')
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
    fig_cm = os.path.join(ML_SHAP_PLOT_DIR, f"{model_name}/confusion_matrix.png") 
    return fig_cm, insights_statements  # Returning Plotly figures and summary report

def load_shap_npz(shap_npz_path):
    """
    Load shap_values, base_values, features, and feature_names from npz.
    """
    data = np.load(shap_npz_path, allow_pickle=True)
    shap_values = data['shap_values']  # shape (samples, features)
    base_values = data['base_values']  # shape (samples,) or (samples, something)
    features = data['features']        # shape (samples, features) or similar
    feature_names = data['feature_names'].tolist()  # list of strings
    return shap_values, base_values, features, feature_names

def generate_shap_explanation_from_npz( shap_explainer_path: str, inputs: dict, prediction: int, disease_name: str) -> str:
    """
    Generates a one or two-liner explanation for the prediction using SHAP values from npz file.

    Parameters:
    - shap_explainer_path: path to .npz SHAP file saved with keys: shap_values, base_values, features, feature_names
    - inputs: dict of feature_name -> feature_value for the current input sample
    - prediction: int (0 or 1) - prediction outcome
    - disease_name: str - disease name for textual explanation

    Returns:
    - explanation string
    """
    shap_values, base_values, features, feature_names = load_shap_npz(shap_explainer_path)

    # Find the row index in 'features' that matches the input sample
    # Since features may be an array, do a row-wise comparison
    input_array = np.array([inputs[feat] for feat in feature_names], dtype=float).reshape(1, -1)

    # Find matching row by comparing with all feature rows (exact match)
    match_indices = np.where(np.all(features == input_array, axis=1))[0]

    if len(match_indices) == 0:
        # If no exact match, default to last row (assuming single sample inference)
        row_idx = -1
    else:
        row_idx = match_indices[0]

    # Extract SHAP values for that specific input row
    shap_vals = shap_values[row_idx]  # shape: (features,)

    # Get top 2 features by absolute SHAP value
    top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]

    explanations = []
    for i in top_indices:
        feature = feature_names[i]
        val = inputs.get(feature, 'N/A')
        direction = "increases" if shap_vals[i] < 0 else "decreases"
        explanations.append(f"{feature} ({val}) {direction} likelihood")

    if prediction == 0:
        explanation_text = f"The model predicts no presence of {disease_name}. "
        explanation_text += "Main factors reducing risk: " + ", ".join(explanations) + "."
    else:
        explanation_text = f"The model predicts presence of {disease_name}. "
        explanation_text += "Main contributing factors: " + ", ".join(explanations) + "."

    return explanation_text