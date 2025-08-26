import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from .explainer import *
from .config_functions import *

# Configurations
load_dotenv()
ML_DATA_UPLOAD_FOLDER = os.getenv('ML_DATA_UPLOAD_FOLDER')
MODEL_WEIGHTS_PATH = os.getenv('ML_WEIGHTS_PATH')
SHAP_VALUES_DIR = os.getenv('ML_SHAP_VALUES_DIR')
SHAP_EXPLAINER_DIR = os.getenv('ML_SHAP_EXPLAINER_DIR')
SHAP_PLOT_DIR = os.getenv('SHAP_PLOT_DIR')

MAX_SHAP_SAMPLES = 200  # Set maximum number of samples for SHAP

def preprocess_data(data,target_column):
    """
    The function preprocesses data by encoding categorical features, balancing classes, and returning a
    balanced dataset.
    
    Args:
      data: The `data` parameter in the `preprocess_data` function is the dataset that you want to
    preprocess. It should be a pandas DataFrame containing the features and the target column that you
    want to balance.
      target_column: The `target_column` parameter in the `preprocess_data` function is the name of the
    column in the dataset that contains the target variable or the class labels. This column is used to
    identify the classes for balancing the dataset by downsampling the larger classes to match the size
    of the smallest class.
    
    Returns:
      The function `preprocess_data` returns a balanced dataset where the classes have been balanced by
    downsampling larger classes to double the size of the minimum class.
    """
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

def train_and_save_model(data, target_column, model_name):
    model_path = os.path.join(MODEL_WEIGHTS_PATH, f"{model_name}.json")
    shap_values_path = os.path.join(SHAP_VALUES_DIR, f"{model_name}.npz")
    shap_explainer_path = os.path.join(SHAP_EXPLAINER_DIR, f"{model_name}.pkl")  # Path for saving the explainer
    
    # Check if model and SHAP values already exist
    if os.path.exists(model_path) and os.path.exists(shap_values_path):
        print(f"\033[32m[INFO::]\033[00mModel and SHAP values for '{model_name}' already exist. Skipping training.")
        return model_path
    print(data)
    # Separate features and target from balanced data
    x_balanced = data.drop(columns=[target_column])
    y_balanced = data[target_column]

    # Apply SMOTE to create a balanced dataset with equal number of instances for both classes
    smote = SMOTE()
    try:
        # Fit SMOTE only if there is an imbalance after downsampling
        x_resampled, y_resampled = smote.fit_resample(x_balanced, y_balanced)
    except:
        print("\033[31m[INFO::]\033[00mSynthetic data generation using Smote failed")
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

def run_model(selected_model, inputs):
    model_path = os.path.join(MODEL_WEIGHTS_PATH, f"{selected_model}.json")
    shap_explainer_path = os.path.join(SHAP_EXPLAINER_DIR, f"{selected_model}.pkl")  # Path for SHAP explainer
    shap_values_path = os.path.join(SHAP_VALUES_DIR, f"{selected_model}.npz")  # Path for SHAP explainer

    # Load the model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Prepare input data for prediction
    input_array = np.array([list(inputs.values())], dtype=float)  # Reshape inputs for prediction
    print(input_array)
    # Get predictions
    predictions = model.predict(input_array)

    # Generate SHAP force plot and save it as an image file
    force_plot_path = os.path.join(ML_SHAP_PLOT_DIR, f"{selected_model}/force_plot.png")
    
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
    explanation_str = generate_shap_explanation_from_npz(shap_explainer_path=shap_values_path,inputs=inputs, disease_name=selected_model, prediction=predictions )
    return predictions[0], force_plot_path, explanation_str  # Return prediction and force plot HTML