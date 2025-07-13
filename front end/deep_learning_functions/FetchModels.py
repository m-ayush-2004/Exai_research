from deep_learning_functions.GenerateModels import build_inceptionv3_model, build_multiscale_cnn_model, build_resnet_model_with_resize
import matplotlib.pyplot as plt
from PIL import Image
from deep_learning_functions.LoadConfig import load_model_config
import os 
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import time

def get_models(model_name, input_shape=(180, 180, 3), num_classes=2):
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

def train_new(model_path, history_path, time_path, base_dir , model, model_name, disease_name, epochs=30, fetch_existing=True, train_generator= None, val_generator = None):
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

def fetch_existing_model(model_path,history_path,time_path):
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
    base_dir = f"C:/Users/Asus/Desktop/Git Codes/Exai_research/Datasets/weights/{model_name}/{disease_name}"
    os.makedirs(base_dir, exist_ok=True)

    model_path = f"{base_dir}/{disease_name}.h5"
    history_path = f"{base_dir}/history.json"
    time_path = f"{base_dir}/training_time.txt"

    # Check for existing model
    # if fetch_existing and os.path.exists(model_path):
    if fetch_existing:
        return fetch_existing_model(model_path,history_path,time_path)

    else:
        return train_new(
            model_path=model_path,
            history_path=history_path, 
            time_path=time_path, 
            base_dir=base_dir,
            model=model,
            model_name=model_name, 
            disease_name=disease_name, 
            epochs=epochs, 
            train_generator=train_generator, 
            val_generator=val_generator
        )



# Load model from weights directory
def load_model_from_weights(model_name, disease_name, config_path='front end/config/config2.json'):
    models = load_model_config(config_path)
    # print(list(os.listdir("./Datasets")))
    # print(list(os.listdir("../")))
    
    if model_name in models and disease_name in models[model_name]:
        weights_path = models[model_name][disease_name]["weights"]
        model, plot_path = get_models(model_name,
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

