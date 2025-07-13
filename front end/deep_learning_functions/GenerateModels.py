import tensorflow as tf
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
