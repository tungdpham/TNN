"""
Visualize CIFAR-10 CNN Architecture using visualkeras
This script creates a visual representation of the CNN model architecture
defined in cifar10_cnn_trainer_v2.cpp
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import visualkeras
from PIL import ImageFont

def create_cifar10_cnn_model():
    """
    Create a Keras model matching the architecture in cifar10_cnn_trainer_v2.cpp
    """
    model = keras.Sequential(name="cifar10_cnn_classifier_v2")
    
    # Input layer: 3x32x32 (RGB image)
    model.add(layers.Input(shape=(32, 32, 3), name="input"))
    
    # Block 1: conv0 -> bn0 -> relu0 -> conv1 -> bn1 -> relu1 -> pool0
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv0"))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn0"))
    model.add(layers.Activation('relu', name="relu0"))
    
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv1"))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn1"))
    model.add(layers.Activation('relu', name="relu1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), 
                                 padding='valid', name="pool0"))
    
    # Block 2: conv2 -> bn2 -> relu2 -> conv3 -> bn3 -> relu3 -> pool1
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv2"))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn2"))
    model.add(layers.Activation('relu', name="relu2"))
    
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv3"))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn3"))
    model.add(layers.Activation('relu', name="relu3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), 
                                 padding='valid', name="pool1"))
    
    # Additional batch norm and activation
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn4_a"))
    model.add(layers.Activation('relu', name="relu4_a"))
    
    # Block 3: conv4 -> bn5 -> relu5 -> conv5 -> bn6 -> relu6 -> conv6 -> bn6_b -> pool2
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv4"))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn5"))
    model.add(layers.Activation('relu', name="relu5"))
    
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv5"))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn6"))
    model.add(layers.Activation('relu', name="relu6"))
    
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv6"))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn6_b"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), 
                                 padding='valid', name="pool2"))
    
    # Additional batch norm
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn3_b"))
    
    # Block 4: conv7 -> conv8 -> conv9 -> bn4_c -> relu4_b -> pool3
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv7"))
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv8"))
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', 
                           use_bias=True, name="conv9"))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn4_c"))
    model.add(layers.Activation('relu', name="relu4_b"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), 
                                 padding='valid', name="pool3"))
    
    # Additional batch norm
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn4_d"))
    
    # Fully connected layers
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(512, use_bias=True, name="fc0"))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn3_c"))
    model.add(layers.Activation('relu', name="relu3_c"))
    model.add(layers.Dense(10, use_bias=True, name="fc1"))  # 10 classes for CIFAR-10
    
    return model


def visualize_architecture(output_file="cifar10_architecture.png"):
    """
    Create and visualize the CNN architecture
    """
    print("Creating CIFAR-10 CNN model...")
    model = create_cifar10_cnn_model()
    
    # Print model summary
    print("\nModel Summary:")
    print("=" * 80)
    model.summary()
    
    # Create visualization
    print(f"\nGenerating architecture visualization to {output_file}...")
    
    try:
        # Try to use a custom font if available, otherwise use default
        try:
            font = ImageFont.truetype("arial.ttf", 16)  # Smaller font
        except:
            font = None
        
        # Create a COMPACT version (recommended for bloated architectures)
        output_file_compact = output_file.replace('.png', '_compact.png')
        visualkeras.layered_view(
            model,
            to_file=output_file_compact,
            legend=True,
            font=font,
            scale_xy=5,        # Smaller scale
            scale_z=0.03,      # Much smaller z-scale (reduces channel height)
            min_z=10,           # Minimum width for all layers (controls initial z value)
            max_z=100,          # Lower cap on maximum height
            spacing=3,         # Tighter spacing between layers
            draw_volume=True
        )
        print(f"✓ Compact architecture visualization saved to {output_file_compact}")
        
    except Exception as e:
        print(f"Error creating layered view: {e}")
        print("Trying alternative visualization method...")
        
        try:
            # Alternative: graph view
            output_file_alt = output_file.replace('.png', '_graph.png')
            visualkeras.graph_view(
                model,
                to_file=output_file_alt,
                legend=True
            )
            print(f"✓ Alternative visualization saved to {output_file_alt}")
        except Exception as e2:
            print(f"Error creating graph view: {e2}")
    
    # Save model architecture as JSON
    json_file = output_file.replace('.png', '_architecture.json')
    with open(json_file, 'w') as f:
        f.write(model.to_json())
    print(f"✓ Model architecture JSON saved to {json_file}")
    
    return model


def print_layer_details(model):
    """
    Print detailed information about each layer
    """
    print("\n" + "=" * 80)
    print("Detailed Layer Information")
    print("=" * 80)
    
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i}: {layer.name}")
        print(f"  Type: {layer.__class__.__name__}")
        try:
            print(f"  Output Shape: {layer.output.shape}")
        except:
            print(f"  Output Shape: N/A")
        
        if hasattr(layer, 'filters'):
            print(f"  Filters: {layer.filters}")
        if hasattr(layer, 'kernel_size'):
            print(f"  Kernel Size: {layer.kernel_size}")
        if hasattr(layer, 'strides'):
            print(f"  Strides: {layer.strides}")
        if hasattr(layer, 'pool_size'):
            print(f"  Pool Size: {layer.pool_size}")
        if hasattr(layer, 'units'):
            print(f"  Units: {layer.units}")
        
        # Count parameters
        params = layer.count_params()
        print(f"  Parameters: {params:,}")


if __name__ == "__main__":
    print("CIFAR-10 CNN Architecture Visualization")
    print("=" * 80)
    print("This script visualizes the CNN architecture from cifar10_cnn_trainer_v2.cpp")
    print()
    
    # Create and visualize the model
    model = visualize_architecture("cifar10_cnn_architecture.png")
    
    # Print detailed layer information
    print_layer_details(model)
    
    # Print total parameters
    total_params = model.count_params()
    print("\n" + "=" * 80)
    print(f"Total Parameters: {total_params:,}")
    print("=" * 80)
    
    print("\n✓ Visualization complete!")
    print("\nGenerated files:")
    print("  - cifar10_cnn_architecture_compact.png (RECOMMENDED: compact, less bloated)")
    print("  - cifar10_cnn_architecture_full.png (full size, all layers shown)")
    print("  - cifar10_cnn_architecture_graph.png (graph view, most compact)")
    print("  - cifar10_cnn_architecture_architecture.json (model JSON)")
