import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import argparse
import os

def preprocess_image(image_path, input_shape):
    """
    Preprocess the image for the ONNX model.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # Original dimensions (width, height)

    # Resize the image to the model's expected height and width
    _, _, height, width = input_shape
    image = image.resize((width, height))

    # Convert the image to a NumPy array and normalize pixel values
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Transpose dimensions to match model's input format: [N, C, H, W]
    image_array = np.transpose(image_array, (2, 0, 1))  # From HWC to CHW

    # Add batch dimension: [1, C, H, W]
    image_array = np.expand_dims(image_array, axis=0)

    return image_array, orig_size

def main(model_path, image_path):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Get input details
    input_details = session.get_inputs()
    input_name = input_details[0].name
    orig_target_sizes_name = None
    if len(input_details) > 1:
        orig_target_sizes_name = input_details[1].name

    # Preprocess the image
    input_shape = input_details[0].shape
    input_data, orig_size = preprocess_image(image_path, input_shape)

    # Prepare the additional input if required
    inputs = {input_name: input_data}
    if orig_target_sizes_name:
        # Pass original image dimensions as a NumPy array with shape [batch_size, 2]
        orig_target_sizes = np.array([[orig_size[1], orig_size[0]]], dtype=np.int64)  # Use int64
        inputs[orig_target_sizes_name] = orig_target_sizes

    # Warm-up run (optional but recommended)
    session.run(None, inputs)

    # Measure inference time
    start_time = time.time()
    output = session.run(None, inputs)
    end_time = time.time()

    # Print inference time
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")

    # Output details
    output_name = session.get_outputs()[0].name
    print(f"Output shape: {output[0].shape}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Test ONNX model inference time.")
    # parser.add_argument("model_path", type=str, help="Path to the ONNX model file")
    # parser.add_argument("input_file", type=str, help="Path to the input file (.npy format)")
    # args = parser.parse_args()
    # main(args.model_path, args.input_file)
    model_path = os.path.join("mixed", "results", "rtdetr-s_117", "model.onnx")
    input_file = os.path.join("mixed", "dataset", "yolo_format", "test", "images", "-144-_png_jpg.rf.5c0c11868d7cc42af7590b0524c5fe2b.jpg")
    main(model_path, input_file)
