import os
import cv2
import torch
import numpy as np
import argparse
import time
from cap_from_youtube import cap_from_youtube
import logging

import tapnet.utils as utils
from tapnet.tapir_inference import TapirInference
from tapnet.tapir_openvino import TapirInferenceOpenVINO


# Print a nice ASCII art banner with application parameters in green
def print_banner(model_path, input_path, device, resolution, num_points, precision):
    """Print a nice ASCII art banner with application parameters."""
    banner_width = 60
    title = "TAPIR Video Tracker"
    padding = (banner_width - len(title) - 2) // 2
    backend = 'OpenVINO' if model_path.endswith(('.onnx', '.xml')) else 'PyTorch'
    green = "\033[32m"  
    red = "\033[31m"
    reset = "\033[0m"  
    
    print("\n" + "-" * banner_width)
    print(" " * padding + title + " " * padding)
    print("-" * banner_width)
    print(f"  | Backend    : {red}{backend}{reset}")
    print(f"  | Model      : {green}{model_path}{reset}")
    print(f"  | Input      : {green}{input_path}{reset}")
    print(f"  | Device     : {red}{device}{reset}")
    print(f"  | Precision  : {green}{precision}{reset}")
    print(f"  | Resolution : {green}{resolution}px{reset}")
    print(f"  | Points     : {green}{num_points}{reset}")
    print("-" * banner_width + "\n")

def select_device(device_arg):
    if device_arg.lower() == 'cpu':
        return torch.device('cpu')
    elif device_arg.lower() == 'gpu':
        if torch.xpu.is_available():
            # For INT8 models, we'll handle device management differently
            # in tapir_inference.py, but still select the device here
            return torch.device('xpu')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("Warning: No XPU or CUDA device available, falling back to CPU")
            return torch.device('cpu')
    else:
        raise ValueError("Device must be 'CPU' or 'GPU'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video tracking with TAPIR model')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model file (.pt for PyTorch, .onnx or .xml for OpenVINO)')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video file or YouTube URL')
    parser.add_argument('-d', '--device', type=str, default='GPU', choices=['CPU', 'GPU', 'NPU'], help='Device to run the model on: CPU or GPU')
    parser.add_argument('-p', '--precision', type=str, default='FP32', choices=['FP32', 'INT8'], help='Model precision: FP32 or INT8')
    parser.add_argument('-r', '--resolution', default=480, type=int, help="Input resolution")
    parser.add_argument('-n', '--num_points', default=100, type=int, help="Number of points")
    args = parser.parse_args()

    # Display banner with application parameters
    print_banner(args.model, args.input, args.device, args.resolution, args.num_points, args.precision)

    # Set device
    print(f"Using device: {args.device}")
    
    # Special handling for INT8 models on XPU
    if args.precision == 'INT8' and str(device).startswith('xpu'):
        print("Note: Using INT8 model on XPU - quantized operators will run on CPU with inputs/outputs on XPU")

    input_size = args.resolution
    num_points = args.num_points
    num_iters = 4  # Use 1 for faster inference, and 4 for better results

    # Initialize video
    if args.input.startswith('http'):
        cap = cap_from_youtube(args.input, resolution="1080p")
    else:
        cap = cv2.VideoCapture(args.input)
    start_time = 0  # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    # Initialize model based on file extension
    
    print(f"Initializing TAPIR model {args.model} with precision {args.precision}...")
     

    model_ext = os.path.splitext(args.model)[1].lower()
    if model_ext in [".onnx", ".xml"]:
        tapir = TapirInferenceOpenVINO(args.model, (input_size, input_size), num_iters, args.device)
    else:
        device = select_device(args.device)
        tapir = TapirInference(args.model, (input_size, input_size), num_iters, device)
            
    

    # Initialize query features
    try:
        query_points = utils.sample_grid_points(input_size, input_size, num_points)
        point_colors = utils.get_colors(num_points)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read the first frame")

        tapir.set_points(frame, query_points)

    except Exception as e:
        print(f"Error setting initial points: {e}")
        raise

    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    track_length = 30
    tracks = np.zeros((num_points, track_length, 2), dtype=object)

    gui_enabled = bool(os.environ.get('DISPLAY'))
    if gui_enabled:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    
    # FPS calculation variables
    prev_time = time.time()
    fps_avg = 0.0
    alpha = 0.1  # Smoothing factor for moving average
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        height, width = frame.shape[:2]
        
        try:
            tracks_out, visibles = tapir(frame)
        except Exception as e:
            print(f"Error during inference: {e}")
            break

        # Record visible points
        tracks = np.roll(tracks, 1, axis=1)
        tracks[visibles, 0] = tracks_out[visibles]
        tracks[~visibles, 0] = [-1, -1]  # Use [-1, -1] to mark invisible points

        # Draw the results
        frame = utils.draw_tracks(frame, tracks, point_colors)
        frame = utils.draw_points(frame, tracks_out, visibles, point_colors)

        # Calculate FPS
        curr_time = time.time()
        delta_time = curr_time - prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 0.0
        fps_avg = alpha * fps + (1 - alpha) * fps_avg if fps_avg > 0 else fps
        prev_time = curr_time

        # Add FPS overlay in top-middle (red text with black background)
        fps_text = f"FPS: {fps_avg:.1f} | Visible: {np.sum(visibles)}/{len(visibles)}"
        if gui_enabled:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 2
            text_color = (0, 0, 255)  # Red in BGR
            text_size, _ = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
            text_x = (frame.shape[1] - text_size[0]) // 2  # Center horizontally
            text_y = text_size[1] + 10  # 10 pixels from top
            # Draw black background rectangle
            bg_padding = 5
            cv2.rectangle(
                frame,
                (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                (0, 0, 0),  # Black in BGR
                -1
            )
            # Draw FPS text
            cv2.putText(
                frame,
                fps_text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"\t{fps_text}")        

    cap.release()
    cv2.destroyAllWindows()