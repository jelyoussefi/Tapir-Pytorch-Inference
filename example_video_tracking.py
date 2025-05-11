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
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model file (.pt for PyTorch)')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video file or YouTube URL')
    parser.add_argument('-d', '--device', type=str, default='GPU', choices=['CPU', 'GPU'], help='Device to run the model on: CPU or GPU')
    parser.add_argument('-p', '--precision', type=str, default='FP32', choices=['FP32', 'INT8'], help='Model precision: FP32 or INT8')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Set device
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    # Special handling for INT8 models on XPU
    if args.precision == 'INT8' and str(device).startswith('xpu'):
        print("Note: Using INT8 model on XPU - quantized operators will run on CPU with inputs/outputs on XPU")

    input_size = 256
    num_points = 100
    num_iters = 4  # Use 1 for faster inference, and 4 for better results

    # Initialize video
    if args.input.startswith('http'):
        cap = cap_from_youtube(args.input, resolution="1080p")
    else:
        cap = cv2.VideoCapture(args.input)
    start_time = 0  # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    # Initialize model
    try:
        print(f"Initializing TAPIR model with precision {args.precision}...")
        tapir = TapirInference(args.model, (input_size, input_size), num_iters, device, args.precision)
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

    # Initialize query features
    try:
        print("Initializing query points...")
        query_points = utils.sample_grid_points(input_size, input_size, num_points)
        point_colors = utils.get_colors(num_points)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read the first frame")

        print("Setting initial points...")
        tapir.set_points(frame, query_points)
        print("Initial points set successfully")
    except Exception as e:
        print(f"Error setting initial points: {e}")
        raise

    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    track_length = 30
    tracks = np.zeros((num_points, track_length, 2), dtype=object)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    # FPS calculation variables
    prev_time = time.time()
    fps_avg = 0.0
    alpha = 0.1  # Smoothing factor for moving average

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if args.verbose:
            print(f"Frame shape: {frame.shape}")
            
        height, width = frame.shape[:2]
        
        try:
            tracks_out, visibles = tapir(frame)
        except Exception as e:
            print(f"Error during inference: {e}")
            break

        # Record visible points
        tracks = np.roll(tracks, 1, axis=1)
        tracks[visibles, 0] = tracks_out[visibles]
        tracks[~visibles, 0] = -1

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
        fps_text = f"FPS: {fps_avg:.1f}"
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

    cap.release()
    cv2.destroyAllWindows()