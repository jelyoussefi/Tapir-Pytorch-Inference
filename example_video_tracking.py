# Suppress TensorFlow and IPEX warnings before any imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logs
os.environ['PYTORCH_IPEX_VERBOSE'] = '0'   # Suppress IPEX verbose output
import logging
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('intel_extension_for_pytorch').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)  # Suppress IPEX info logs
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='intel_extension_for_pytorch')

import cv2
import torch
import numpy as np
import argparse
import time
import intel_extension_for_pytorch as ipex
from openvino.runtime import Core
from cap_from_youtube import cap_from_youtube

import tapnet.utils as utils
from tapnet.tapir_inference import TapirInference

def select_device(device_arg):
    if device_arg.lower() == 'cpu':
        return torch.device('cpu')
    elif device_arg.lower() == 'gpu':
        if torch.xpu.is_available():
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
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to input model file (.pt or .onnx)')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video file or YouTube URL')
    parser.add_argument('-d', '--device', type=str, default='GPU', choices=['CPU', 'GPU'], help='Device to run the model on: CPU or GPU')
    parser.add_argument('-ov', '--openvino', action='store_true', help='Use OpenVINO for inference')
    args = parser.parse_args()

    # Set device
    device = select_device(args.device)
    print(f"Using device: {device}")

    input_size = 480
    num_points = 100
    num_iters = 4  # Use 1 for faster inference, and 4 for better results

    # Initialize video
    if args.input.startswith('http'):
        cap = cap_from_youtube(args.input, resolution="1080p")
    else:
        cap = cv2.VideoCapture(args.input)
    start_time = 0  # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    # Initialize OpenVINO if enabled
    if args.openvino:
        predictor_onnx_path = args.model
        encoder_onnx_path = os.path.splitext(args.model)[0] + '_pointencoder.onnx'
        # Export to ONNX if model is .pt and ONNX files don't exist
        if args.model.endswith('.pt'):
            predictor_onnx_path = os.path.splitext(args.model)[0] + '.onnx'
            if not os.path.exists(predictor_onnx_path) or not os.path.exists(encoder_onnx_path):
                # Initialize PyTorch model for export
                tapir = TapirInference(args.model, (input_size, input_size), num_iters, device)
                # Export to ONNX
                causal_state_shape = (num_iters, tapir.model.num_mixer_blocks, num_points, 2, 512 + 2048)
                causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=device)
                feature_grid = torch.zeros((1, input_size//8, input_size//8, 256), dtype=torch.float32, device=device)
                hires_feats_grid = torch.zeros((1, input_size//4, input_size//4, 128), dtype=torch.float32, device=device)
                query_points = torch.zeros((num_points, 2), dtype=torch.float32, device=device)
                input_frame = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32, device=device)
                query_feats = torch.zeros((1, num_points, 256), dtype=torch.float32, device=device)
                hires_query_feats = torch.zeros((1, num_points, 128), dtype=torch.float32, device=device)
                utils.export_to_onnx(
                    predictor=tapir.predictor,
                    encoder=tapir.encoder,
                    input_frame=input_frame,
                    query_points=query_points,
                    feature_grid=feature_grid,
                    hires_feats_grid=hires_feats_grid,
                    query_feats=query_feats,
                    hires_query_feats=hires_query_feats,
                    causal_state=causal_state,
                    output_dir=os.path.dirname(args.model) or '.',
                    predictor_onnx_path=predictor_onnx_path,
                    encoder_onnx_path=encoder_onnx_path,
                    dynamic=False
                )
        # Initialize OpenVINO
        ov_core = Core()
        ov_encoder_model = ov_core.read_model(encoder_onnx_path)
        ov_predictor_model = ov_core.read_model(predictor_onnx_path)
        ov_encoder_compiled = ov_core.compile_model(ov_encoder_model, 'GPU' if 'xpu' in device.type else 'CPU')
        ov_predictor_compiled = ov_core.compile_model(ov_predictor_model, 'GPU' if 'xpu' in device.type else 'CPU')
        ov_encoder_request = ov_encoder_compiled.create_infer_request()
        ov_predictor_request = ov_predictor_compiled.create_infer_request()
    else:
        # Initialize PyTorch model
        tapir = TapirInference(args.model, (input_size, input_size), num_iters, device)

    # Initialize query features
    query_points = utils.sample_grid_points(input_size, input_size, num_points)
    point_colors = utils.get_colors(num_points)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame")

    if args.openvino:
        # Preprocess frame
        input_frame = utils.preprocess_frame(frame, resize=(input_size, input_size), device=device)
        # Initialize PyTorch model temporarily to get feature grids
        tapir_temp = TapirInference(args.model, (input_size, input_size), num_iters, device)
        tapir_temp.set_points(frame, query_points)
        feature_grid = torch.zeros((1, input_size//8, input_size//8, 256), dtype=torch.float32, device=device)
        hires_feats_grid = torch.zeros((1, input_size//4, input_size//4, 128), dtype=torch.float32, device=device)
        query_points_tensor = torch.tensor(query_points, dtype=torch.float32, device=device)
        # Run encoder inference with OpenVINO
        ov_encoder_request.infer({
            'query_points': query_points_tensor[None].cpu().numpy(),
            'feature_grid': feature_grid.cpu().numpy(),
            'hires_feats_grid': hires_feats_grid.cpu().numpy()
        })
        query_feats = torch.from_numpy(ov_encoder_request.get_output_tensor(0).data).to(device)
        hires_query_feats = torch.from_numpy(ov_encoder_request.get_output_tensor(1).data).to(device)
        causal_state = torch.zeros((num_iters, tapir_temp.model.num_mixer_blocks, num_points, 2, 512 + 2048), dtype=torch.float32, device=device)
    else:
        tapir.set_points(frame, query_points)

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

        height, width = frame.shape[:2]
        if args.openvino:
            # Run inference with OpenVINO
            input_frame = utils.preprocess_frame(frame, resize=(input_size, input_size), device=device)
            ov_predictor_request.infer({
                'input_frame': input_frame.cpu().numpy(),
                'query_feats': query_feats.cpu().numpy(),
                'hires_query_feats': hires_query_feats.cpu().numpy(),
                'causal_state': causal_state.cpu().numpy()
            })
            tracks_out = ov_predictor_request.get_output_tensor(0).data
            visibles = ov_predictor_request.get_output_tensor(1).data
            causal_state = torch.from_numpy(ov_predictor_request.get_output_tensor(2).data).to(device)
            tracks_out = tracks_out.squeeze()
            visibles = visibles.squeeze()
            # Postprocess outputs
            tracks_out = tracks_out * np.array([width / input_size, height / input_size])
            visibles = utils.postprocess_occlusions(torch.from_numpy(visibles), torch.ones_like(torch.from_numpy(visibles))).numpy()
        else:
            # Run inference with PyTorch
            tracks_out, visibles = tapir(frame)

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
        font_scale = 1.0
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