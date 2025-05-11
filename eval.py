#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import TAPIR modules
from tapnet.tapir_inference import TapirInference
from dataset.tapir_dataset import TapVidDataset

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

def compute_metrics(trajs_g, visibs_g, trajs_e, visibs_e):
    """
    Compute evaluation metrics comparing model outputs with ground truth
    
    Args:
        trajs_g: Ground truth trajectories [N, S, 2]
        visibs_g: Ground truth visibility [N, S]
        trajs_e: Estimated trajectories [N, S, 2]
        visibs_e: Estimated visibility [N, S]
        
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy arrays if they aren't already
    if isinstance(visibs_e, torch.Tensor):
        visibs_e = visibs_e.cpu().numpy()
    if isinstance(visibs_g, torch.Tensor):
        visibs_g = visibs_g.cpu().numpy()
    if isinstance(trajs_e, torch.Tensor):
        trajs_e = trajs_e.cpu().numpy()
    if isinstance(trajs_g, torch.Tensor):
        trajs_g = trajs_g.cpu().numpy()
    
    # Make sure arrays have the same type for comparison
    visibs_e = visibs_e.astype(np.float32)
    visibs_g = visibs_g.astype(np.float32)
    
    # Compute occlusion accuracy (how well the model predicts point visibility)
    occ_acc = np.mean((visibs_e > 0.5) == (visibs_g > 0.5))
    
    # Compute endpoint error (distance between predicted and ground truth points)
    # Only consider visible points according to ground truth
    visible_mask = visibs_g > 0.5
    if np.sum(visible_mask) > 0:
        epe = np.sqrt(np.sum((trajs_e - trajs_g)**2, axis=-1))
        epe_visible = epe[visible_mask]
        epe_mean = np.mean(epe_visible)
        epe_median = np.median(epe_visible)
    else:
        epe_mean = np.nan
        epe_median = np.nan
    
    # PCK (Percentage of Correct Keypoints) under different thresholds
    # Shows how many points are tracked within specific error thresholds
    thresholds = [0.01, 0.02, 0.05, 0.1]
    pck_metrics = {}
    for threshold in thresholds:
        pck = np.mean((epe < threshold) & visible_mask) if np.sum(visible_mask) > 0 else np.nan
        pck_metrics[f'pck_{int(threshold*100)}'] = pck
    
    # Compute temporal consistency - how error changes over consecutive frames
    temporal_consistency = np.nan
    if np.sum(visible_mask) > 0:
        # Calculate frame-to-frame movement
        frame_diffs = []
        num_points, num_frames = trajs_g.shape[:2]
        for p in range(num_points):
            for f in range(1, num_frames):
                if visibs_g[p, f] > 0.5 and visibs_g[p, f-1] > 0.5:
                    # Ground truth movement
                    gt_diff = np.sqrt(np.sum((trajs_g[p, f] - trajs_g[p, f-1])**2))
                    # Predicted movement
                    pred_diff = np.sqrt(np.sum((trajs_e[p, f] - trajs_e[p, f-1])**2))
                    # Difference in movement magnitude
                    frame_diffs.append(abs(gt_diff - pred_diff))
        
        if frame_diffs:
            temporal_consistency = 1.0 - min(1.0, np.mean(frame_diffs) * 10)  # Scale for interpretability
    
    # Combine all metrics
    metrics = {
        'occlusion_accuracy': float(occ_acc),  # Higher is better
        'epe_mean': float(epe_mean),          # Lower is better 
        'epe_median': float(epe_median),      # Lower is better
        'temporal_consistency': float(temporal_consistency) if not np.isnan(temporal_consistency) else np.nan,  # Higher is better (max 1.0)
    }
    
    # Add PCK metrics
    for k, v in pck_metrics.items():
        metrics[k] = float(v) if not np.isnan(v) else np.nan
    
    return metrics

def evaluate_model(model, dataset, num_points_per_run=15):
    """
    Evaluate model on dataset by comparing with ground truth
    
    Args:
        model: TapirInference model
        dataset: Evaluation dataset
        num_points_per_run: Number of points to process at once
        
    Returns:
        DataFrame with evaluation metrics
    """
    results = []
    n = num_points_per_run
    
    for data_idx in tqdm(range(min(10, len(dataset))), desc="Evaluating videos"):
        try:
            # Get video data
            video_data = dataset[data_idx]
            video_frames = video_data['rgbs']  # S,H,W,C format
            video_id = video_data['vid']
            
            # Get ground truth trajectories and visibility
            trajs_g = video_data['trajs']  # N,S,2 format (normalized coordinates)
            visibs_g = video_data['visibs']  # N,S format
            
            num_points = trajs_g.shape[0]
            num_frames = video_frames.shape[0]
            height, width = video_frames.shape[1:3]
            
            # Initialize storage for estimated trajectories and visibility
            trajs_e = np.zeros_like(trajs_g)
            visibs_e = np.zeros_like(visibs_g)
            
            # Process points in chunks to avoid memory issues
            for start_idx in range(0, num_points, n):
                end_idx = min(start_idx + n, num_points)
                chunk_size = end_idx - start_idx
                
                # Get initial points from first frame (convert from normalized to pixel coordinates)
                query_points = np.zeros((chunk_size, 2), dtype=np.float32)
                query_points[:, 0] = trajs_g[start_idx:end_idx, 0, 0] * width  # x coordinate
                query_points[:, 1] = trajs_g[start_idx:end_idx, 0, 1] * height  # y coordinate
                
                # Initialize tracking with first frame
                first_frame = video_frames[0]
                model.set_points(first_frame, query_points)
                
                # Track points through all frames
                for frame_idx in range(num_frames):
                    # Get current frame
                    frame = video_frames[frame_idx]
                    
                    # Track points
                    tracks, visibles = model(frame)
                    
                    # Store results (normalized coordinates)
                    trajs_e[start_idx:end_idx, frame_idx, 0] = tracks[:, 0] / width
                    trajs_e[start_idx:end_idx, frame_idx, 1] = tracks[:, 1] / height
                    visibs_e[start_idx:end_idx, frame_idx] = visibles
            
            # Compute metrics for this video
            metrics = compute_metrics(trajs_g, visibs_g, trajs_e, visibs_e)
            
            # Add video info
            metrics.update({
                'vid': video_id,
                'num_frames': num_frames,
                'num_points': num_points
            })
            
            results.append(metrics)
            
        except Exception as e:
            print(f"Error processing video at index {data_idx}: {e}")
            continue
    
    if not results:
        print("No valid results were produced. Check for errors in the evaluation process.")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(results).set_index('vid')
    
    # Add average row
    df.loc['avg'] = df.mean(numeric_only=True)
    
    return df

def plot_metrics(df, output_path=None):
    """
    Create visualizations of model performance metrics
    
    Args:
        df: DataFrame with evaluation metrics
        output_path: Path to save visualization
    """
    if df.empty:
        print("No data to visualize")
        return
    
    # Get video IDs (excluding the average row)
    videos = df.index.tolist()
    if 'avg' in videos:
        videos.remove('avg')
    
    if not videos:
        print("No videos to visualize")
        return
    
    # Extract key metrics
    metrics = ['occlusion_accuracy', 'epe_mean', 'epe_median', 'pck_1', 'pck_2', 'pck_5', 'temporal_consistency']
    metrics = [m for m in metrics if m in df.columns]
    
    if not metrics:
        print("No metrics to visualize")
        return
    
    metric_labels = []
    for m in metrics:
        if m == 'occlusion_accuracy':
            metric_labels.append('Occlusion Accuracy')
        elif m == 'epe_mean':
            metric_labels.append('Mean EPE')
        elif m == 'epe_median':
            metric_labels.append('Median EPE')
        elif m == 'temporal_consistency':
            metric_labels.append('Temporal Consistency')
        elif m.startswith('pck_'):
            threshold = m.split('_')[1]
            metric_labels.append(f'PCK@0.{threshold}')
        else:
            metric_labels.append(m)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot metrics for each video
    x = np.arange(len(videos))
    width = 0.1
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        if metric in df.columns:
            values = [df.loc[v, metric] if not pd.isna(df.loc[v, metric]) else 0 for v in videos]
            plt.bar(x + i*width - (len(metrics)-1)*width/2, values, width, label=label, color=color)
    
    # Add average performance line for each metric
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        if metric in df.columns and 'avg' in df.index and not pd.isna(df.loc['avg', metric]):
            avg_value = df.loc['avg', metric]
            plt.axhline(y=avg_value, linestyle='--', color=color, alpha=0.7, 
                        label=f'Avg {label}: {avg_value:.3f}')
    
    # Add labels and legend
    plt.xlabel('Video')
    plt.ylabel('Metric Value')
    plt.title('FP32 Model Performance Metrics')
    plt.xticks(x, videos, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save or show figure
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Performance visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def print_model_quality_summary(df):
    """
    Print a human-readable summary of model quality
    
    Args:
        df: DataFrame with evaluation metrics
    """
    if df.empty or 'avg' not in df.index:
        print("No data available for quality summary")
        return
    
    # Get average metrics
    avg_metrics = df.loc['avg']
    
    # Define expected ranges for good performance
    good_occ_acc = 0.7
    good_epe = 0.05
    good_pck5 = 0.7
    
    # Print summary header
    print("\n" + "="*50)
    print("     TAPIR MODEL QUALITY ASSESSMENT")
    print("="*50)
    
    # Print key metrics
    if 'occlusion_accuracy' in avg_metrics:
        print(f"\nOcclusion Accuracy: {avg_metrics['occlusion_accuracy']:.3f}")
    if 'epe_mean' in avg_metrics:
        print(f"Mean Endpoint Error: {avg_metrics['epe_mean']:.3f}")
    if 'epe_median' in avg_metrics:
        print(f"Median Endpoint Error: {avg_metrics['epe_median']:.3f}")
    
    # Print PCK metrics
    pck_available = False
    for threshold in [1, 2, 5, 10]:
        if f'pck_{threshold}' in avg_metrics and not pd.isna(avg_metrics[f'pck_{threshold}']):
            if not pck_available:
                print("\nPercentage of Correct Keypoints (PCK):")
                pck_available = True
            print(f"  PCK@0.{threshold:02d}: {avg_metrics[f'pck_{threshold}']:.3f}")
    
    # Print temporal consistency if available
    if 'temporal_consistency' in avg_metrics and not pd.isna(avg_metrics['temporal_consistency']):
        print(f"\nTemporal Consistency: {avg_metrics['temporal_consistency']:.3f}")
    
    # Check if we have enough metrics to provide an assessment
    if not ('occlusion_accuracy' in avg_metrics and 'epe_mean' in avg_metrics and 'pck_5' in avg_metrics):
        print("\nNot enough metrics available for quality assessment")
        return
        
    # Provide overall assessment
    print("\n" + "-"*50)
    print("QUALITY ASSESSMENT:")
    
    if avg_metrics['occlusion_accuracy'] > good_occ_acc and \
       avg_metrics['epe_mean'] < good_epe and \
       avg_metrics.get('pck_5', 0) > good_pck5:
        quality = "EXCELLENT"
    elif avg_metrics['occlusion_accuracy'] > good_occ_acc*0.8 and \
         avg_metrics['epe_mean'] < good_epe*1.5 and \
         avg_metrics.get('pck_5', 0) > good_pck5*0.8:
        quality = "GOOD"
    elif avg_metrics['occlusion_accuracy'] > good_occ_acc*0.6 and \
         avg_metrics['epe_mean'] < good_epe*2.5 and \
         avg_metrics.get('pck_5', 0) > good_pck5*0.6:
        quality = "FAIR"
    else:
        quality = "NEEDS IMPROVEMENT"
    
    print(f"The FP32 model quality is: {quality}")
    
    # Provide specific strengths and weaknesses
    print("\nStrengths:")
    if 'occlusion_accuracy' in avg_metrics and avg_metrics['occlusion_accuracy'] > good_occ_acc*0.8:
        print("- Good at determining when points are visible or occluded")
    if 'epe_mean' in avg_metrics and 'epe_median' in avg_metrics and avg_metrics['epe_median'] < avg_metrics['epe_mean']*0.7:
        print("- Consistent tracking for most points (low median error)")
    if 'pck_5' in avg_metrics and avg_metrics.get('pck_5', 0) > good_pck5*0.8:
        print("- High percentage of points tracked within acceptable threshold")
    
    print("\nAreas for improvement:")
    if 'occlusion_accuracy' in avg_metrics and avg_metrics['occlusion_accuracy'] < good_occ_acc:
        print("- Could improve occlusion detection")
    if 'epe_mean' in avg_metrics and avg_metrics['epe_mean'] > good_epe:
        print("- Could reduce tracking error")
    if 'pck_5' in avg_metrics and avg_metrics.get('pck_5', 0) < good_pck5:
        print("- Could increase percentage of correctly tracked points")
    
    print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description='Evaluate TAPIR model quality against dataset ground truth')
    
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='Path to FP32 PyTorch model')
    
    parser.add_argument('-d', '--dataset', type=str, default='/workspace/dataset/tapvid_davis/',
                       help='Path to the TapVid dataset directory')
    
    parser.add_argument('--resize', type=int, nargs=2, default=[256, 256],
                       help='Resolution to resize frames to (height width)')
    
    parser.add_argument('--num_points', type=int, default=100,
                       help='Number of points to process per run')
    
    parser.add_argument('--output_dir', type=str, default='model_quality',
                       help='Directory to save evaluation results')
    
    parser.add_argument('--device', type=str, default='CPU',
                       help='Device to run evaluation on (cpu or gpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = TapVidDataset(args.dataset, resize=tuple(args.resize))
    
    # Initialize model
    print(f"Initializing FP32 model: {args.model}")
    model = TapirInference(args.model, tuple(args.resize), 4, device, "FP32")
    
    # Evaluate model
    print("Evaluating model against dataset ground truth...")
    metrics_df = evaluate_model(model, dataset, args.num_points)
    
    # Check if we got valid results
    if metrics_df.empty:
        print("No valid evaluation results were produced.")
        return
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'model_quality_metrics.csv')
    metrics_df.to_csv(metrics_file)
    print(f"Metrics saved to {metrics_file}")
    
    # Create visualization
    viz_file = os.path.join(args.output_dir, 'model_quality_visualization.png')
    plot_metrics(metrics_df, viz_file)
    
    # Print summary
    print_model_quality_summary(metrics_df)

if __name__ == '__main__':
    main()