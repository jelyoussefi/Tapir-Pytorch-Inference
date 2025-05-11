import os
import torch
import numpy as np
import pickle
from PIL import Image
import io
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class TapVidDataset(Dataset):
    """Dataset class for TapVid dataset, compatible with the evaluation script"""
    
    def __init__(self, dataset_path, resize=(256, 256)):
        """
        Initialize TapVid dataset
        
        Args:
            dataset_path: Path to the dataset directory
            resize: Resolution to resize frames to (height, width)
        """
        self.gt = []
        self.resize = resize
        self.dataset_path = dataset_path

        # Load the dataset based on the type
        if 'tapvid_davis' in self.dataset_path:
            self._load_davis_dataset()
        elif 'tapvid_rgb_stacking' in self.dataset_path:  # Fixed typo: removed 'elf'
            self._load_rgb_stacking_dataset()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_path}")
        
        print(f"Loaded {len(self.gt)} video sequences from {dataset_path}")
        
    def _load_davis_dataset(self):
        """Load TapVid Davis dataset"""
        with open(os.path.join(self.dataset_path, 'tapvid_davis.pkl'), 'rb') as f:
            self.vid_gt = pickle.load(f)  # type: dict
        
        for vid, vid_gt in self.vid_gt.items():
            self.gt.append(self._process_video_data(vid, vid_gt))
    
    def _load_rgb_stacking_dataset(self):
        """Load TapVid RGB stacking dataset"""
        with open(os.path.join(self.dataset_path, 'tapvid_rgb_stacking.pkl'), 'rb') as f:
            self.vid_gt = pickle.load(f)  # type: list
        
        for i, vid_gt in enumerate(self.vid_gt):
            self.gt.append(self._process_video_data(str(i), vid_gt))
    
    def _process_video_data(self, vid, vid_gt):
        """
        Process video data to extract frames, trajectories, and visibility information
        
        Args:
            vid: Video ID
            vid_gt: Ground truth data for the video
            
        Returns:
            Dictionary with processed video data
        """
        # Process RGB frames
        rgbs = vid_gt['video']  # list of H,W,C uint8 images
        if isinstance(rgbs[0], bytes):
            rgbs = [np.array(Image.open(io.BytesIO(rgb))) for rgb in rgbs]
        
        # Convert to tensor and normalize
        rgbs = torch.from_numpy(np.stack(rgbs)).float()
        rgbs = (2 * (rgbs / 255.0) - 1.0)  # Normalize to [-1, 1]
        
        # Resize frames
        rgbs = F.resize(rgbs.permute(0, 3, 1, 2), self.resize, antialias=True).permute(0, 2, 3, 1)
        
        # Process trajectories and occlusion information
        trajs = vid_gt['points']  # N,S,2 array normalized coordinates
        valids = 1 - vid_gt['occluded']  # N,S array (1 = visible, 0 = occluded)
        
        # Keep only trajectories that start visible
        vis_ok = valids[:, 0] > 0
        
        sample = {
            'vid': vid,
            'rgbs': rgbs,  # S,H,W,C (sequence, height, width, channels)
            'trajs': torch.from_numpy(trajs[vis_ok]).float(),  # N,S,2 (num_points, sequence, xy)
            'visibs': torch.from_numpy(valids[vis_ok]).float(),  # N,S (num_points, sequence)
        }
        return sample
    
    def __len__(self):
        return len(self.gt)
    
    def __getitem__(self, idx):
        return self.gt[idx]


