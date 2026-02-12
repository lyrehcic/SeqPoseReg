import numpy as np
import json
import cv2
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
import torch
import torch.nn.functional as F

ANGLE_JOINTS = [
    'left_elbow', 'right_elbow', 'left_knee', 'right_knee',
    'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
    'shoulderL_center_wristL', 'shoulderR_center_wristR',
    'hipL_center_ankleL', 'hipR_center_ankleR',
    'wristL_center_ankleL', 'wristR_center_ankleR'
]
DISTANCE_NAMES = [
    'shoulder_left_to_center', 'shoulder_right_to_center',
    'elbow_left_to_center', 'elbow_right_to_center',
    'wrist_left_to_center', 'wrist_right_to_center',
    'hip_left_to_center', 'hip_right_to_center',
    'knee_left_to_center', 'knee_right_to_center',
    'ankle_left_to_center', 'ankle_right_to_center'
]
'''
WEIGHT_CONFIG = {
    'arm_angles': {'joints': ['elbow', 'wrist'], 'weight': 1.5, 'derivative_ratio': 0.6},
    'leg_angles': {'joints': ['knee', 'ankle'], 'weight': 1.3, 'derivative_ratio': 0.7},
    'torso_angles': {'joints': ['shoulder', 'hip'], 'weight': 1.0, 'derivative_ratio': 0.4},
    'arm_distances': {'dists': ['elbow', 'wrist'], 'weight': 1.4, 'derivative_ratio': 0.5},
    'leg_distances': {'dists': ['knee', 'ankle'], 'weight': 1.2, 'derivative_ratio': 0.6},
    'torso_distances': {'dists': ['shoulder', 'hip'], 'weight': 1.0, 'derivative_ratio': 0.3}
}
'''
WEIGHT_CONFIG = {
    'arm_angles': {'joints': ['elbow', 'wrist'], 'weight': 1.5, 'derivative_ratio': 0},
    'leg_angles': {'joints': ['knee', 'ankle'], 'weight': 1.3, 'derivative_ratio': 0},
    'torso_angles': {'joints': ['shoulder', 'hip'], 'weight': 1.0, 'derivative_ratio': 0},
    'arm_distances': {'dists': ['elbow', 'wrist'], 'weight': 1.4, 'derivative_ratio': 0},
    'leg_distances': {'dists': ['knee', 'ankle'], 'weight': 1.2, 'derivative_ratio': 0},
    'torso_distances': {'dists': ['shoulder', 'hip'], 'weight': 1.0, 'derivative_ratio': 0}
}

class FeatureProcessor:
    def __init__(self, max_null_ratio=0.4):
        self.max_null_ratio = max_null_ratio
        self.feature_weights = None
        self.valid_dims = None

    def _get_feature_type(self, idx):

        if idx < len(ANGLE_JOINTS):
            name = ANGLE_JOINTS[idx]
            for key, cfg in WEIGHT_CONFIG.items():
                if 'joints' in cfg and any(j in name for j in cfg['joints']):
                    return key, cfg
        else:
            name = DISTANCE_NAMES[idx - len(ANGLE_JOINTS)]
            for key, cfg in WEIGHT_CONFIG.items():
                if 'dists' in cfg and any(d in name for d in cfg['dists']):
                    return key, cfg
        return 'default', {'weight':1.0, 'derivative_ratio':0.5}

    def preprocess(self, data):

        raw_features = np.array([
            [frame['angles'].get(j, np.nan) for j in ANGLE_JOINTS] +
            [frame['distances'].get(d, np.nan) for d in DISTANCE_NAMES]
            for frame in data
        ], dtype=np.float32).T  # (26, N)

        valid_dims = []
        dim_weights = []
        for dim_idx in range(raw_features.shape[0]):
            null_ratio = np.isnan(raw_features[dim_idx]).mean()
            if null_ratio <= self.max_null_ratio:
                valid_dims.append(dim_idx)
                ftype, cfg = self._get_feature_type(dim_idx)
                dim_weights.append((
                    cfg['weight'], 
                    cfg['weight'] * cfg['derivative_ratio']
                ))

        processed = []
        self.feature_weights = []
        for dim_idx, (w_base, w_deriv) in zip(valid_dims, dim_weights):
            dim_data = raw_features[dim_idx].copy()

            valid_mask = ~np.isnan(dim_data)
            if valid_mask.sum() == 0:
                filled = np.zeros_like(dim_data)
            else:
                x_valid = np.where(valid_mask)[0]
                if len(x_valid) < 2:
                    filled = np.full_like (dim_data, dim_data[x_valid[0]] if len(x_valid)==1 else np.zeros_like(dim_data))
                else:
                    interp_fn = interp1d(x_valid, dim_data[x_valid], 
                                      kind='quadratic', fill_value="extrapolate")
                    filled = interp_fn(np.arange(len(dim_data)))
            
            derivatives = np.gradient(filled)

            invalid_segs = self._find_invalid_segments(dim_data)
            for s,e in invalid_segs:
                filled[s:e+1] = np.nan
                derivatives[s:e+1] = np.nan
            
            processed.append(np.stack([filled, derivatives], axis=1))
            self.feature_weights.extend([w_base, w_deriv])

        self.valid_dims = valid_dims
        return np.concatenate(processed, axis=1) if processed else np.zeros((raw_features.shape[1], 0))

    def _find_invalid_segments(self, data, window=15, threshold=0.5):

        n = len(data)
        invalid = np.zeros(n, dtype=bool)
        for i in range(n - window + 1):
            if np.isnan(data[i:i+window]).mean() > threshold:
                invalid[i:i+window] = True

        segments = []
        start = None
        for i in range(n):
            if invalid[i]:
                if start is None: start = i
                end = i
            elif start is not None:
                segments.append((start, end))
                start = None
        if start is not None: segments.append((start, n-1))
        return segments

class VideoAligner:
    def __init__(self, template_feat, full_feat, weights, fps=30):
        self.template = self._normalize(template_feat)
        self.full = self._normalize(full_feat)
        self.weights = torch.tensor(weights, dtype=torch.float32).to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.fps = fps
        self.device = self.weights.device

    def _normalize(self, data):

        data = np.nan_to_num(data, nan=0.0)
        return (data - np.nanmean(data, axis=0)) / (np.nanstd(data, axis=0) + 1e-8)

    def _weighted_dtw(self, x, y):

        x = torch.tensor(x, device=self.device)
        y = torch.tensor(y, device=self.device)

        mask = (~torch.isnan(x)) & (~torch.isnan(y))
        x = torch.nan_to_num(x, 0.0)
        y = torch.nan_to_num(y, 0.0)

        dist = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0)
        weighted_dist = dist * self.weights.view(1, 1, -1).mean(dim=-1)

        dp = torch.full((x.size(0)+1, y.size(0)+1), float('inf'), device=self.device)
        dp[0,0] = 0
        for i in range(x.size(0)):
            for j in range(y.size(0)):
                cost = weighted_dist[i,j]
                dp[i+1,j+1] = cost + min(dp[i,j], dp[i+1,j], dp[i,j+1])
        return dp[-1,-1]

    def align(self, search_range=30):

        t_len = len(self.template)
        f_len = len(self.full)

        min_cost = float('inf')
        best_pos = 0
        for start in tqdm(range(0, f_len-t_len, search_range)):
            cost = self._weighted_dtw(self.template, self.full[start:start+t_len])
            if cost < min_cost:
                min_cost = cost
                best_pos = start

        refine_start = max(0, best_pos - search_range)
        refine_end = min(f_len-t_len, best_pos + search_range)
        for start in range(refine_start, refine_end+1):
            cost = self._weighted_dtw(self.template, self.full[start:start+t_len])
            if cost < min_cost:
                min_cost = cost
                best_pos = start
        
        return {
            'start_frame': best_pos,
            'start_time': best_pos/self.fps,
            'end_time': (best_pos + t_len)/self.fps,
            'confidence': self._calc_confidence(min_cost)
        }

    def _calc_confidence(self, cost):

        baseline = self._weighted_dtw(self.template, self.full[:len(self.template)])
        return max(0, 100 * (1 - cost/(baseline + 1e-8)))

def process_video(video_path, output_path, start_frame, end_frame):

    cap = cv2.VideoCapture(video_path)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret: break
        writer.write(frame)
    
    writer.release()
    cap.release()

def main(template_json, full_json, template_vid, full_vid, output_dir):

    with open(template_json) as f:
        template_data = json.load(f)
    with open(full_json) as f:
        full_data = json.load(f)

    processor = FeatureProcessor(max_null_ratio=0.3)
    template_feat = processor.preprocess(template_data)
    full_feat = processor.preprocess(full_data)

    aligner = VideoAligner(template_feat, full_feat, processor.feature_weights)
    result = aligner.align()

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/alignment.json", "w") as f:
        json.dump({
            "start_time": result['start_time'],
            "end_time": result['end_time'],
            "confidence": f"{result['confidence']:.1f}%"
        }, f)

    process_video(
        full_vid, 
        f"{output_dir}/aligned_segment.mp4",
        result['start_frame'],
        result['start_frame'] + len(template_feat)
    )
    
    return result


if __name__ == "__main__":
    result = main(
        template_json="{TEMPLATE_JSON_FILE}",
        full_json="{FULL_JSON_FILE}",
        template_vid="{TEMPLATE_VIDEO_FILE}",
        full_vid="{FULL_VIDEO_FILE}",
        output_dir="{OUTPUT_DIRECTORY}"
    )
