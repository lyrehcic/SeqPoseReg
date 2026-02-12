import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn.functional as F
import json

def preprocess_features(data, max_null_ratio=0.3):

    processed = []
    for seq in data:

        features = []
        for frame in seq:
            frame_features = []

            for joint in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 
                          'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                          'shoulderL_center_wristL', 'shoulderR_center_wristR', 
                          'hipL_center_ankleL', 'hipR_center_ankleR', 
                          'wristL_center_ankleL', 'wristR_center_ankleR']:
                value = frame['angles'].get(joint, np.nan)
                if not isinstance(value, (int, float)):
                    value = np.nan  
                frame_features.append(value)

            for dist in ['shoulder_left_to_center', 'shoulder_right_to_center', 
                         'elbow_left_to_center', 'elbow_right_to_center', 
                         'wrist_left_to_center', 'wrist_right_to_center', 
                         'hip_left_to_center', 'hip_right_to_center', 
                         'knee_left_to_center', 'knee_right_to_center', 
                         'ankle_left_to_center', 'ankle_right_to_center']:
                value = frame['distances'].get(dist, np.nan)
                if not isinstance(value, (int, float)):
                    value = np.nan  
                frame_features.append(value)
            features.append(frame_features)
        
        features = np.array(features).T  # (num_features, seq_len)

        valid_features = []
        for dim in features:

            if np.isnan(dim).mean() > max_null_ratio:
                continue
                

            mask = ~np.isnan(dim)
            indices = np.where(mask)[0]
            if len(indices) < 2:
                continue
                
            f = interp1d(indices, dim[mask], kind='linear', 
                        fill_value="extrapolate")
            dim_filled = f(np.arange(len(dim)))
            

            derivatives = np.zeros_like(dim_filled)
            derivatives[1:] = np.diff(dim_filled)
            derivatives[0] = derivatives[1]
            

            valid_features.append(np.stack([dim_filled, derivatives], axis=1))
        
        if valid_features:
            processed.append(np.concatenate(valid_features, axis=1))
    
    return processed

def soft_dtw_cost_matrix(x, y, gamma=1.0):
    n = x.shape[0]
    m = y.shape[0]
    dim = x.shape[1]
    
    D = torch.zeros((n+1, m+1), device=x.device)
    D[0, 1:] = torch.inf
    D[1:, 0] = torch.inf

    XX = torch.sum(x**2, dim=1)
    YY = torch.sum(y**2, dim=1)
    XY = torch.matmul(x, y.T)
    dist = XX[:, None] + YY[None, :] - 2*XY
    dist = torch.sqrt(torch.clamp(dist, min=0))

    for i in range(n):
        for j in range(m):
            cost = dist[i, j]
            min_cost = torch.logsumexp(torch.stack([
                D[i, j],
                D[i, j+1],
                D[i+1, j]
            ]) / -gamma, dim=0) * -gamma
            D[i+1, j+1] = cost + min_cost
    
    return D

def multi_dtw_alignment(query, reference):

    query = (query - np.mean(query, axis=0)) / (np.std(query, axis=0) + 1e-8)
    reference = (reference - np.mean(reference, axis=0)) / (np.std(reference, axis=0) + 1e-8)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(query, dtype=torch.float32, device=device)
    y = torch.tensor(reference, dtype=torch.float32, device=device)

    gamma = 0.5 
    D = soft_dtw_cost_matrix(x, y, gamma)
    
    path = []
    i, j = x.shape[0], y.shape[0]
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        costs = [
            D[i-1, j-1],
            D[i-1, j],
            D[i, j-1]
        ]
        idx = torch.argmin(torch.tensor(costs))
        if idx == 0:
            i -= 1
            j -= 1
        elif idx == 1:
            i -= 1
        else:
            j -= 1
    
    return np.array(path[::-1])

def align_sequences(seq1, seq2):

    processed = preprocess_features([seq1, seq2])
    
    min_dim = min(p.shape[1] for p in processed)
    processed = [p[:, :min_dim] for p in processed]
    
    alignment_path = multi_dtw_alignment(processed[0], processed[1])
    
    return alignment_path

def save_alignment_to_json(alignment_path, seq1, seq2, output_file):
    frame_rate1 = 30 
    frame_rate2 = 30
    alignment_result = []
    
    for idx1, idx2 in alignment_path:
        timestamp_a = idx1 / frame_rate1  
        timestamp_b = idx2 / frame_rate2  
        time_diff = abs(timestamp_a - timestamp_b)  
        confidence = 100 - (time_diff * 30) 
        
        alignment_result.append({
            "video_a": f"{timestamp_a:.3f}s",
            "video_b": f"{timestamp_b:.3f}s",
            "time_diff": f"{time_diff:.3f}s",
            "confidence": round(confidence, 1)
        })
    
    with open(output_file, "w") as f:
        json.dump(alignment_result, f, indent=4)


if __name__ == "__main__":

    with open(r"/xx2.json") as f:
        data1 = json.load(f)
    with open(r"/xx1.json") as f:
        data2 = json.load(f)
    
    alignment = align_sequences(data1, data2)
    save_alignment_to_json(alignment, data1, data2, r"/xx.json")
    print("Alignment result saved to alignment_result.json")
