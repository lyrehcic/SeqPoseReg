import json
import math
from collections import OrderedDict
from typing import Dict, Tuple

KEYPOINT_CONFIG = OrderedDict([
    ('shoulder_left', 5),
    ('shoulder_right', 6),
    ('elbow_left', 7),
    ('elbow_right', 8),
    ('wrist_left', 9),
    ('wrist_right', 10),
    ('hip_left', 11),
    ('hip_right', 12),
    ('knee_left', 13),
    ('knee_right', 14),
    ('ankle_left', 15),
    ('ankle_right', 16),
    ('center', None)  
])

ANGLE_CONFIG = {
    'left_elbow': ('elbow_left', 'shoulder_left', 'wrist_left'),
    'right_elbow': ('elbow_right', 'shoulder_right', 'wrist_right'),
    'left_knee': ('knee_left', 'hip_left', 'ankle_left'),
    'right_knee': ('knee_right', 'hip_right', 'ankle_right'),
    'left_shoulder': ('shoulder_left', 'hip_left', 'elbow_left'),
    'right_shoulder': ('shoulder_right', 'hip_right', 'elbow_right'),
    'left_hip': ('hip_left', 'hip_right', 'knee_left'),
    'right_hip': ('hip_right', 'hip_left', 'knee_right'),

    'shoulderL_center_wristL': ('center', 'shoulder_left', 'wrist_left'),
    'shoulderR_center_wristR': ('center', 'shoulder_right', 'wrist_right'),
    'hipL_center_ankleL': ('center', 'hip_left', 'ankle_left'),
    'hipR_center_ankleR': ('center', 'hip_right', 'ankle_right'),
    'wristL_center_ankleL': ('center', 'wrist_left', 'ankle_left'),
    'wristR_center_ankleR': ('center', 'wrist_right', 'ankle_right')
}

def calculate_angle(a, b, c, min_confidence=0.2):

    if None in [a, b, c] or any(val is None for point in [a, b, c] for val in point.values()):
        return None
    
    ax, ay, a_conf = a['x'], a['y'], a['confidence']
    bx, by, b_conf = b['x'], b['y'], b['confidence']
    cx, cy, c_conf = c['x'], c['y'], c['confidence']
    
    if any(conf < min_confidence for conf in [a_conf, b_conf, c_conf]):
        return None
    
    vec_ba = (ax - bx, ay - by)
    vec_bc = (cx - bx, cy - by)
    
    dot_product = vec_ba[0] * vec_bc[0] + vec_ba[1] * vec_bc[1]
    mod_ba = math.sqrt(vec_ba[0]**2 + vec_ba[1]**2)
    mod_bc = math.sqrt(vec_bc[0]**2 + vec_bc[1]**2)
    
    if mod_ba * mod_bc < 1e-6:
        return None
    
    cosine = max(min(dot_product / (mod_ba * mod_bc), 1.0), -1.0)
    angle = math.degrees(math.acos(cosine))
    return round(angle, 1)

def process_video_angles(input_file, output_file, fps=30):

    processed_frames = []
    
    try:
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"error: input {input_file} ")
        return
    except json.JSONDecodeError:
        print(f"error: input {input_file} ")
        return
    
    for frame_idx, frame in enumerate(raw_data):
        if not frame:
            continue
        
        frame_data = {
            "timestamp": round(frame_idx / fps, 3),
            "keypoints": {},
            "angles": {},
            "distances": {}
        }
        
        try:
            kps = frame[0]['keypoints']
            if len(kps) < 17:
                continue
        except:
            continue

        for kp_name, idx in KEYPOINT_CONFIG.items():
            if idx is None:
                continue
            x, y, conf = kps[idx]
            frame_data["keypoints"][kp_name] = {
                "x": x if conf > 0.2 else None,
                "y": y if conf > 0.2 else None,
                "confidence": conf
            }
        
        def get_point(name):
            return frame_data["keypoints"].get(name, {})
        
        shoulders = [get_point('shoulder_left'), get_point('shoulder_right')]
        hips = [get_point('hip_left'), get_point('hip_right')]
        
        valid_points = all(
            p['x'] is not None and p['y'] is not None and p['confidence'] >= 0.2
            for p in shoulders + hips
        )
        
        if valid_points:
            sum_x = sum(p['x'] for p in shoulders + hips)
            sum_y = sum(p['y'] for p in shoulders + hips)
            min_conf = min(p['confidence'] for p in shoulders + hips)
            
            frame_data["keypoints"]['center'] = {
                "x": sum_x / 4,
                "y": sum_y / 4,
                "confidence": min_conf
            }
        else:
            frame_data["keypoints"]['center'] = {
                "x": None,
                "y": None,
                "confidence": 0.0
            }
        
        for angle_name, (center, start, end) in ANGLE_CONFIG.items():
            center_pt = frame_data["keypoints"].get(center)
            start_pt = frame_data["keypoints"].get(start)
            end_pt = frame_data["keypoints"].get(end)
            
            if all([center_pt, start_pt, end_pt]):
                angle = calculate_angle(start_pt, center_pt, end_pt)
                frame_data["angles"][angle_name] = angle
            else:
                frame_data["angles"][angle_name] = None
        
        center_point = frame_data["keypoints"]["center"]
        for kp_name in KEYPOINT_CONFIG:
            if kp_name == "center":
                continue
            
            kp = frame_data["keypoints"][kp_name]
            distance = None
            
            if (kp["x"] is not None and kp["y"] is not None and 
                center_point["x"] is not None and center_point["y"] is not None and
                kp["confidence"] >= 0.2 and center_point["confidence"] >= 0.2):
                
                dx = kp["x"] - center_point["x"]
                dy = kp["y"] - center_point["y"]
                distance = round(math.hypot(dx, dy), 1)
                
            frame_data["distances"][f"{kp_name}_to_center"] = distance
        
        processed_frames.append(frame_data)
    

    output_data = []
    for frame in processed_frames:
        output_entry = {
            "timestamp": frame["timestamp"],
            "angles": frame["angles"],
            "distances": frame["distances"]
        }
        output_data.append(output_entry)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"results is saved in {output_file}")
    except IOError:
        print(f"error:results generated in {output_file} is failure")

process_video_angles(r"/xx.json", r"/json_angle_dis_xx.json")



