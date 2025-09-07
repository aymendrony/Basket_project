from rfdetr import RFDETRMedium
import supervision as sv
import numpy as np
from PIL import Image
import cv2


_ball_model = None
_hoop_model = None
_players_model = None
_ref_model = None
_period_model = None
_shot_clock_model = None
_team_name_model = None
_team_points_model = None
_time_remaining_model = None

def get_optimized_ball_model():
    global _ball_model
    if _ball_model is None:
        _ball_model = RFDETRMedium(pretrain_weights="./models/ball_model.pth")
        _ball_model.optimize_for_inference()
    return _ball_model

def get_optimized_hoop_model():
    global _hoop_model
    if _hoop_model is None:
        _hoop_model = RFDETRMedium(pretrain_weights="./models/general_model.pth")
        _hoop_model.optimize_for_inference()
    return _hoop_model

def get_optimized_players_model():
    global _players_model
    if _players_model is None:
        _players_model = RFDETRMedium(pretrain_weights="./models/general_model.pth")
        _players_model.optimize_for_inference()
    return _players_model

def get_optimized_ref_model():
    global _ref_model
    if _ref_model is None:
        _ref_model = RFDETRMedium(pretrain_weights="./models/general_model.pth")
        _ref_model.optimize_for_inference()
    return _ref_model

def get_optimized_period_model():
    global _period_model
    if _period_model is None:
        _period_model = RFDETRMedium(pretrain_weights="./models/general_model.pth")
        _period_model.optimize_for_inference()
    return _period_model

def get_optimized_shot_clock_model():
    global _shot_clock_model
    if _shot_clock_model is None:
        _shot_clock_model = RFDETRMedium(pretrain_weights="./models/general_model.pth")
        _shot_clock_model.optimize_for_inference()
    return _shot_clock_model

def get_optimized_team_name_model():
    global _team_name_model
    if _team_name_model is None:
        _team_name_model = RFDETRMedium(pretrain_weights="./models/general_model.pth")
        _team_name_model.optimize_for_inference()
    return _team_name_model

def get_optimized_team_points_model():
    global _team_points_model
    if _team_points_model is None:
        _team_points_model = RFDETRMedium(pretrain_weights="./models/general_model.pth")
        _team_points_model.optimize_for_inference()
    return _team_points_model

def get_optimized_time_remaining_model():
    global _time_remaining_model
    if _time_remaining_model is None:
        _time_remaining_model = RFDETRMedium(pretrain_weights="./models/general_model.pth")
        _time_remaining_model.optimize_for_inference()
    return _time_remaining_model

def ball_detection(frame, conf=0.8):
    model = get_optimized_ball_model()
    detections = model.predict(frame, threshold=conf)
    return detections

def hoop_detection(frame, conf=0.68):
    model = get_optimized_hoop_model()
    detections = model.predict(frame, threshold=conf)

    CUSTOM_CLASSES = ["Rien", "Ball", "Hoop", "Period", "Player", "Ref",
                 "Shot Clock", "Team Name", "Team Points", "Time Remaining"]

    TARGET_CLASSES = [2] 
    TARGET_CLASS_NAMES = ["Hoop"]

    if len(detections) > 0:
        mask = np.isin(detections.class_id, TARGET_CLASSES)
        
        if np.any(mask):  
            filtered_detections = sv.Detections(
                xyxy=detections.xyxy[mask],
                confidence=detections.confidence[mask],
                class_id=detections.class_id[mask]
            )
            return filtered_detections
    
    # Return None if no detections found
    return None

def players_detection(frame, conf=0.68):
    model = get_optimized_players_model()
    detections = model.predict(frame, threshold=conf)

    CUSTOM_CLASSES = ["Rien", "Ball", "Hoop", "Period", "Player", "Ref",
                 "Shot Clock", "Team Name", "Team Points", "Time Remaining"]

    TARGET_CLASSES = [4] 
    TARGET_CLASS_NAMES = ["player"]

    if len(detections) > 0:
        mask = np.isin(detections.class_id, TARGET_CLASSES)
        
        if np.any(mask):  
            filtered_detections = sv.Detections(
                xyxy=detections.xyxy[mask],
                confidence=detections.confidence[mask],
                class_id=detections.class_id[mask]
            )
            return filtered_detections
    
    # Return None if no detections found
    return None

def ref_detection(frame, conf=0.68):
    model = get_optimized_ref_model()
    detections = model.predict(frame, threshold=conf)

    CUSTOM_CLASSES = ["Rien", "Ball", "Hoop", "Period", "Player", "Ref",
                 "Shot Clock", "Team Name", "Team Points", "Time Remaining"]

    TARGET_CLASSES = [5] 
    TARGET_CLASS_NAMES = ["ref"]

    if len(detections) > 0:
        mask = np.isin(detections.class_id, TARGET_CLASSES)
        
        if np.any(mask):  
            filtered_detections = sv.Detections(
                xyxy=detections.xyxy[mask],
                confidence=detections.confidence[mask],
                class_id=detections.class_id[mask]
            )
            return filtered_detections
    
    # Return None if no detections found
    return None

def period_detection(frame, conf=0.68):
    model = get_optimized_period_model()
    detections = model.predict(frame, threshold=conf)

    CUSTOM_CLASSES = ["Rien", "Ball", "Hoop", "Period", "Player", "Ref",
                 "Shot Clock", "Team Name", "Team Points", "Time Remaining"]

    TARGET_CLASSES = [3] 
    TARGET_CLASS_NAMES = ["period"]

    if len(detections) > 0:
        mask = np.isin(detections.class_id, TARGET_CLASSES)
        
        if np.any(mask):  
            filtered_detections = sv.Detections(
                xyxy=detections.xyxy[mask],
                confidence=detections.confidence[mask],
                class_id=detections.class_id[mask]
            )
            return filtered_detections
    
    # Return None if no detections found
    return None

def shot_clock_detection(frame, conf=0.68):
    model = get_optimized_shot_clock_model()
    detections = model.predict(frame, threshold=conf)

    CUSTOM_CLASSES = ["Rien", "Ball", "Hoop", "Period", "Player", "Ref",
                 "Shot Clock", "Team Name", "Team Points", "Time Remaining"]

    TARGET_CLASSES = [6] 
    TARGET_CLASS_NAMES = ["shot clock"]

    if len(detections) > 0:
        mask = np.isin(detections.class_id, TARGET_CLASSES)
        
        if np.any(mask):  
            filtered_detections = sv.Detections(
                xyxy=detections.xyxy[mask],
                confidence=detections.confidence[mask],
                class_id=detections.class_id[mask]
            )
            return filtered_detections
    
    # Return None if no detections found
    return None

def team_name_detection(frame, conf=0.68):
    model = get_optimized_team_name_model()
    detections = model.predict(frame, threshold=conf)
    
    CUSTOM_CLASSES = ["Rien", "Ball", "Hoop", "Period", "Player", "Ref",
                 "Shot Clock", "Team Name", "Team Points", "Time Remaining"]

    TARGET_CLASSES = [7] 
    TARGET_CLASS_NAMES = ["team name"]
    
    if len(detections) > 0 and len(detections.class_id) > 0:
        mask = np.isin(detections.class_id, TARGET_CLASSES)
        
        if np.any(mask):  
            filtered_detections = sv.Detections(
                xyxy=detections.xyxy[mask],
                confidence=detections.confidence[mask],
                class_id=detections.class_id[mask]
            )
            return filtered_detections
    
    # Return None if no detections found
    return None

def team_points_detection(frame, conf=0.68):
    model = get_optimized_team_points_model()
    detections = model.predict(frame, threshold=conf)

    CUSTOM_CLASSES = ["Rien", "Ball", "Hoop", "Period", "Player", "Ref",
                 "Shot Clock", "Team Name", "Team Points", "Time Remaining"]

    TARGET_CLASSES = [8] 
    TARGET_CLASS_NAMES = ["team points"]

    if len(detections) > 0:
        mask = np.isin(detections.class_id, TARGET_CLASSES)
        
        if np.any(mask):  
            filtered_detections = sv.Detections(
                xyxy=detections.xyxy[mask],
                confidence=detections.confidence[mask],
                class_id=detections.class_id[mask]
            )
            return filtered_detections
    
    # Return None if no detections found
    return None

def time_remaining_detection(frame, conf=0.68):
    model = get_optimized_time_remaining_model()
    detections = model.predict(frame, threshold=conf)

    CUSTOM_CLASSES = ["Rien", "Ball", "Hoop", "Period", "Player", "Ref",
                 "Shot Clock", "Team Name", "Team Points", "Time Remaining"]

    TARGET_CLASSES = [9] 
    TARGET_CLASS_NAMES = ["time remaining"]

    if len(detections) > 0:
        mask = np.isin(detections.class_id, TARGET_CLASSES)
        
        if np.any(mask):  
            filtered_detections = sv.Detections(
                xyxy=detections.xyxy[mask],
                confidence=detections.confidence[mask],
                class_id=detections.class_id[mask]
            )
            return filtered_detections
    
    # Return None if no detections found
    return None

def all_detections(frame, conf_ball, conf):
    ball_detections = ball_detection(frame, conf_ball)
    hoop_detections = hoop_detection(frame, conf)
    players_detections = players_detection(frame, conf)
    ref_detections = ref_detection(frame, conf)
    period_detections = period_detection(frame, conf)
    shot_clock_detections = shot_clock_detection(frame, conf)
    team_name_detections = team_name_detection(frame, conf)
    team_points_detections = team_points_detection(frame, conf)
    time_remaining_detections = time_remaining_detection(frame, conf)

    all_detections = {
        "ball": ball_detections,
        "hoop": hoop_detections,
        "player": players_detections,
        "ref": ref_detections,
        "period": period_detections,
        "shot_clock": shot_clock_detections,
        "team_name": team_name_detections,
        "team_points": team_points_detections,
        "time_remaining": time_remaining_detections
    }

    return all_detections