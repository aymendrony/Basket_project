import supervision as sv 
from rfdetr import RFDETRMedium
import cv2
from tqdm import tqdm
import numpy as np
from typing import Dict, List
from IPython.core.display import display, HTML
from PIL import Image
import base64
from io import BytesIO


RF_DETR_CLASSES = [
    "Rien", "Ball", "Hoop", "Period", "Player", "Ref", 
    "Shot Clock", "Team Name", "Team Points", "Time Remaining"
]

# Mapping des classes
PLAYER_CLASS_NAMES = ["Player"]
PLAYER_CLASS_IDS = [RF_DETR_CLASSES.index(name) for name in PLAYER_CLASS_NAMES if name in RF_DETR_CLASSES]
REFEREE_CLASS_NAMES = ["Ref"]
REFEREE_CLASS_IDS = [RF_DETR_CLASSES.index(name) for name in REFEREE_CLASS_NAMES if name in RF_DETR_CLASSES]

def get_detections(frame, rfdetr_model, threshold=0.5):
    """
    Obtient les détections RF-DETR pour une frame donnée.
    
    Args:
        frame: Frame OpenCV (BGR)
        rfdetr_model: Modèle RF-DETR initialisé
        threshold: Seuil de confiance
        
    Returns:
        sv.Detections: Détections avec xyxy, class_id, confidence
    """
    # Conversion BGR → RGB → PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Prédiction RF-DETR
    result = rfdetr_model.predict(pil_image, threshold=threshold)
    
    if len(result) == 0:
        return sv.Detections.empty()
    
    # Conversion en format supervision
    xyxy = result.xyxy
    class_ids = result.class_id
    confidences = result.confidence
    
    detections = sv.Detections(
        xyxy=xyxy,
        class_id=class_ids,
        confidence=confidences
    )
    
    return detections

def get_players_detections(frame, rfdetr_model, player_class_ids=None, threshold=0.5):
    """
    Obtient les détections des joueurs avec RF-DETR.
    
    Args:
        frame: Frame OpenCV
        rfdetr_model: Modèle RF-DETR initialisé
        player_class_ids: IDs des classes de joueurs (par défaut: Player + Ref)
        threshold: Seuil de confiance
        
    Returns:
        sv.Detections: Détections des joueurs uniquement
    """
    if player_class_ids is None:
        player_class_ids = PLAYER_CLASS_IDS
    
    detections = get_detections(frame, rfdetr_model, threshold)
    
    if len(detections) == 0:
        return sv.Detections.empty()
    
    # Filtrage des joueurs
    player_indices = np.where(np.isin(detections.class_id, player_class_ids))[0]
    
    if len(player_indices) == 0:
        return sv.Detections.empty()
    
    player_detections = sv.Detections(
        xyxy=detections.xyxy[player_indices],
        class_id=detections.class_id[player_indices],
        confidence=detections.confidence[player_indices]
    )
    
    return player_detections

def get_player_detections(frame, rfdetr_model, player_class_id=None, threshold=0.5):
    """
    Obtient les détections d'un type spécifique de joueur.
    
    Args:
        frame: Frame OpenCV
        rfdetr_model: Modèle RF-DETR initialisé
        player_class_id: ID de la classe spécifique (par défaut: Player)
        threshold: Seuil de confiance
        
    Returns:
        sv.Detections: Détections du type de joueur spécifié
    """
    if player_class_id is None:
        player_class_id = RF_DETR_CLASSES.index("Player")
    
    detections = get_detections(frame, rfdetr_model, threshold)
    
    if len(detections) == 0:
        return sv.Detections.empty()
    
    # Filtrage par classe spécifique
    player_indices = np.where(detections.class_id == player_class_id)[0]
    
    if len(player_indices) == 0:
        return sv.Detections.empty()
    
    player_detections = sv.Detections(
        xyxy=detections.xyxy[player_indices],
        class_id=detections.class_id[player_indices],
        confidence=detections.confidence[player_indices]
    )
    
    return player_detections

def get_referees_detections(frame, rfdetr_model, referee_class_id=None, threshold=0.5):
    """
    Obtient les détections des arbitres avec RF-DETR.
    
    Args:
        frame: Frame OpenCV
        rfdetr_model: Modèle RF-DETR initialisé
        referee_class_id: ID de la classe arbitre (par défaut: Ref)
        threshold: Seuil de confiance
        
    Returns:
        sv.Detections: Détections des arbitres uniquement
    """
    if referee_class_id is None:
        referee_class_id = RF_DETR_CLASSES.index("Ref")
    
    detections = get_detections(frame, rfdetr_model, threshold)
    
    if len(detections) == 0:
        return sv.Detections.empty()
    
    # Filtrage des arbitres
    referee_indices = np.where(detections.class_id == referee_class_id)[0]
    
    if len(referee_indices) == 0:
        return sv.Detections.empty()
    
    referee_detections = sv.Detections(
        xyxy=detections.xyxy[referee_indices],
        class_id=detections.class_id[referee_indices],
        confidence=detections.confidence[referee_indices]
    )
    
    return referee_detections

def get_labels(frame, rfdetr_model, threshold=0.5):
    """
    Obtient les labels formatés pour toutes les détections RF-DETR.
    
    Args:
        frame: Frame OpenCV
        rfdetr_model: Modèle RF-DETR initialisé
        threshold: Seuil de confiance
        
    Returns:
        List[str]: Liste des labels formatés
    """
    detections = get_detections(frame, rfdetr_model, threshold)
    
    if len(detections) == 0:
        return []
    
    labels = []
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        class_name = RF_DETR_CLASSES[class_id] if 0 <= class_id < len(RF_DETR_CLASSES) else f"class_{class_id}"
        labels.append(f"{class_name} {confidence:.2f}")
    
    return labels

def get_players_crops(source_video_path, rfdetr_model, stride=50, player_class_ids=None, threshold=0.3):
    """
    Collecte les crops des joueurs depuis une vidéo avec RF-DETR.
    
    Args:
        source_video_path: Chemin vers la vidéo source
        rfdetr_model: Modèle RF-DETR initialisé
        stride: Pas entre les frames analysées
        player_class_ids: IDs des classes de joueurs
        threshold: Seuil de confiance
        
    Returns:
        List[np.ndarray]: Liste des crops des joueurs
    """
    if player_class_ids is None:
        player_class_ids = PLAYER_CLASS_IDS
    
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, 
        stride=stride
    )
    
    crops = []
    
    for frame in tqdm(frame_generator, desc='Collecting player crops with RF-DETR'):
        # Détection des joueurs
        player_detections = get_players_detections(
            frame, rfdetr_model, player_class_ids, threshold
        )
        
        if len(player_detections) == 0:
            continue
        
        # Extraction des crops
        frame_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
        crops.extend(frame_crops)
    
    return crops

def get_detections_with_tracking(frame, rfdetr_model, tracker, threshold=0.5):
    """
    Obtient les détections RF-DETR avec tracking ByteTrack.
    
    Args:
        frame: Frame OpenCV
        rfdetr_model: Modèle RF-DETR initialisé
        tracker: Instance ByteTrack
        threshold: Seuil de confiance
        
    Returns:
        sv.Detections: Détections avec IDs de tracking
    """
    detections = get_detections(frame, rfdetr_model, threshold)
    
    if len(detections) == 0:
        return detections
    
    # Application du tracking
    tracked_detections = tracker.update_with_detections(detections)
    
    return tracked_detections

def get_players_detections_with_tracking(frame, rfdetr_model, tracker, player_class_ids=None, threshold=0.5):
    """
    Obtient les détections des joueurs avec tracking.
    
    Args:
        frame: Frame OpenCV
        rfdetr_model: Modèle RF-DETR initialisé
        tracker: Instance ByteTrack
        player_class_ids: IDs des classes de joueurs
        threshold: Seuil de confiance
        
    Returns:
        sv.Detections: Détections des joueurs avec IDs de tracking
    """
    if player_class_ids is None:
        player_class_ids = PLAYER_CLASS_IDS
    
    # Détection + tracking
    tracked_detections = get_detections_with_tracking(frame, rfdetr_model, tracker, threshold)
    
    if len(tracked_detections) == 0:
        return sv.Detections.empty()
    
    # Filtrage des joueurs
    player_indices = np.where(np.isin(tracked_detections.class_id, player_class_ids))[0]
    
    if len(player_indices) == 0:
        return sv.Detections.empty()
    
    player_detections = sv.Detections(
        xyxy=tracked_detections.xyxy[player_indices],
        class_id=tracked_detections.class_id[player_indices],
        confidence=tracked_detections.confidence[player_indices],
        tracker_id=tracked_detections.tracker_id[player_indices] if hasattr(tracked_detections, 'tracker_id') else None
    )
    
    return player_detections


# Fonction d'initialisation du modèle RF-DETR
def initialize_rfdetr_model(model_path, device='auto'):
    """
    Initialise le modèle RF-DETR.
    
    Args:
        model_path: Chemin vers le fichier .pth du modèle
        device: 'cpu', 'cuda', ou 'auto'
        
    Returns:
        RFDETRMedium: Modèle initialisé
    """
    try:
        model = RFDETRMedium(pretrain_weights=model_path)
        
        # Optimisation pour l'inférence
        try:
            model.optimize_for_inference()
            print(" Modèle RF-DETR optimisé pour l'inférence")
        except Exception as e:
            print(f"  Optimisation échouée: {e}")
        
        # Gestion du device
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            model.to('cuda')
            print(" Modèle déplacé vers GPU")
        else:
            print("  Modèle sur CPU")
        
        return model
        
    except Exception as e:
        print(f" Erreur lors de l'initialisation du modèle RF-DETR: {e}")
        raise
