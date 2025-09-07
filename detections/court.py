

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List, Optional
import supervision as sv


class CourtDetector:
    """Détecteur de terrain de basket avec keypoints et lignes."""
    
    def __init__(self, model_path: str):
        """
        Initialise le détecteur de terrain.
        
        Args:
            model_path: Chemin vers le modèle YOLO pour la détection de terrain
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.5
        
    def detect_court_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """
        Détecte les keypoints du terrain de basket.
        
        Args:
            frame: Image d'entrée
            
        Returns:
            Array des keypoints détectés avec coordonnées et confiance
        """
        results = self.model(frame, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            # Extraction des keypoints si disponibles
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()  # Premier objet détecté
                confidences = result.keypoints.conf[0].cpu().numpy()
                
                # Combiner coordonnées et confiances
                court_keypoints = np.column_stack([keypoints, confidences])
                return court_keypoints
        
        return np.array([])
    
    def detect_court_lines(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Détecte les lignes du terrain using edge detection.
        
        Args:
            frame: Image d'entrée
            
        Returns:
            Liste des lignes détectées comme paires de points
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        court_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                court_lines.append(((x1, y1), (x2, y2)))
        
        return court_lines
    
    def filter_court_region(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Extrait la région du terrain basée sur les keypoints.
        
        Args:
            frame: Image d'entrée
            keypoints: Keypoints du terrain
            
        Returns:
            Masque de la région du terrain
        """
        if len(keypoints) < 4:
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        # Prendre les keypoints avec confiance > seuil
        valid_points = keypoints[keypoints[:, 2] > self.confidence_threshold]
        
        if len(valid_points) < 4:
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        # Créer un masque polygonal
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = valid_points[:, :2].astype(np.int32)
        
        # Convex hull pour créer un polygone
        hull = cv2.convexHull(points)
        cv2.fillPoly(mask, [hull], 255)
        
        return mask
    
    def get_court_corners(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extrait les 4 coins principaux du terrain.
        
        Args:
            keypoints: Tous les keypoints détectés
            
        Returns:
            Array des 4 coins du terrain [top-left, top-right, bottom-right, bottom-left]
        """
        if len(keypoints) < 4:
            return np.array([])
        
        valid_points = keypoints[keypoints[:, 2] > self.confidence_threshold][:, :2]
        
        if len(valid_points) < 4:
            return np.array([])
        
        tl = valid_points[np.argmin(valid_points[:, 0] + valid_points[:, 1])]
        tr = valid_points[np.argmax(valid_points[:, 0] - valid_points[:, 1])]
        br = valid_points[np.argmax(valid_points[:, 0] + valid_points[:, 1])]
        bl = valid_points[np.argmin(valid_points[:, 0] - valid_points[:, 1])]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)


def adapt_keypoints_to_frame(keypoints: np.ndarray, target_count: int = 43) -> np.ndarray:
    """
    Adapte les keypoints détectés au format attendu par la configuration.
    
    Args:
        keypoints: Keypoints détectés
        target_count: Nombre de keypoints attendus
        
    Returns:
        Array adapté avec padding si nécessaire
    """
    if len(keypoints) == 0:
        return np.zeros((target_count, 2))
    
    if len(keypoints) < target_count:
        padded = np.zeros((target_count, 2))
        padded[:len(keypoints)] = keypoints[:, :2]  
        return padded
    
    sorted_indices = np.argsort(keypoints[:, 2])[::-1]  
    return keypoints[sorted_indices[:target_count], :2]


def estimate_court_homography(source_points: np.ndarray, target_points: np.ndarray) -> Optional[np.ndarray]:
    """
    Estime la matrice d'homographie entre les points détectés et la configuration de référence.
    
    Args:
        source_points: Points détectés dans l'image
        target_points: Points de référence du terrain
        
    Returns:
        Matrice d'homographie 3x3 ou None si échec
    """
    if len(source_points) < 4 or len(target_points) < 4:
        return None
    
    try:
        homography, _ = cv2.findHomography(
            source_points.astype(np.float32),
            target_points.astype(np.float32),
            cv2.RANSAC,
            5.0
        )
        return homography
    except Exception as e:
        print(f"Erreur lors du calcul de l'homographie: {e}")
        return None


def validate_court_detection(keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
    """
    Valide que la détection du terrain est cohérente.
    
    Args:
        keypoints: Keypoints détectés
        frame_shape: Dimensions de l'image (hauteur, largeur)
        
    Returns:
        True si la détection est valide
    """
    if len(keypoints) < 4:
        return False
    
    h, w = frame_shape
    valid_points = keypoints[keypoints[:, 2] > 0.3]  
    
    in_bounds = (
        (valid_points[:, 0] >= 0) & (valid_points[:, 0] < w) &
        (valid_points[:, 1] >= 0) & (valid_points[:, 1] < h)
    )
    
    return np.sum(in_bounds) >= 4


def draw_court_overlay(frame: np.ndarray, keypoints: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Dessine une superposition du terrain détecté.
    
    Args:
        frame: Image de base
        keypoints: Keypoints du terrain
        color: Couleur de la superposition (BGR)
        
    Returns:
        Frame avec superposition
    """
    overlay = frame.copy()
    
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.4:
            cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
            cv2.putText(overlay, str(i), (int(x)+5, int(y)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return overlay
