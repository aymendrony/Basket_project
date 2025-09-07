import time
import os
import cv2
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO

from utils.common.view_transformer import ViewTransformer
from utils.configs.basket_config import BasketBallCourtConfiguration
from utils.annotators.basket_annot import reorder_keypoints_to_labels, dessiner_terrain_reconstruit


class KeypointFilter:
    """
    Filtre temporel pour lisser les positions des keypoints.
    Utilise une fenêtre glissante avec choix entre médiane et moyenne.
    """
    
    def __init__(self, window_size=9, use_median=True):
        """
        Args:
            window_size (int): Taille de la fenêtre de lissage
            use_median (bool): True pour médiane, False pour moyenne
        """
        self.window_size = window_size
        self.use_median = use_median
        self.buffer = deque(maxlen=window_size)

    def apply(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Applique le filtre temporel aux keypoints.
        
        Args:
            keypoints: Array de shape (K, 2) avec les coordonnées (x, y)
            
        Returns:
            Array de shape (K, 2) avec les keypoints lissés
        """
        self.buffer.append(keypoints)
        arr = np.stack(list(self.buffer), axis=0)  # (t, K, 2)
        
        if self.use_median:
            filt = np.median(arr, axis=0)  # Médiane temporelle
        else:
            filt = np.mean(arr, axis=0)    # Moyenne temporelle
            
        return filt.astype(np.float32)


def process_video_with_basket_court_optimized(
    video_path,
    output_path,
    config,
    detection_interval=30,
    smoothing_window=3,
    use_median=True,
    csv_output_path=None,
    confidence_threshold=0.4,
    model_path='./models/BasketBall_good_court.pt'
):
    """
    Traite une vidéo de basket avec détection de terrain et reconstruction par homographie.
    
    Args:
        video_path (str): Chemin vers la vidéo d'entrée
        output_path (str): Chemin vers la vidéo de sortie
        config (BasketBallCourtConfiguration): Configuration du terrain de basket
        detection_interval (int): Intervalle de détection (frames)
        smoothing_window (int): Taille de la fenêtre de lissage
        use_median (bool): Utiliser la médiane pour le lissage
        csv_output_path (str, optional): Chemin du fichier CSV de sortie
        confidence_threshold (float): Seuil de confiance pour les keypoints (0.0-1.0)
        model_path (str): Chemin vers le modèle YOLO
        
    Returns:
        dict: Dictionnaire avec les statistiques du traitement
    """
    
    model_kp = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
    
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if csv_output_path is None:
        base, _ = os.path.splitext(output_path)
        csv_output_path = f"{base}_basket_court_coords.csv"
    csv_dir = os.path.dirname(csv_output_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    kf_vertices = KeypointFilter(window_size=smoothing_window, use_median=use_median)
    
    transformer      = None
    frame_count      = 0
    detection_count  = 0
    time_detection   = 0.0
    time_projection  = 0.0

    pitch_all_pts = np.array(config.vertices, dtype=np.float32)
    K = pitch_all_pts.shape[0]

    ref_kpts = sv.KeyPoints(xy=np.zeros((1, 0, 2), dtype=np.float32))

    vertex_cols = []
    for i, label in enumerate(config.labels):
        vertex_cols.extend([f"{label}_x", f"{label}_y"])

    csv_rows = []

    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
    
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % detection_interval == 0:
            start_det = time.time()
            try:
                results = model_kp(frame, verbose=False)
                if len(results) > 0 and getattr(results[0], "keypoints", None) is not None:
                    result = results[0]
                    xys = result.keypoints.xy[0].cpu().numpy()
                    confs = result.keypoints.conf[0].cpu().numpy()
                    keypoints_robo = np.column_stack([xys, confs])
                    
                    keypoints_ordered = reorder_keypoints_to_labels(keypoints_robo, config)
                    
                    valid_indices = []
                    valid_points = []
                    for i, kp in enumerate(keypoints_ordered):
                        if not np.isnan(kp[0]) and not np.isnan(kp[1]) and kp[2] >= confidence_threshold:
                            valid_indices.append(i)
                            valid_points.append(kp[:2])
                    
                    valid_points = np.array(valid_points)
                    
                    if len(valid_points) >= 4:
                        frame_ref_pts = valid_points.astype(np.float32)
                        pitch_ref_pts = pitch_all_pts[valid_indices].astype(np.float32)
                        
                        transformer = ViewTransformer(
                            source=pitch_ref_pts,
                            target=frame_ref_pts
                        )
                        
                        ref_kpts = sv.KeyPoints(xy=frame_ref_pts[np.newaxis, ...])
                        detection_count += 1
                        
                        pbar.set_postfix({
                            'detected': f'{len(valid_points)}pts',
                            'detections': detection_count
                        })
                        
            except Exception as e:
                pass
            time_detection += time.time() - start_det

        start_proj = time.time()
        annotated_frame = frame
        smooth_pts = None
        if transformer is not None:
            try:
                frame_all_pts = transformer.transform_points(points=pitch_all_pts)
                smooth_pts = kf_vertices.apply(frame_all_pts)   # (K,2)

                annotated_frame = dessiner_terrain_reconstruit(
                    frame=frame,
                    frame_all_points=smooth_pts,
                    config=config,
                    seuil_confiance=0.0
                )
                
            except Exception as e:
                pass
        time_projection += time.time() - start_proj

        out.write(annotated_frame)

        row_vals = []
        if smooth_pts is None:
            row_vals = [np.nan] * (2 * K)
        else:
            row_vals = smooth_pts.reshape(-1).tolist()

        csv_rows.append(
            {
                "frame": frame_count,
                "time_s": frame_count / fps,
                **{col: val for col, val in zip(vertex_cols, row_vals)}
            }
        )

        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(csv_rows, columns=["frame", "time_s"] + vertex_cols)
    df.to_csv(csv_output_path, index=False)

    return {
        'frame_count': frame_count,
        'detection_count': detection_count,
        'time_detection': time_detection,
        'time_projection': time_projection,
        'total_time': time_detection + time_projection,
        'csv_path': csv_output_path,
        'video_path': output_path,
        'fps': fps,
        'width': width,
        'height': height
    }


def print_processing_stats(results):
    """
    Affiche les statistiques de traitement de manière formatée.
    
    Args:
        results (dict): Résultats de process_video_with_basket_court_optimized
    """
    print(f"\nTRAITEMENT TERMINÉ")
    print(f"Vidéo: {results['video_path']}")
    print(f"CSV: {results['csv_path']}")
    print(f"Frames traités: {results['frame_count']}")
    print(f"Détections effectuées: {results['detection_count']}")
    print(f"Temps total: {results['total_time']:.2f}s")
    print(f"Dimensions: {results['width']}x{results['height']}")
    print(f"FPS: {results['fps']:.1f}")
    
    if results['frame_count'] > 0:
        print(f"Temps moyen par frame: {results['total_time']/results['frame_count']:.4f}s")
