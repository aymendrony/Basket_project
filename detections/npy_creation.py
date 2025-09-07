from rfdetr import RFDETRMedium
import supervision as sv
import numpy as np
from PIL import Image
import cv2
from detections.all_detections import all_detections
import os
import sys
from teamClassifier.teamclassifier import TeamClassifier
from tqdm import tqdm

def maintain_team_consistency(previous_team_assignments, current_team_assignments, player_crops):
    """
    Maintient la cohérence des team_id entre les frames en utilisant la similarité des features
    """
    if previous_team_assignments is None or len(previous_team_assignments) == 0:
        return current_team_assignments
    
    # Si pas assez de joueurs pour faire du clustering, on garde l'assignation actuelle
    if len(player_crops) < 2:
        return current_team_assignments
    
    try:
        temp_classifier = TeamClassifier(device='cpu', batch_size=8)
        current_features = temp_classifier.extract_features(player_crops)
        
        # Extraire les features des joueurs précédents 
        if hasattr(previous_team_assignments, 'features'):
            previous_features = previous_team_assignments['features']
            
            # Calculer la similarité entre les features actuelles et précédentes
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Trouver la meilleure correspondance pour maintenir la cohérence
            similarities = cosine_similarity(current_features, previous_features)
            
            # Créer un mapping pour maintenir la cohérence
            team_mapping = {}
            for i, current_team in enumerate(current_team_assignments):
                best_match_idx = np.argmax(similarities[i])
                previous_team = previous_team_assignments['teams'][best_match_idx]
                team_mapping[current_team] = previous_team
            
            # Appliquer le mapping
            consistent_assignments = [team_mapping.get(team, team) for team in current_team_assignments]
            
            # Stocker les features et teams pour la prochaine frame
            return {
                'teams': consistent_assignments,
                'features': current_features
            }
        else:
            # Première frame, on garde l'assignation actuelle
            return {
                'teams': current_team_assignments,
                'features': current_features
            }
    except Exception as e:
        print(f"Erreur dans maintain_team_consistency: {e}")
        return current_team_assignments

def get_detections_per_class(video_path, output_dir="detections_npy"):
    """
    Crée un fichier .npy pour chaque classe avec format: frame, x, y, w, h, confidence, team_id (pour les joueurs)
    """
    
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Vidéo: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Dictionnaire pour stocker les données par classe
    class_data = {
        'ball': [],
        'hoop': [],
        'period': [],
        'player': [],
        'ref': [],
        'shot_clock': [],
        'team_name': [],
        'team_points': [],
        'time_remaining': []
    }

    previous_team_assignments = None
    team_classifier = None
    
    frame_count = 0
    
    # Créer la barre de progression
    pbar = tqdm(total=total_frames, desc="Traitement des frames", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        detections_dict = all_detections(image, conf_ball=0.8, conf=0.75)
        
        # Traitement des joueurs avec clustering
        if 'player' in detections_dict and detections_dict['player'] is not None and len(detections_dict['player']) > 0:
            players = detections_dict['player']
            
            # Extraire les crops des joueurs
            player_crops = []
            player_boxes = []
            for i, (x1, y1, x2, y2) in enumerate(players.xyxy):
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    player_crops.append(crop)
                    player_boxes.append((x1, y1, x2, y2, players.confidence[i]))
            
            # Classification des équipes si on a au moins 2 joueurs
            if len(player_crops) >= 2:
                try:
                    if team_classifier is None:
                        team_classifier = TeamClassifier(device='cpu', batch_size=8)

                    team_classifier.fit(player_crops)
                    team_assignments = team_classifier.get_team_assignments(player_crops)
                    
                    # Maintenir la cohérence avec les frames précédentes
                    current_team_ids = []
                    for i in range(len(player_crops)):
                        if i in team_assignments['team1']:
                            current_team_ids.append(0)
                        else:
                            current_team_ids.append(1)

                    consistent_assignments = maintain_team_consistency(
                        previous_team_assignments, 
                        current_team_ids, 
                        player_crops
                    )
                    
                    if isinstance(consistent_assignments, dict):
                        team_ids = consistent_assignments['teams']
                        previous_team_assignments = consistent_assignments
                    else:
                        team_ids = consistent_assignments
                        previous_team_assignments = {
                            'teams': team_ids,
                            'features': team_classifier.extract_features(player_crops)
                        }
                    
                except Exception as e:
                    print(f"\nErreur classification équipes frame {frame_count}: {e}")
                    team_ids = [0 if i < len(player_crops)//2 else 1 for i in range(len(player_crops))]
            else:
                # Pas assez de joueurs pour le clustering
                team_ids = [0] * len(player_crops)
            
            # Ajouter les données des joueurs avec team_id
            for i, (x1, y1, x2, y2, confidence) in enumerate(player_boxes):
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                team_id = team_ids[i] if i < len(team_ids) else 0
                class_data['player'].append([frame_count, x, y, w, h, confidence, team_id])
        
        # Traitement des autres classes
        for class_name, detections in detections_dict.items():
            if class_name == 'player':
                continue  
                
            if detections is not None and len(detections) > 0:
                best_idx = np.argmax(detections.confidence)
                
                # Convertir xyxy en x,y,w,h
                x1, y1, x2, y2 = detections.xyxy[best_idx]
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                confidence = detections.confidence[best_idx]
                
                class_data[class_name].append([frame_count, x, y, w, h, confidence])

        # Mettre à jour la barre de progression
        pbar.update(1)
        pbar.set_postfix({
            'Frame': frame_count,
            'Joueurs': len(player_crops) if 'player' in detections_dict and detections_dict['player'] is not None else 0
        })

    cap.release()
    pbar.close()
    
    print("\nSauvegarde des fichiers .npy...")
    
    # Sauvegarder chaque classe dans son propre fichier .npy
    for class_name, data in class_data.items():
        if data:  
            data_array = np.array(data, dtype=np.float32)
            
            output_path = os.path.join(output_dir, f"{class_name}_detections.npy")
            np.save(output_path, data_array)
            
            print(f"Classe '{class_name}': {len(data)} détections sauvegardées dans {output_path}")
        else:
            print(f"Classe '{class_name}': Aucune détection trouvée")
    
    return class_data

def load_class_detections(class_name, output_dir="detections_npy"):
    file_path = os.path.join(output_dir, f"{class_name}_detections.npy")
    
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        return None

def get_detections_for_frame(class_name, frame_number, output_dir="detections_npy"):
    """
    Récupère les détections d'une classe pour une frame spécifique
    """
    data = load_class_detections(class_name, output_dir)
    if data is not None:
        frame_detections = data[data[:, 0] == frame_number]
        return frame_detections
    return None

get_detections_per_class("./videos_input/video_test_2.mp4")