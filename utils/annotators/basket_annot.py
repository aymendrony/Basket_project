
# pip install sportypy

from typing import Optional, List
import cv2
import supervision as sv
import numpy as np
from utils.configs.basket_config import BasketBallCourtConfiguration
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sportypy.surfaces.basketball import NBACourt

CM_TO_FEET = 0.03280839895  # 1 cm = 0.0328084 ft



def annotate_box_frame(frame, detections, labels):


    box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(["#00FF04", '#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
    )
    label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(["#00FF04", '#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000')
    )


    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)
    
    return annotated_frame


def annotate_frame(frame, ball_detections, all_detections):
    
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )

    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame,
        detections=all_detections
    )

    annotated_frame = triangle_annotator.annotate(
    scene=annotated_frame,
    detections=ball_detections
    )

    return annotated_frame 


def annotate_track_frame(frame,ball_detections,all_detections, labels):
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )

    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )

    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame,
        detections=all_detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=all_detections,
        labels=labels)
    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame,
        detections=ball_detections)

    return annotated_frame


def court_annotate_frame(frame, keypoints, config, with_edges=False, confidence_mask=None, yolo_indices=None):
    """
    Annote un frame avec les keypoints du terrain de basketball.
    
    Args:
        frame: L'image à annoter
        keypoints: sv.KeyPoints object avec les keypoints détectés
        config: BasketBallCourtConfiguration object
        with_edges: Si True, dessine aussi les edges entre keypoints
        confidence_mask: Masque booléen pour filtrer les keypoints par confiance
        yolo_indices: Indices YOLO originaux des keypoints filtrés
    
    Returns:
        Frame annoté avec les keypoints et optionnellement les edges
    """
    annotated_frame = frame.copy()
    
    # Si on veut dessiner les edges et qu'on a les indices YOLO
    if with_edges and yolo_indices is not None and len(yolo_indices) > 1:
        # Créer un mapping des indices YOLO vers les positions dans le tableau filtré
        yolo_to_filtered = {yolo_idx: i for i, yolo_idx in enumerate(yolo_indices)}
        
        # Filtrer les edges valides
        valid_edges = []
        for edge_start, edge_end in config.edges:
            # Convertir de 1-based à 0-based (les edges utilisent des numéros de labels)
            # Trouver les indices YOLO correspondants aux labels
            start_label = f"{edge_start:02d}"
            end_label = f"{edge_end:02d}"
            
            try:
                start_yolo_idx = config.labels.index(start_label)
                end_yolo_idx = config.labels.index(end_label)
                
                # Vérifier si ces indices YOLO sont dans nos keypoints filtrés
                if start_yolo_idx in yolo_to_filtered and end_yolo_idx in yolo_to_filtered:
                    filtered_start = yolo_to_filtered[start_yolo_idx]
                    filtered_end = yolo_to_filtered[end_yolo_idx]
                    valid_edges.append((filtered_start, filtered_end))
            except ValueError:
                # Label non trouvé, ignorer cet edge
                continue
        
        # Dessiner les edges si on en a
        if valid_edges:
            edge_annotator = sv.EdgeAnnotator(
                color=sv.Color.from_hex('#00BFFF'),
                thickness=3,
                edges=valid_edges
            )
            annotated_frame = edge_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints
            )
    
    # Dessiner les keypoints par-dessus
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#FF1493'),
        radius=8
    )
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=keypoints
    )
    
    # Optionnel: ajouter les IDs des keypoints
    if yolo_indices is not None:
        keypoints_xy = keypoints.xy[0] 
        for i, (yolo_idx, point) in enumerate(zip(yolo_indices, keypoints_xy)):
            cv2.putText(
                annotated_frame,
                f"{yolo_idx}",
                (int(point[0]) + 15, int(point[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                annotated_frame,
                f"{yolo_idx}",
                (int(point[0]) + 15, int(point[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  
                1,
                cv2.LINE_AA
            )
    
    return annotated_frame


def draw_court(
    config: BasketBallCourtConfiguration,
    background_color: sv.Color = sv.Color(210, 115, 50),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,          
    line_thickness: int = 4,    
    point_radius: int = 8,      
    scale: float = 0.1          
) -> np.ndarray:
    """
    Returns:
        np.ndarray: Image of the basketball court.
    """
    import cv2
    import numpy as np
    

    return draw_court_orange_from_config(
        config=config,
        width=int(1280 * scale * 10), 
        height=int(720 * scale * 10),   
        padding=int(padding * scale),  
        line_thickness=line_thickness
    )


def draw_court_orange_from_config(config: BasketBallCourtConfiguration,
                                  width=1280,
                                  height=720,
                                  padding=60,
                                  line_thickness=4) -> np.ndarray:
    import cv2, numpy as np

    L_cm, W_cm = float(config.length), float(config.width)

    H, W = height, width
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # zone jouable
    x0, y0 = padding, padding
    x1, y1 = W - padding, H - padding
    play_w, play_h = x1-x0, y1-y0

    s = min(play_w/L_cm, play_h/W_cm)
    new_w, new_h = int(round(L_cm*s)), int(round(W_cm*s))
    x0 = (W - new_w)//2
    y0 = (H - new_h)//2

    def cm_to_px(x_cm, y_cm):
        return x0 + int(round(x_cm*s)), y0 + int(round(y_cm*s))

    verts = config.vertices  # Utiliser config.vertices
    # Créer un mapping des labels vers les indices
    label_to_idx = {label: i for i, label in enumerate(config.labels)}
    P = lambda code: cm_to_px(*verts[label_to_idx[code]])

    # parquet orange
    img[:] = (20,110,240)
    for i in range(16):
        shade = 12 if i%2==0 else 26
        overlay = img.copy()
        xL = x0 + i*new_w//16
        xR = x0 + (i+1)*new_w//16
        cv2.rectangle(overlay,(xL,y0),(xR,y0+new_h),(20+shade,110+shade,240+shade),-1)
        cv2.addWeighted(overlay,0.22,img,0.78,0,img)

    # lignes principales
    LINE=(255,255,255); THICK=max(2,line_thickness)
    RIM =(40,160,255);  BACKBOARD=(230,230,230)

    cv2.rectangle(img,(padding,padding),(W-padding,H-padding),LINE,THICK)
    cv2.line(img,P("19"),P("23"),LINE,THICK)
    cv2.circle(img,P("21"),int(round(config.center_circle_radius*s)),LINE,THICK)

    # raquettes
    cy_cm=W_cm/2
    y_top=cy_cm-config.paint_width/2
    y_bot=cy_cm+config.paint_width/2
    cv2.rectangle(img,cm_to_px(0,y_top),cm_to_px(config.paint_length,y_bot),LINE,THICK)            # gauche
    cv2.rectangle(img,cm_to_px(L_cm-config.paint_length,y_top),cm_to_px(L_cm,y_bot),LINE,THICK)    # droite

    # segments corner-3
    for a,b in [("02","10"),("07","11"),("35","31"),("40","32")]:
        cv2.line(img,P(a),P(b),LINE,THICK)

    # arcs 3 pts: arrêt EXACT aux corner-3 + orientation corrigée
    def circle_line_intersections(hx,hy,r_cm,x_line):
        dx=x_line-hx
        if abs(dx)>r_cm: return None
        dy=(r_cm**2-dx**2)**0.5
        return (x_line,hy-dy),(x_line,hy+dy)

    def angle_deg(hx,hy,x_cm,y_cm):
        X,Y=cm_to_px(x_cm,y_cm); Hx,Hy=cm_to_px(hx,hy)
        return np.degrees(np.arctan2(Y-Hy, X-Hx))

    def norm180(a):
        return (a + 180.0) % 360.0 - 180.0

    hoopL, hoopR = verts[label_to_idx["09"]], verts[label_to_idx["33"]]  # Utiliser label_to_idx
    r3_cm=config.three_point_radius; r3_px=int(round(r3_cm*s))
    x_corner_L=config.corner_3_line_length
    x_corner_R=L_cm-config.corner_3_line_length
    y_min=config.three_point_side_distance; y_max=W_cm-y_min

    def draw_arc_side(hoop, x_corn, want_mid_near_deg):
        inter=circle_line_intersections(hoop[0],hoop[1],r3_cm,x_corn)
        if not inter: return
        (xA,yA),(xB,yB)=inter
        # clip vertical aux extrémités des lignes de corner-3
        yA=min(max(yA,y_min),y_max); yB=min(max(yB,y_min),y_max)
        a1=norm180(angle_deg(hoop[0],hoop[1],xA,yA))
        a2=norm180(angle_deg(hoop[0],hoop[1],xB,yB))

        # Deux parcours possibles (CCW) : choisir celui dont le milieu est le plus proche de want_mid_near_deg
        start1, end1 = (a1, a2) if a1 <= a2 else (a2, a1)
        mid1 = norm180((start1+end1)/2.0)

        start2, end2 = (a1, a2+360) if a1 > a2 else (a1+360, a2)
        mid2 = norm180((start2+end2)/2.0)

        d1 = abs(norm180(mid1 - want_mid_near_deg))
        d2 = abs(norm180(mid2 - want_mid_near_deg))

        a_start, a_end = (start2, end2) if d2 < d1 else (start1, end1)

        Hx,Hy=cm_to_px(*hoop)
        cv2.ellipse(img,(Hx,Hy),(r3_px,r3_px),0, a_start, a_end, LINE, THICK)

    # Gauche : arc tourné vers le centre -> milieu proche de 0°
    draw_arc_side(hoopL, x_corner_L, want_mid_near_deg=0.0)
    # Droite : arc tourné vers le centre -> milieu proche de ±180°
    draw_arc_side(hoopR, x_corner_R, want_mid_near_deg=180.0)

    # paniers (planche + anneau) des deux côtés
    bb_from_bl = float(getattr(config,"hoop_distance_from_baseline",122))
    rim_from_bl= float(config.hoop_distance)
    # gauche
    p_bb_L = cm_to_px(bb_from_bl, cy_cm)
    cv2.line(img,(p_bb_L[0], p_bb_L[1]-int(round(91.5*s))), (p_bb_L[0], p_bb_L[1]+int(round(91.5*s))), BACKBOARD, max(THICK-1,1))
    cv2.circle(img, cm_to_px(rim_from_bl, cy_cm), int(round(22.86*s)), RIM, max(THICK,3))
    # droite
    p_bb_R = cm_to_px(L_cm-bb_from_bl, cy_cm)
    cv2.line(img,(p_bb_R[0], p_bb_R[1]-int(round(91.5*s))), (p_bb_R[0], p_bb_R[1]+int(round(91.5*s))), BACKBOARD, max(THICK-1,1))
    cv2.circle(img, cm_to_px(L_cm-rim_from_bl, cy_cm), int(round(22.86*s)), RIM, max(THICK,3))

    # demi-cercles lancers-francs (extérieur plein, intérieur pointillé)
    ft_r_px = int(round(config.center_circle_radius*s))
    def dashed_half_circle(center_px, start_deg, end_deg, step=3):
        for a in np.linspace(start_deg, end_deg, 40):
            if int(a) % 2 == 0:
                x = center_px[0] + int(ft_r_px*np.cos(np.deg2rad(a)))
                y = center_px[1] + int(ft_r_px*np.sin(np.deg2rad(a)))
                x2= center_px[0] + int(ft_r_px*np.cos(np.deg2rad(a+step)))
                y2= center_px[1] + int(ft_r_px*np.sin(np.deg2rad(a+step)))
                cv2.line(img,(x,y),(x2,y2),LINE,THICK)

    # gauche (centre à x = paint_length)
    cL = cm_to_px(config.paint_length, cy_cm)
    # extérieur (vers centre terrain, à droite) = -90..90 plein 
    cv2.ellipse(img, cL, (ft_r_px, ft_r_px), 0, -90, 90, LINE, THICK)
    # intérieur (vers panier gauche, à gauche) = 90..270 pointillé 
    dashed_half_circle(cL, 90, 270)

    # droite (centre à x = L - paint_length)
    cR = cm_to_px(L_cm - config.paint_length, cy_cm)
    # extérieur (vers centre, à gauche) = 90..270 plein
    cv2.ellipse(img, cR, (ft_r_px, ft_r_px), 0, 90, 270, LINE, THICK)
    # intérieur (vers panier droit, à droite) = -90..90 pointillé
    dashed_half_circle(cR, -90, 90)

    # petits traits (sortie de balle / fautes)
    tick_xs = [853.0, L_cm - 853.0]   # positions en cm depuis baseline
    tick_len_cm = 30.0                # longueur verticale du trait (ajuste si besoin)
    for tx in tick_xs:
        # haut
        p1 = cm_to_px(tx, 0.0)
        p2 = cm_to_px(tx, tick_len_cm)
        cv2.line(img, p1, p2, LINE, THICK)

        # bas
        p3 = cm_to_px(tx, W_cm)
        p4 = cm_to_px(tx, W_cm - tick_len_cm)
        cv2.line(img, p3, p4, LINE, THICK)

    return img


def draw_points_on_court(
    config: BasketBallCourtConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a basketball court.

    Args:
        config (BasketBallCourtConfiguration): Configuration object containing the
            dimensions and layout of the court.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the court in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the court dimensions.
            Defaults to 0.1.
        court (Optional[np.ndarray], optional): Existing court image to draw points on.
            If None, a new court will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the basketball court with points drawn on it.
    """
    if court is None:
        court = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return court


def draw_paths_on_court(
    config: BasketBallCourtConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws paths on a basketball court.

    Args:
        config (BasketBallCourtConfiguration): Configuration object containing the
            dimensions and layout of the court.
        paths (List[np.ndarray]): List of paths, where each path is an array of (x, y)
            coordinates.
        color (sv.Color, optional): Color of the paths.
            Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the court in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the court dimensions.
            Defaults to 0.1.
        court (Optional[np.ndarray], optional): Existing court image to draw paths on.
            If None, a new court will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the basketball court with paths drawn on it.
    """
    if court is None:
        court = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=court,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness
            )

        return court


def game_style_annotate_frame(frame, players_detections, court_ball_xy, court_players_xy, court_referees_xy, CONFIG):

    annotated_frame = draw_court(CONFIG)
    annotated_frame = draw_points_on_court(
        config=CONFIG,
        xy=court_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        court=annotated_frame)
    annotated_frame = draw_points_on_court(
        config=CONFIG,
        xy=court_players_xy[players_detections.class_id == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        court=annotated_frame)
    annotated_frame = draw_points_on_court(
        config=CONFIG,
        xy=court_players_xy[players_detections.class_id == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=16,
        court=annotated_frame)
    annotated_frame = draw_points_on_court(
        config=CONFIG,
        xy=court_referees_xy,
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        court=annotated_frame)

    return annotated_frame

def reorder_keypoints_to_labels(keypoints_robo, config: BasketBallCourtConfiguration) -> np.ndarray:
    """
    Réordonne les keypoints prédits (indexés par IDs Roboflow 0..N-1)
    pour correspondre à l'ordre de config.labels.
    Retour: array shape (len(config.labels), 3) [x, y, conf]
    """
    label_to_robo = {label: rid for rid, label in config.roboflow_to_labels.items()}
    ordered = np.zeros((len(config.labels), 3), dtype=float)
    
    for i, label in enumerate(config.labels):
        rid = label_to_robo.get(label, None)
        if rid is not None and rid < len(keypoints_robo):
            ordered[i] = keypoints_robo[rid]
        else:
            ordered[i] = np.array([np.nan, np.nan, 0.0])
    
    return ordered

def dessiner_lignes_bleues_supervision(frame, keypoints_ordered, config, seuil_confiance=0.4):
    """
    Dessine les lignes bleues et keypoints stylés avec supervision
    CORRIGÉ : Même logique que la fonction cv2, juste avec supervision
    """
    annotated_frame = frame.copy()
    
    valid_keypoints = []
    valid_confidences = []
    
    for i, kp in enumerate(keypoints_ordered):
        if not np.isnan(kp[0]) and not np.isnan(kp[1]) and kp[2] >= seuil_confiance:
            valid_keypoints.append(kp[:2])  # x, y seulement
            valid_confidences.append(kp[2])
    
    if len(valid_keypoints) > 0:
        valid_keypoints = np.array(valid_keypoints)
        valid_confidences = np.array(valid_confidences)
        
        # Créer l'objet KeyPoints pour supervision
        from supervision.keypoint.core import KeyPoints
        keypoints_sv = KeyPoints(
            xy=valid_keypoints.reshape(1, -1, 2),  # Shape: (1, N, 2)
            confidence=valid_confidences.reshape(1, -1)  # Shape: (1, N)
        )
        
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),  # Rose
            radius=8
        )
        annotated_frame = vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=keypoints_sv
        )
    
    for edge in config.edges:
        label1, label2 = edge
        
        try:
            idx1 = config.labels.index(f"{label1:02d}")
            idx2 = config.labels.index(f"{label2:02d}")
            
            if (idx1 < len(keypoints_ordered) and idx2 < len(keypoints_ordered) and
                not np.isnan(keypoints_ordered[idx1][0]) and not np.isnan(keypoints_ordered[idx1][1]) and
                not np.isnan(keypoints_ordered[idx2][0]) and not np.isnan(keypoints_ordered[idx2][1]) and
                keypoints_ordered[idx1][2] >= seuil_confiance and 
                keypoints_ordered[idx2][2] >= seuil_confiance):
                
                pt1 = keypoints_ordered[idx1][:2]  # x, y seulement
                pt2 = keypoints_ordered[idx2][:2]  # x, y seulement
                
                line_keypoints = np.array([pt1, pt2])
                line_confidences = np.array([keypoints_ordered[idx1][2], keypoints_ordered[idx2][2]])
                
                line_keypoints_sv = KeyPoints(
                    xy=line_keypoints.reshape(1, -1, 2),  # Shape: (1, 2, 2)
                    confidence=line_confidences.reshape(1, -1)  # Shape: (1, 2)
                )
                
                edge_annotator = sv.EdgeAnnotator(
                    color=sv.Color.from_hex('#00BFFF'),  # Bleu
                    thickness=3,
                    edges=[(0, 1)] 
                )
                annotated_frame = edge_annotator.annotate(
                    scene=annotated_frame,
                    key_points=line_keypoints_sv
                )
                
        except ValueError:
            continue
    
    return annotated_frame


def dessiner_terrain_reconstruit(frame, frame_all_points, config, seuil_confiance=0.0):
    """
    Version adaptée pour tracer le terrain reconstruit par homographie
    """
    annotated_frame = frame.copy()
    
    if len(frame_all_points) > 0:
        keypoints_sv = sv.KeyPoints(
            xy=frame_all_points.reshape(1, -1, 2),  # Shape: (1, N, 2)
            confidence=np.ones(len(frame_all_points)).reshape(1, -1)  # Shape: (1, N) ← CORRIGÉ !
        )
        
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),  # Rose
            radius=10  # Plus petit car beaucoup de points
        )
        annotated_frame = vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=keypoints_sv
        )
    
    for edge in config.edges:
        label1, label2 = edge
        
        try:
            idx1 = config.labels.index(f"{label1:02d}")
            idx2 = config.labels.index(f"{label2:02d}")
            
            if idx1 < len(frame_all_points) and idx2 < len(frame_all_points):
                pt1 = frame_all_points[idx1][:2]  # x, y seulement
                pt2 = frame_all_points[idx2][:2]  # x, y seulement
                
                line_keypoints = np.array([pt1, pt2])
                line_confidences = np.array([1.0, 1.0]).reshape(1, -1)  # Shape: (1, 2) ← CORRIGÉ !
                
                line_keypoints_sv = sv.KeyPoints(
                    xy=line_keypoints.reshape(1, -1, 2),  # Shape: (1, 2, 2)
                    confidence=line_confidences  # Shape: (1, 2)
                )
                
                edge_annotator = sv.EdgeAnnotator(
                    color=sv.Color.from_hex('#00BFFF'),  # Bleu
                    thickness=6,  # Plus fin car beaucoup de lignes
                    edges=[(0, 1)]  # Edge entre les 2 points
                )
                annotated_frame = edge_annotator.annotate(
                    scene=annotated_frame,
                    key_points=line_keypoints_sv
                )
                
        except ValueError:
            continue
    
    return annotated_frame