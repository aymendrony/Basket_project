# CONFIGS
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

@dataclass
class BasketBallCourtConfiguration:
    width: int = 1524   # [cm] total court width
    length: int = 2865  # [cm] total court length

    sideline_distance: int = 853  # [cm] distance from baseline to the foul line along the side

    paint_width: int = 579   # [cm] lane (paint) width
    paint_length: int = 488  # [cm] lane (paint) length

    center_circle_radius: int = 183  # [cm] radius of center circle

    three_point_radius: int = 724          # [cm] 3-point arc radius (from basket center)
    three_point_side_distance: int = 91    # [cm] distance from sideline to 3-point line (corner)
    corner_3_line_length: int =  427         # [cm] distance from baseline to the end of the corner-3 line

    hoop_distance_from_baseline: int = 122  # [cm] distance from hoop (backboard plane) to baseline
    hoop_distance: int = 160             # [cm] distance from hoop (basket itself) to baseline

    # MAPPING entre IDs Roboflow et labels personnalisés
    roboflow_to_labels: dict = field(default_factory=lambda: {
        0: "01",   # top-left
        1: "02",   # haut gauche extérieur
        2: "04",   # haut extérieur gauche
        3: "05",   # bas extérieur gauche
        4: "07",   # bas gauche extérieur
        5: "08",   # bottom-left
        6: "09",   # centre anneau gauche
        7: "10",   # haut gauche intérieur
        8: "11",   # bas gauche intérieur
        9: "12",   # haut intérieur gauche
        10: "13",  # milieu intérieur gauche
        11: "14",  # bas intérieur gauche
        12: "15",  # point faute haut
        13: "16",  # point 3pts gauche
        14: "17",  # point faute bas
        15: "19",  # ligne médiane haut
        16: "21",  # centre
        17: "23",  # ligne médiane bas
        18: "25",  # point faute haut droit
        19: "26",  # point 3pts droit
        20: "27",  # point faute bas droit
        21: "28",  # haut intérieur droit
        22: "29",  # milieu intérieur droit
        23: "30",  # bas intérieur droit
        24: "31",  # haut droit intérieur
        25: "32",  # bas droit intérieur
        26: "33",  # centre anneau droit
        27: "34",  # top-right
        28: "35",  # haut droit extérieur
        29: "37",  # haut extérieur droit
        30: "38",  # bas extérieur droit
        31: "40",  # bas droit extérieur
        32: "41"   # bottom-right
    })


    @property
    def vertices(self) -> List[Tuple[float, float]]: 
        """
        Reconstruit les vertices du terrain de basket en cm
        (origine = coin haut gauche, x=longueur, y=largeur).
        Retourne la liste ordonnée correspondant à self.labels.
        """

        L = float(self.length)
        W = float(self.width)

        # Dimensions
        paint_w = float(self.paint_width)
        paint_l = float(self.paint_length)
        hoop_dx = float(self.hoop_distance)    # centre anneau depuis baseline
        r_3pt = float(self.three_point_radius)
        d_3pt_side = float(self.three_point_side_distance)
        cx, cy = L / 2.0, W / 2.0


        # Coins
        v1  = (0.0, 0.0)   # 1: top-left
        v34 = (L, 0.0)     # 34: top-right
        v8  = (0.0, W)     # 8: bottom-left
        v41 = (L, W)       # 41: bottom-right

        # Hoops
        v9  = (hoop_dx, cy)       # centre anneau gauche
        v33 = (L - hoop_dx, cy)   # centre anneau droit

        # Paint gauche
        y_lo = (W - paint_l) / 2.0
        y_hi = (W + paint_l) / 2.0
        v4  = (0, y_lo)   # haut ext gauche
        v5  = (0, y_hi)   # bas ext gauche
        v12 = (paint_w, y_lo)   # haut int gauche
        v13 = (paint_w, cy)     # milieu int gauche
        v14 = (paint_w, y_hi)   # bas int gauche

        # Paint droite (symétrique)
        v37 = (L, y_lo)  # haut ext droit
        v38 = (L, y_hi)  # bas ext droit
        v28 = (L - paint_w, y_lo)  # haut int droit
        v29 = (L - paint_w, cy)    # milieu int droit
        v30 = (L - paint_w, y_hi)  # bas int droit

        # Segments verticaux 3pts
        v2  = (0, d_3pt_side)   # haut gauche extérieur
        v7  = (0, W-d_3pt_side)     # bas gauche extérieur
        v10 = (float(self.corner_3_line_length), d_3pt_side)  # haut gauche intérieur
        v11 = (float(self.corner_3_line_length), W-d_3pt_side)  # bas gauche intérieur

        v35 = (L, d_3pt_side)   # haut droit extérieur
        v40 = (L, W-d_3pt_side)     # bas droit extérieur
        v31 = (L - float(self.corner_3_line_length), d_3pt_side)  # haut droit intérieur
        v32 = (L - float(self.corner_3_line_length), W-d_3pt_side)  # bas droit intérieur

        # Points du cercle 3pts en face du panier
        v16 = (v9[0] + r_3pt, cy)     # gauche
        v26 = (v33[0] - r_3pt, cy)    # droite

        # Ligne médiane
        v19 = (cx, 0.0)   # haut
        v21 = (cx, cy)    # centre
        v23 = (cx, W)     # bas

        # Points faute/remise
        v15 = (float(self.sideline_distance), 0.0)
        v25 = (L-float(self.sideline_distance), 0.0)
        v17 = (float(self.sideline_distance), W)
        v27 = (L-float(self.sideline_distance), W)

        mapping = {
            "01": v1,
            "02": v2,
            "04": v4,
            "05": v5,
            "07": v7,
            "08": v8,
            "09": v9,
            "10": v10,
            "11": v11,
            "12": v12,
            "13": v13,
            "14": v14,
            "15": v15,
            "16": v16,
            "17": v17,
            "19": v19,
            "21": v21,
            "23": v23,
            "25": v25,
            "26": v26,
            "27": v27,
            "28": v28,
            "29": v29,
            "30": v30,
            "31": v31,
            "32": v32,
            "33": v33,
            "34": v34,
            "35": v35,
            "37": v37,
            "38": v38,
            "40": v40,
            "41": v41,
        }

        return [mapping[label] for label in self.labels]

    # Méthode pour convertir les IDs Roboflow en labels
    def get_label_from_roboflow_id(self, roboflow_id: int) -> str:
        """Convertit un ID Roboflow en label personnalisé."""
        return self.roboflow_to_labels.get(roboflow_id, f"unknown_{roboflow_id}")
    
    # Méthode pour obtenir les edges avec IDs Roboflow
    def get_roboflow_edges(self) -> List[Tuple[int, int]]:
        """Retourne les edges avec les IDs Roboflow pour la compatibilité."""
        roboflow_edges = []
        for edge in self.edges:
            # Convertir les labels en IDs Roboflow
            label1, label2 = edge
            # Trouver l'ID Roboflow correspondant
            roboflow_id1 = None
            roboflow_id2 = None
            for robo_id, label in self.roboflow_to_labels.items():
                if label == f"{label1:02d}":
                    roboflow_id1 = robo_id
                if label == f"{label2:02d}":
                    roboflow_id2 = robo_id
                if roboflow_id1 is not None and roboflow_id2 is not None:
                    break
            
            if roboflow_id1 is not None and roboflow_id2 is not None:
                roboflow_edges.append((roboflow_id1, roboflow_id2))
        
        return roboflow_edges


    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 4), (4, 5), (5, 7), (7, 8), 
        (8, 17), (17,23), (23,27), (27,41),
        (41,40), (40,38), (38,37), (37,35), (35,34),
        (34,25), (25,19), (19,15), (15,1),
        (2,10), (10,16), (16,11), (11,7),
        (4,12), (12,13), (13,14), (14,5),
        (19, 21), (21, 23),
        (35, 31), (31, 26), (26, 32), (32, 40),
        (37, 28), (28, 29), (29, 30), (30, 38)
    ])

    # EDGES avec IDs Roboflow (pour compatibilité directe avec le squelette)
    roboflow_edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), 
        (5, 14), (14, 23), (23, 20), (20, 32),
        (32, 31), (31, 30), (30, 29), (29, 28), (28, 27),
        (27, 18), (18, 15), (15, 12), (12, 0),
        (1, 7), (7, 13), (13, 8), (8, 4),
        (2, 9), (9, 10), (10, 11), (11, 3),
        (15, 16), (16, 17),
        (28, 24), (24, 19), (19, 25), (25, 31),
        (29, 21), (21, 22), (22, 23), (23, 30)
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "04", "05", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "19", "21",
        "23", "25", "26", "27", "28", "29", "30", "31", "32",
        "33", "34", "35", "37", "38", "40", "41"
    ])


    colors: List[str] = field(default_factory=lambda: [
        "#FF6347", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF6347",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF6347", "#FF1493", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347", "#00BFFF", "#FF6347", "#00BFFF", "#00BFFF", "#00BFFF",
        "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347", "#00BFFF", "#00BFFF",
        "#00BFFF", "#00BFFF", "#FF6347"
    ])