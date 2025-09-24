# Basket Court Detection & Analytics  

- **Vidéo annotée** :
[![Voir la vidéo](output/video_thumbnail.png)](output/basket_players.mp4)


- **Homography** :  
![Homography](output/Homography.png)

- **Frame annotée** :
![Clustering annotation](output/annotated_img.png)

En raison de la taille des modèles et des visualisations, on a décidé de les mettre sur drive et vous pouvez les consulter directement via ce lien : https://drive.google.com/drive/folders/183f0wHigSy_S76Y08t2-pyPvP7xptyx9?usp=sharing , l’arborescence du projet reste tout de meme inchangée

## Description  
Ce projet propose une pipeline développée pour l’analyse d’un match de **basket-ball** à partir de vidéos.  
Il s’appuie sur un modèle YOLO personnalisé pour la détection du terrain, un modèle RF-DETR pour la detection des joueurs , arbitres, balle, panier ...

- **Détection des lignes et keypoints** du terrain de basket grâce à un modèle YOLO entraîné en local
- **Annotation automatique de vidéos** avec les détections  
- **Détection du ballon et des joueurs** en utilisant un modèle RF-DETR local
- **Reconnaissance des chiffres** pour estimer automatiquement le **score affiché à l’écran**, et cela à l'aide de réseaux de neurones de géométries différentes
- **Prédiction de tirs réussis ou ratés** à partir de la trajectoire du ballon  

L’objectif est de fournir une boîte à outils initiale robuste pour la **sport analytics appliquée au basket-ball**.  

---

## Structure du projet  
```
Basket_Court/
│
├── Main_notebook.ipynb              # Notebook principal - pipeline complet
├── score_detection.ipynb              # Notebook pour la reconnaissance des nombres
├── models/
│   ├── BasketBall_good_court.pt    # Modèle YOLO pour détecter le terrain
│   └── digits_dense_ultra.h5       # Réseau de neurones pour détecter le score
│   └── digits_conv_sparse.h5       # Réseau de neurones pour détecter le score
│   └── digits_hybrid.h5            # Réseau de neurones pour détecter le score
│   └── ball_model                  # Modèle RF-DETR pour détecter la balle
│   └── general_model               # Modèle RF-DETR pour détecter les autres classes
├── videos_input/
│   └── basket_cut.mp4              # Vidéo d'entrée 
├── output/
│   └── basket_cut_annotated.mp4    # Vidéo annotée en sortie
├── utils/
│   ├── configs/
│   │   └── basket_config.py        # Configuration du terrain
│   │   └── process_video.py        # Traitement vidéo
│   └── annotators/
│       └── basket_annot.py         # Fonctions pour dessiner le terrain 
│   └── common/
│       └── view_transformer.py     # Fonctions faire l'homographie 
└── README.md                       # Documentation
└── requirements.txt                # Bibliothèques nécessaires
```

---

## Utilisation  

### 1. Notebook interactif  
Lancer le notebook principal et exécuter les cellules :  
```bash
jupyter notebook Main_notebook.ipynb
```

### 2. Fonctionnalités principales  
- Détection du terrain : keypoints + lignes tracées  
- Génération d’une vidéo annotée (joueurs, ballon, terrain)  
- Extraction automatique du **score** via reconnaissance des chiffres  
- Prédiction automatique des **tirs réussis ou ratés**  
- Correction de la prédiction avec la reconnaissance des nombres à partir du crop du résultat de match

### 3. Fonctionnalités Avancées

### Détection Intelligente
- **Homographie automatique** : Projection 2D du terrain en coordonnées réelles
- **Clustering des joueurs** : Séparation automatique des équipes


### 4. Résultats attendus  
- Une **vidéo annotée** avec les détections en temps réel  
- Un **score automatiquement mis à jour** depuis la vidéo  
- Une **prédiction des tirs réussis** basée sur la trajectoire du ballon  


---

## Améliorations possibles  
- Meilleur CNN pour la reconnaissance des scores, entraîné sur un dataset spécifique aux matchs de basket  
- Ajout du **tracking multi-joueurs** avec ID unique par joueur (plus robuste que le bytetrack et le boostrack qui ne sont pas optimisés pour le basket)
- Détection et classification d’**événements complexes** (passes, dribbles, fautes)  

---

## Auteurs  
- **Amine SAADI**  – Étudiant en 3ème année à l'École polytechnique  
- **Aymen HAOUAS** – Étudiant en 3ème année à l'École polytechnique   

