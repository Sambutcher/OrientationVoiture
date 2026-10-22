# Orientation voiture

Application web (PWA) pour iPhone, hors connexion : filmez la rue, chaque voiture reçoit son
orientation parmi 8 secteurs (de face, trois-quarts avant, de profil, trois-quarts arrière, de
dos, à gauche ou à droite), affichée sur une boussole vue de dessus.

## Fonctionnement

1. **Détection** : YOLO11n (Ultralytics, AGPL-3.0) converti en TensorFlow.js trouve les voitures.
2. **Orientation** : chaque voiture est recadrée et passée à un MobileNetV3-Small de 128 px qui
   prédit le secteur d'azimut. Ce petit réseau est *distillé* d'[Orient Anything](https://github.com/SpatialVision/Orient-Anything)
   (Wang et al. 2024, CC BY 4.0) : le grand modèle a étiqueté des voitures de
   [COCO 2017](https://cocodataset.org), le petit a appris à l'imiter avec environ 300 fois moins de calcul.

## Développement

- Tests : `node --test tests/`
- Serveur HTTPS pour l'iPhone : `python3 tools/serve.py`, puis `https://<ip-du-pc>:8443` (`?debug` pour les mesures)
- Distillation (GitHub Actions) : `gh workflow run distillation.yml --ref premiere-version`,
  puis copier l'artefact `orientation-tfjs` dans `models/orientation/`.

## Crédits

- Orient Anything : Z. Wang et al., *Orient Anything: Learning Robust Object Orientation
  Estimation from Rendering 3D Models*, 2024 — code et poids CC BY 4.0.
- COCO 2017 : annotations CC BY 4.0 ; images Flickr sous leurs licences respectives (utilisées
  pour l'apprentissage seulement, non distribuées).
- YOLO11n : Ultralytics, AGPL-3.0. TensorFlow.js : Apache-2.0.
