# Orientation voiture — notes pour Claude

- PWA **vanilla JS sans build** (modules ES), cible iPhone SE 3ᵉ gén. (Safari), hors ligne.
  Ne pas introduire de bundler ni de framework. Licence AGPL-3.0. Socle repris de Street Biathlon
  (`~/Code/street-biathlon`) : caméra, YOLO11n TF.js (entrée 224×384 portrait), service worker.
- But : orientation des voitures filmées dans la rue, **8 secteurs d'azimut** (tous les 45°),
  azimut seul. Précision modeste voulue par Samuel, priorité à la vitesse.
- Chaîne : YOLO11n (classe COCO 2 « car ») → découpe carrée (plus grand côté × 1,2, gris hors
  image) → **élève MobileNetV3-Small 128 px** (`models/orientation/`) → probabilités des 8 secteurs,
  lissées par voiture d'une analyse à l'autre (`suivre` de `js/orientation.js`).
- L'élève est **distillé d'Orient Anything** (ViT-L `ronormsigma1`, détourage rembg comme la démo ;
  Wang et al. 2024, CC BY 4.0) sur des voitures de COCO 2017 : workflow `distillation.yml`
  (sélection → étiquetage réparti → apprentissage → conversion ONNX → onnx2tf → TF.js float16).
  Le PC de dev n'a pas l'AVX : tout le calcul tourne sur GitHub Actions. Essai rapide :
  `gh workflow run distillation.yml -R Sambutcher/OrientationVoiture --ref premiere-version -f max_train=300 -f max_val=40 -f morceaux=2 -f epoques=2`.
  Résultat : artefact `orientation-tfjs` (avec `rapport.md` et `planche.jpg`) à copier dans `models/orientation/`.
- La découpe doit rester identique entre `distillation/commun.py` et `js/orientation.js`.
  `SENS` de `js/orientation.js` relie l'azimut d'Orient Anything au cap affiché (0° = la voiture
  vient vers nous, 90° = vers la droite) : à vérifier sur `planche.jpg` après chaque distillation.
- Logique pure (`js/orientation.js`, `js/geometry.js`) testée par `node --test tests/`
  (Node 18 apt, pas de npm). Garder le DOM et `tf` hors de ces modules.
- Tout fichier servi doit figurer dans `FICHIERS` de `sw.js` ; `VERSION` (sw.js + index.html)
  change à chaque mise en ligne. `tests/pwa.test.js` vérifie les deux.
- Dev sur iPhone : `python3 tools/serve.py` (HTTPS auto-signé, port 8443, sans service worker).
  `?debug` affiche les cadences et les probabilités.
- Ne pas tester les pages dans Chromium : Samuel teste lui-même (navigateur, iPhone).
- Git : Claude travaille sur des branches, ne pousse jamais sur `main` ; Samuel fusionne.
- Langue du code et des textes : français.
