// Orientation d'une voiture avec l'élève MobileNetV3-Small (distillé d'Orient Anything, export
// TensorFlow.js). Interface : chargerClassifieur(url) puis classer(canvas) → probabilités des
// 8 secteurs. Le canvas est la découpe carrée de la voiture (decoupeCarree de orientation.js).
// `tf` est le global chargé par vendor/tf.min.js.

let modele = null;

export async function chargerClassifieur(url) {
  modele = await tf.loadGraphModel(url);
  const [, hauteur, largeur] = modele.inputs[0].shape;
  // Première inférence à vide : compile les shaders WebGL avant d'ouvrir la caméra.
  tf.tidy(() => modele.execute(tf.zeros([1, hauteur, largeur, 3])));
  return { largeur, hauteur };
}

export async function classer(canvas) {
  const sortie = tf.tidy(() => modele.execute(tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0)));
  const probas = await sortie.data();
  sortie.dispose();
  return Array.from(probas);
}
