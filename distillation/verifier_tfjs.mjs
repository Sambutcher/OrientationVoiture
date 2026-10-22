// Étape 5 : le modèle TF.js donne-t-il les mêmes secteurs que PyTorch ? Usage (Node + tfjs-node) :
// node distillation/verifier_tfjs.mjs models/orientation <reference/>
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import tf from '@tensorflow/tfjs-node';

const [dossier, dossierRef] = process.argv.slice(2);
const modele = await tf.loadGraphModel(pathToFileURL(resolve(dossier, 'model.json')).href);
console.log('entrée', modele.inputs.map((i) => `${i.name} ${i.shape}`), 'sortie', modele.outputs.map((o) => `${o.name} ${o.shape}`));

const ref = JSON.parse(await readFile(resolve(dossierRef, 'sortie.json'), 'utf8'));
const brut = await readFile(resolve(dossierRef, 'entree.bin'));
const entree = tf.tensor(new Float32Array(brut.buffer, brut.byteOffset, brut.length / 4), ref.forme);
// Lot fixé à 1 par la conversion : une image à la fois.
const obtenu = [];
for (let i = 0; i < ref.forme[0]; i++) {
  obtenu.push((await modele.predict(entree.slice([i], [1])).array())[0]);
}

const argmax = (v) => v.indexOf(Math.max(...v));
let ecart = 0;
let accords = 0;
obtenu.forEach((v, i) => {
  v.forEach((p, k) => { ecart = Math.max(ecart, Math.abs(p - ref.secteurs[i][k])); });
  if (argmax(v) === argmax(ref.secteurs[i])) accords++;
});
console.log(`secteurs identiques : ${accords}/${obtenu.length}, écart max de probabilité : ${ecart.toFixed(4)}`);
if (accords < obtenu.length - 1 || ecart > 0.05) {
  console.error('Conversion TF.js incorrecte.');
  process.exit(1);
}
