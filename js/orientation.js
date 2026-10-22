// Orientation des voitures, logique pure (sans navigateur ni tf) : recadrage donné au modèle,
// secteurs d'azimut, suivi des voitures d'une image à l'autre et lissage de leurs probabilités.

// Même recadrage qu'à l'apprentissage (distillation/commun.py) : ne changer qu'ensemble.
export const MARGE = 1.2;
export const TAILLE_ENTREE = 128;
export const GRIS = 'rgb(128,128,128)';

// Secteur k du modèle (azimut d'Orient Anything ≈ k × 45°, 0 = voiture vue de face) → cap de la
// voiture vu de dessus : 0° = elle vient vers nous, 90° = elle va vers la droite de l'image,
// 180° = elle s'éloigne. SENS : signe reliant l'azimut à ce cap. Vérifié sur planche.jpg de la
// distillation : à 90° les voitures ont l'avant à gauche de l'image, à 270° à droite.
const SENS = -1;
export const NB_SECTEURS = 8;

const NOMS = [
  'de face',
  'trois-quarts avant, vers la droite',
  'de profil, vers la droite',
  'trois-quarts arrière, vers la droite',
  'de dos',
  'trois-quarts arrière, vers la gauche',
  'de profil, vers la gauche',
  'trois-quarts avant, vers la gauche',
];

// Cap (degrés, 0–359) de la voiture pour le secteur k.
export function cap(k) {
  return (((SENS * k * 45) % 360) + 360) % 360;
}

export function nomSecteur(k) {
  return NOMS[cap(k) / 45];
}

// Carré centré sur la boîte, côté = plus grand côté × MARGE.
export function decoupeCarree(boite) {
  const cote = Math.max(boite.w, boite.h) * MARGE;
  return { x: boite.x + boite.w / 2 - cote / 2, y: boite.y + boite.h / 2 - cote / 2, cote };
}

// Rectangles source → destination pour dessiner la découpe dans un carré de `taille` px, la
// partie hors de l'image source restant grise (drawImage déborde mal sur certains Safari).
export function rectanglesDecoupe(decoupe, srcW, srcH, taille = TAILLE_ENTREE) {
  const k = taille / decoupe.cote;
  const sx = Math.max(0, decoupe.x);
  const sy = Math.max(0, decoupe.y);
  const sw = Math.min(srcW, decoupe.x + decoupe.cote) - sx;
  const sh = Math.min(srcH, decoupe.y + decoupe.cote) - sy;
  if (sw <= 0 || sh <= 0) return null;
  return { sx, sy, sw, sh, dx: (sx - decoupe.x) * k, dy: (sy - decoupe.y) * k, dw: sw * k, dh: sh * k };
}

export function secteur(probas) {
  let k = 0;
  for (let i = 1; i < probas.length; i++) if (probas[i] > probas[k]) k = i;
  return k;
}

function iou(a, b) {
  const l = Math.max(a.x, b.x);
  const t = Math.max(a.y, b.y);
  const r = Math.min(a.x + a.w, b.x + b.w);
  const d = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, r - l) * Math.max(0, d - t);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

const IOU_MEME_VOITURE = 0.3;
const POIDS_NOUVEAU = 0.4; // part de la nouvelle mesure dans la moyenne glissante

// Voitures de l'image précédente ({ id, boite, probas }) + mesures de l'image courante
// ({ boite, probas }) → voitures courantes, probabilités lissées pour celles déjà vues.
export function suivre(precedentes, mesures, prochainId = { valeur: 1 }) {
  const libres = new Set(precedentes);
  return mesures.map((m) => {
    let meilleure = null;
    let meilleurIou = IOU_MEME_VOITURE;
    for (const p of libres) {
      const v = iou(p.boite, m.boite);
      if (v >= meilleurIou) { meilleure = p; meilleurIou = v; }
    }
    if (!meilleure) return { id: prochainId.valeur++, boite: m.boite, probas: m.probas };
    libres.delete(meilleure);
    const probas = m.probas.map((p, i) => meilleure.probas[i] + (p - meilleure.probas[i]) * POIDS_NOUVEAU);
    return { id: meilleure.id, boite: m.boite, probas };
  });
}
