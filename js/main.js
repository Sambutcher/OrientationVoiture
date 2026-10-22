// Orientation des voitures : la caméra filme la rue, YOLO11n trouve les voitures, l'élève
// MobileNetV3 donne pour chacune l'un des 8 secteurs d'azimut. La logique pure est dans
// orientation.js et geometry.js ; ce fichier relie le navigateur (caméra, canvas, tf) à cette logique.
import { demarrerCamera } from './camera.js';
import { chargerDetecteur, detecter } from './detector.js';
import { chargerClassifieur, classer } from './classifieur.js';
import { cadrage, letterbox, versEcran } from './geometry.js';
import { GRIS, TAILLE_ENTREE, decoupeCarree, rectanglesDecoupe, secteur, suivre } from './orientation.js';
import { creerRendu } from './render.js';

const URL_DETECTEUR = 'models/yolo11n/model.json';
const URL_ORIENTATION = 'models/orientation/model.json';
const DEBUG = new URLSearchParams(location.search).has('debug');
const MAX_VOITURES = 4; // les plus grandes à l'écran
const COTE_MIN = 40; // px écran : en dessous, la voiture est trop petite pour juger

const video = document.getElementById('video');
const canvas = document.getElementById('vue');
const accueil = document.getElementById('accueil');
const boutonDemarrer = document.getElementById('demarrer');
const statut = document.getElementById('statut');

// Pas de service worker sur le serveur de dev (tools/serve.py) : sinon l'ancien code reste en cache.
if ('serviceWorker' in navigator && location.port !== '8443') {
  navigator.serviceWorker.register('sw.js').catch((e) => console.warn('Service worker :', e));
}

boutonDemarrer.addEventListener('click', async () => {
  boutonDemarrer.disabled = true;
  try {
    garderEcranAllume();
    statut.textContent = 'Chargement des modèles…';
    const entree = await chargerDetecteur(URL_DETECTEUR);
    await chargerClassifieur(URL_ORIENTATION);
    statut.textContent = 'Ouverture de la caméra…';
    await demarrerCamera(video);
    accueil.hidden = true;
    lancer(entree);
  } catch (e) {
    console.error(e);
    statut.textContent = `Impossible de démarrer : ${e.message || e}`;
    boutonDemarrer.disabled = false;
  }
});

function garderEcranAllume() {
  if (!('wakeLock' in navigator)) return;
  const demander = () => navigator.wakeLock.request('screen').catch((e) => console.warn('Wake Lock :', e));
  demander();
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') demander();
  });
}

function lancer(entree) {
  const rendu = creerRendu(canvas);
  const canvasEntree = document.createElement('canvas'); // entrée du détecteur
  canvasEntree.width = entree.largeur;
  canvasEntree.height = entree.hauteur;
  const ctxEntree = canvasEntree.getContext('2d', { willReadFrequently: true });
  const canvasDecoupe = document.createElement('canvas'); // entrée du classifieur
  canvasDecoupe.width = TAILLE_ENTREE;
  canvasDecoupe.height = TAILLE_ENTREE;
  const ctxDecoupe = canvasDecoupe.getContext('2d', { willReadFrequently: true });

  let voitures = []; // { id, boite (écran), probas lissées }
  const prochainId = { valeur: 1 };
  let analyseEnCours = false;
  const stats = { analyses: 0, detectionMs: 0, orientationMs: 0, debut: performance.now(), texte: [] };

  // Une analyse à la fois : détection sur toute la vue, puis orientation de chaque voiture
  // recadrée dans la vidéo pleine résolution.
  async function analyser(c) {
    const debut = performance.now();
    const lb = letterbox(c.sw, c.sh, canvasEntree.width, canvasEntree.height);
    ctxEntree.fillStyle = 'rgb(114,114,114)';
    ctxEntree.fillRect(0, 0, canvasEntree.width, canvasEntree.height);
    ctxEntree.drawImage(video, c.sx, c.sy, c.sw, c.sh, lb.dx, lb.dy, lb.dw, lb.dh);
    const boites = (await detecter(canvasEntree))
      .map((b) => versEcran(b, lb, rendu.largeur, rendu.hauteur))
      .filter((b) => Math.max(b.w, b.h) >= COTE_MIN)
      .sort((a, b) => b.w * b.h - a.w * a.h)
      .slice(0, MAX_VOITURES);
    const milieu = performance.now();

    const k = c.sw / rendu.largeur; // écran → vidéo
    const mesures = [];
    for (const b of boites) {
      const source = { x: c.sx + b.x * k, y: c.sy + b.y * k, w: b.w * k, h: b.h * k };
      const r = rectanglesDecoupe(decoupeCarree(source), video.videoWidth, video.videoHeight);
      if (!r) continue;
      ctxDecoupe.fillStyle = GRIS;
      ctxDecoupe.fillRect(0, 0, TAILLE_ENTREE, TAILLE_ENTREE);
      ctxDecoupe.drawImage(video, r.sx, r.sy, r.sw, r.sh, r.dx, r.dy, r.dw, r.dh);
      mesures.push({ boite: b, probas: await classer(canvasDecoupe) });
    }
    voitures = suivre(voitures, mesures, prochainId);
    stats.analyses++;
    stats.detectionMs = milieu - debut;
    stats.orientationMs = mesures.length ? (performance.now() - milieu) / mesures.length : 0;
  }

  function image(t) {
    const c = cadrage(video.videoWidth, video.videoHeight, rendu.largeur, rendu.hauteur);
    rendu.video(video, c);
    if (!analyseEnCours) {
      analyseEnCours = true;
      analyser(c)
        .catch((e) => console.error(e))
        .finally(() => { analyseEnCours = false; });
    }
    for (const v of voitures) {
      const s = secteur(v.probas);
      rendu.voiture(v.boite, s, v.probas[s]);
    }
    if (DEBUG) {
      const duree = (t - stats.debut) / 1000;
      if (duree >= 1) {
        stats.texte = [
          `${(stats.analyses / duree).toFixed(1)} analyses/s, ${voitures.length} voiture(s)`,
          `détection ${stats.detectionMs.toFixed(0)} ms, orientation ${stats.orientationMs.toFixed(0)} ms/voiture`,
          ...voitures.map((v) => `#${v.id} ${v.probas.map((p) => p.toFixed(2)).join(' ')}`),
        ];
        stats.analyses = 0;
        stats.debut = t;
      }
      rendu.texte(stats.texte);
    }
    requestAnimationFrame(image);
  }
  requestAnimationFrame(image);
}
