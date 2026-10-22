// Dessin dans le canvas plein écran (coordonnées en points CSS) : vidéo, voitures détectées avec
// leur orientation, texte de diagnostic.
import { cap, nomSecteur } from './orientation.js';

const CONFIANCE_MIN = 0.4; // en dessous, orientation affichée comme incertaine

export function creerRendu(canvas) {
  const ctx = canvas.getContext('2d');
  let largeur = 0;
  let hauteur = 0;

  function redimensionner() {
    const dpr = window.devicePixelRatio || 1;
    largeur = window.innerWidth;
    hauteur = window.innerHeight;
    canvas.width = Math.round(largeur * dpr);
    canvas.height = Math.round(hauteur * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener('resize', redimensionner);
  redimensionner();

  function video(source, c) {
    ctx.drawImage(source, c.sx, c.sy, c.sw, c.sh, 0, 0, largeur, hauteur);
  }

  // Cadre de la voiture, boussole vue de dessus (nous sommes en bas) et nom du secteur.
  function voiture(boite, k, confiance) {
    const sure = confiance >= CONFIANCE_MIN;
    const couleur = sure ? '#ffd23f' : '#9a9a9a';
    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = couleur;
    ctx.strokeRect(boite.x, boite.y, boite.w, boite.h);

    const r = Math.max(16, Math.min(30, boite.w / 5));
    const cx = boite.x + boite.w / 2;
    const cy = Math.max(r + 4, boite.y - r - 30);
    boussole(cx, cy, r, cap(k), couleur);

    const texte = sure ? nomSecteur(k) : `${nomSecteur(k)} ?`;
    ctx.font = '600 14px -apple-system, system-ui, sans-serif';
    const l = ctx.measureText(texte).width + 16;
    const x = Math.min(Math.max(4, cx - l / 2), largeur - l - 4);
    const y = cy + r + 4;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.beginPath();
    ctx.roundRect(x, y, l, 22, 11);
    ctx.fill();
    ctx.fillStyle = couleur;
    ctx.textBaseline = 'middle';
    ctx.fillText(texte, x + 8, y + 11);
    ctx.restore();
  }

  // Cap : 0° = la voiture vient vers nous (vers le bas), 90° = vers la droite de l'image.
  function boussole(cx, cy, r, capDeg, couleur) {
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fill();
    // Nous : petit repère en bas du cercle.
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(cx, cy + r - 2);
    ctx.lineTo(cx - 4, cy + r + 4);
    ctx.lineTo(cx + 4, cy + r + 4);
    ctx.fill();
    // Voiture vue de dessus, avant (pare-brise) vers son cap.
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(Math.PI - (capDeg * Math.PI) / 180);
    const w = r * 0.62;
    const h = r * 1.3;
    ctx.fillStyle = couleur;
    ctx.beginPath();
    ctx.roundRect(-w / 2, -h / 2, w, h, w * 0.3);
    ctx.fill();
    ctx.fillStyle = '#222';
    ctx.fillRect(-w * 0.36, -h * 0.3, w * 0.72, h * 0.16);
    ctx.restore();
  }

  function texte(lignes) {
    ctx.save();
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(0, 0, largeur, lignes.length * 16 + 50);
    ctx.fillStyle = '#fff';
    lignes.forEach((l, i) => ctx.fillText(l, 8, 50 + i * 16));
    ctx.restore();
  }

  return {
    video, voiture, texte,
    get largeur() { return largeur; },
    get hauteur() { return hauteur; },
  };
}
