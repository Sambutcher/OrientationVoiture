import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cadrage, letterbox, versEcran } from '../js/geometry.js';
import { NB_SECTEURS, cap, nomSecteur, decoupeCarree, rectanglesDecoupe, secteur, suivre } from '../js/orientation.js';

test('cadrage couvre l\'écran, letterbox et retour en coordonnées écran', () => {
  const p = cadrage(1280, 720, 360, 640);
  assert.equal(p.sh, 720);
  assert.equal(p.sw, 405);
  const lb = letterbox(360, 640, 320, 320);
  assert.deepEqual(lb, { dx: 70, dy: 0, dw: 180, dh: 320 });
  assert.deepEqual(versEcran({ x: 70, y: 0, w: 180, h: 320 }, lb, 360, 640), { x: 0, y: 0, w: 360, h: 640 });
});

test('découpe carrée centrée, plus grand côté × 1,2 (comme distillation/commun.py)', () => {
  assert.deepEqual(decoupeCarree({ x: 100, y: 50, w: 200, h: 100 }), { x: 80, y: -20, cote: 240 });
});

test('rectanglesDecoupe : partie hors image laissée grise', () => {
  const r = rectanglesDecoupe({ x: 80, y: -20, cote: 240 }, 1000, 1000, 120);
  assert.deepEqual(r, { sx: 80, sy: 0, sw: 240, sh: 220, dx: 0, dy: 10, dw: 120, dh: 110 });
  const bord = rectanglesDecoupe({ x: 900, y: 900, cote: 200 }, 1000, 1000, 100);
  assert.deepEqual(bord, { sx: 900, sy: 900, sw: 100, sh: 100, dx: 0, dy: 0, dw: 50, dh: 50 });
  assert.equal(rectanglesDecoupe({ x: 1200, y: 0, cote: 100 }, 1000, 1000), null);
});

test('8 secteurs distincts, de face au secteur 0, et le miroir inverse la direction', () => {
  const noms = new Set(Array.from({ length: NB_SECTEURS }, (_, k) => nomSecteur(k)));
  assert.equal(noms.size, NB_SECTEURS);
  assert.equal(nomSecteur(0), 'de face');
  assert.equal(nomSecteur(4), 'de dos');
  for (let k = 1; k < NB_SECTEURS; k++) {
    // Retournement horizontal : secteur k ↔ 8 − k, la droite devient la gauche.
    const miroir = nomSecteur(NB_SECTEURS - k);
    if (k !== 4) assert.equal(miroir.replace('gauche', 'droite'), nomSecteur(k).replace('gauche', 'droite'));
    assert.equal((cap(k) + cap(NB_SECTEURS - k)) % 360, 0);
  }
});

test('secteur = probabilité maximale', () => {
  assert.equal(secteur([0.1, 0.05, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05]), 2);
});

test('suivre : même voiture lissée, nouvelle voiture avec un nouvel identifiant', () => {
  const ids = { valeur: 1 };
  const un = [1, 0, 0, 0, 0, 0, 0, 0];
  const deux = [0, 0, 1, 0, 0, 0, 0, 0];
  let v = suivre([], [{ boite: { x: 0, y: 0, w: 100, h: 50 }, probas: un }], ids);
  assert.equal(v[0].id, 1);
  v = suivre(v, [
    { boite: { x: 5, y: 0, w: 100, h: 50 }, probas: deux },
    { boite: { x: 500, y: 0, w: 100, h: 50 }, probas: deux },
  ], ids);
  assert.equal(v[0].id, 1);
  assert.ok(Math.abs(v[0].probas[0] - 0.6) < 1e-9 && Math.abs(v[0].probas[2] - 0.4) < 1e-9);
  assert.equal(v[1].id, 2);
  assert.deepEqual(v[1].probas, deux);
});
