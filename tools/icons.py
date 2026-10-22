#!/usr/bin/env python3
"""Génère les icônes de la PWA (nécessite Pillow) : python3 tools/icons.py"""
import math
from pathlib import Path

from PIL import Image, ImageDraw

DOSSIER = Path(__file__).resolve().parent.parent / "icons"
ECHELLE = 4  # dessin sur-échantillonné puis réduit pour lisser les bords
JAUNE = "#ffd23f"


def tourner(points, angle, cx, cy):
    c, s = math.cos(angle), math.sin(angle)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in points]


def icone(taille):
    t = taille * ECHELLE
    img = Image.new("RGB", (t, t), "#111111")
    d = ImageDraw.Draw(img)
    # Boussole comme dans l'appli : cercle, repère « nous » en bas, voiture vue de dessus de trois-quarts.
    r = t * 0.36
    cx, cy = t / 2, t * 0.47
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#262626", outline="#3a3a3a", width=ECHELLE * 2)
    m = t * 0.05
    d.polygon([(cx, cy + r - m * 0.6), (cx - m, cy + r + m), (cx + m, cy + r + m)], fill="#ffffff")
    w, h = r * 0.62, r * 1.3
    angle = math.radians(180 - 45)  # avant vers nous et vers la droite
    caisse = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    d.polygon(tourner(caisse, angle, cx, cy), fill=JAUNE)
    pare_brise = [(-w * 0.36, -h * 0.3), (w * 0.36, -h * 0.3), (w * 0.36, -h * 0.14), (-w * 0.36, -h * 0.14)]
    d.polygon(tourner(pare_brise, angle, cx, cy), fill="#222222")
    return img.resize((taille, taille), Image.LANCZOS)


DOSSIER.mkdir(exist_ok=True)
for nom, taille in (("icon-192.png", 192), ("icon-512.png", 512), ("apple-touch-icon.png", 180)):
    icone(taille).save(DOSSIER / nom, optimize=True)
    print(DOSSIER / nom)
