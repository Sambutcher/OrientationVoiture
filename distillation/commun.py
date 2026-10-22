"""Réglages et fonctions partagés par les étapes de la distillation.

Le recadrage doit rester identique à celui de l'appli (decoupeCarree de js/orientation.js) :
carré centré sur la boîte de la voiture, côté = plus grand côté × MARGE, gris hors de l'image.
"""
import numpy as np
from PIL import Image

MARGE = 1.2
GRIS = (128, 128, 128)
TAILLE_STOCKAGE = 256  # recadrages enregistrés à cette taille
TAILLE_ELEVE = 128  # entrée du modèle élève
SECTEURS = 8  # secteurs d'azimut de 45°, le secteur 0 centré sur 0°


def decoupe_carree(boite):
    x, y, w, h = boite
    cote = max(w, h) * MARGE
    return (x + w / 2 - cote / 2, y + h / 2 - cote / 2, cote)


def recadrer(image, boite, taille=TAILLE_STOCKAGE):
    gauche, haut, cote = decoupe_carree(boite)
    l, t, c = round(gauche), round(haut), round(cote)
    fond = Image.new("RGB", (c, c), GRIS)
    morceau = image.crop((max(0, l), max(0, t), min(image.width, l + c), min(image.height, t + c)))
    fond.paste(morceau, (max(0, -l), max(0, -t)))
    return fond.resize((taille, taille), Image.BICUBIC)


def secteur_de_degre():
    """Secteur de chacun des 360 degrés : 338–22° → 0, 23–67° → 1, etc."""
    return (np.arange(360) + 22) // 45 % SECTEURS


def probas_secteurs(probas_360):
    """Probabilités par degré (… × 360) → probabilités par secteur (… × 8)."""
    s = secteur_de_degre()
    return np.stack([probas_360[..., s == k].sum(-1) for k in range(SECTEURS)], -1)


def secteur_miroir(k):
    """Image retournée horizontalement : l'azimut a devient −a."""
    return (SECTEURS - k) % SECTEURS
