"""Étape 2 : azimut des recadrages selon Orient Anything (le professeur).

Usage : python etiqueter.py <donnees/> <morceau> <nb_morceaux> <sortie.json>
Professeur : ViT-L « ronormsigma1 », détourage rembg comme la démo officielle. Sur le jeu
d'évaluation, on mesure aussi ce que donneraient des variantes plus légères (ViT-S en 224 et
112 px, sans détourage) pour les comparer à l'élève.

Modèle repris d'Orient Anything (Wang et al. 2024, https://github.com/SpatialVision/Orient-Anything,
CC BY 4.0) : DINOv2 puis MLP, sorties 360 azimuts + 180 élévations + 360 rotations + 2 confiance.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import rembg
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, Dinov2Model

from commun import probas_secteurs

LOT = 8
PROFESSEUR = ("facebook/dinov2-large", 1024, "ronormsigma1/dino_weight.pt")
LEGER = ("facebook/dinov2-small", 384, "cropsmallEx03/dino_weight.pt")


class MLP(nn.Module):
    def __init__(self, entree, sortie):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(entree, sortie), nn.BatchNorm1d(sortie), nn.ReLU(inplace=True))
        self.net2 = nn.Sequential(nn.Linear(sortie, sortie), nn.BatchNorm1d(sortie))

    def forward(self, x):
        return self.net2(self.net1(x))


class OrientAnything(nn.Module):
    def __init__(self, dino, dim, poids):
        super().__init__()
        self.dinov2 = Dinov2Model.from_pretrained(dino)
        self.down_sampler = MLP(dim, 360 + 180 + 360 + 2)
        chemin = hf_hub_download("Viglong/Orient-Anything", poids)
        self.load_state_dict(torch.load(chemin, map_location="cpu"))
        self.eval()

    @torch.no_grad()
    def forward(self, pixels):
        return self.down_sampler(self.dinov2(pixel_values=pixels).last_hidden_state[:, 0])


def preparateur(dino, taille=224):
    # DINOv2 : côté court à 256/224 de la taille, puis découpe centrale.
    return AutoImageProcessor.from_pretrained(
        dino, size={"shortest_edge": round(taille * 256 / 224)}, crop_size={"height": taille, "width": taille})


def predire(modele, prep, images):
    """→ (probabilités sur 360°, confiance) par image."""
    sorties = []
    for i in range(0, len(images), LOT):
        pixels = prep(images=images[i:i + LOT], return_tensors="pt")["pixel_values"]
        sorties.append(modele(pixels))
    s = torch.cat(sorties)
    return torch.softmax(s[:, :360], -1).numpy(), torch.softmax(s[:, -2:], -1)[:, 0].numpy()


def detourer(image, session):
    """Comme background_preprocess de la démo : rembg puis objet centré, 85 % du cadre, fond noir."""
    rgba = np.array(rembg.remove(image, session=session))
    ys, xs = np.where(rgba[..., 3] > 0)
    if len(xs) < 50:
        return image  # rien de détouré : image d'origine
    objet = rgba[ys.min():ys.max(), xs.min():xs.max()]
    cote = max(objet.shape[:2])
    final = int(cote / 0.85)
    fond = np.zeros((final, final, 4), np.uint8)
    y0, x0 = (final - objet.shape[0]) // 2, (final - objet.shape[1]) // 2
    fond[y0:y0 + objet.shape[0], x0:x0 + objet.shape[1]] = objet
    return Image.fromarray(fond).convert("RGB")


def main():
    donnees, morceau, nb, sortie = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    manifeste = json.loads((donnees / "manifeste.json").read_text())[morceau::nb]
    images = [Image.open(donnees / "recadrages" / l["fichier"]).convert("RGB") for l in manifeste]
    debut = time.time()

    session = rembg.new_session("u2net")
    detourees = [detourer(im, session) for im in images]
    print(f"détourage : {time.time() - debut:.0f} s pour {len(images)} images", flush=True)

    prof = OrientAnything(*PROFESSEUR)
    prep_l = preparateur(PROFESSEUR[0])
    p360, conf = predire(prof, prep_l, detourees)
    print(f"professeur : {time.time() - debut:.0f} s", flush=True)

    resultats = [
        {"id": l["id"], "jeu": l["jeu"], "azimut": int(p.argmax()), "confiance": round(float(c), 4),
         "secteurs": [round(float(v), 4) for v in probas_secteurs(p)]}
        for l, p, c in zip(manifeste, p360, conf)
    ]

    # Évaluation : variantes sans détourage (ce qui tournerait dans l'appli).
    val = [i for i, l in enumerate(manifeste) if l["jeu"] == "val"]
    if val:
        brutes = [images[i] for i in val]
        variantes = {"L224_brut": (prof, prep_l)}
        leger = OrientAnything(*LEGER)
        variantes["S224_brut"] = (leger, preparateur(LEGER[0], 224))
        variantes["S112_brut"] = (leger, preparateur(LEGER[0], 112))
        for nom, (m, prep) in variantes.items():
            q360, _ = predire(m, prep, brutes)
            for i, q in zip(val, q360):
                resultats[i][nom] = int(q.argmax())
        print(f"variantes : {time.time() - debut:.0f} s", flush=True)

    Path(sortie).write_text(json.dumps(resultats))


if __name__ == "__main__":
    main()
