"""Étape 1 : recadrages de voitures tirés de COCO 2017 (train → apprentissage, val → évaluation).

Usage : python selection.py <annotations/> <sortie/> [--max-train N] [--max-val N]
Écrit <sortie>/recadrages/<id>.jpg et <sortie>/manifeste.json.
"""
import argparse
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from commun import recadrer

CATEGORIE_VOITURE = 3  # « car » dans les annotations COCO
LARGEUR_MIN = 64  # px : en dessous, l'orientation devient illisible
HAUTEUR_MIN = 48
# Surface du masque / surface de la boîte : une voiture entière remplit ~70–85 % de sa boîte ;
# bien moins, elle est cachée par un passant, un poteau ou une autre voiture.
REMPLISSAGE_MIN = 0.55


def voitures(fichier, maximum, graine):
    coco = json.loads(Path(fichier).read_text())
    images = {i["id"]: i for i in coco["images"]}
    retenues = [
        a for a in coco["annotations"]
        if a["category_id"] == CATEGORIE_VOITURE and not a["iscrowd"]
        and a["bbox"][2] >= LARGEUR_MIN and a["bbox"][3] >= HAUTEUR_MIN
        and a["area"] >= REMPLISSAGE_MIN * a["bbox"][2] * a["bbox"][3]
    ]
    print(f"{Path(fichier).name} : {len(retenues)} voitures retenues", flush=True)
    random.Random(graine).shuffle(retenues)
    retenues = retenues[:maximum]
    return [(a, images[a["image_id"]]) for a in retenues]


def telecharger(url):
    for essai in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except OSError:
            if essai == 3:
                raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("annotations")
    p.add_argument("sortie")
    p.add_argument("--max-train", type=int, default=12000)
    p.add_argument("--max-val", type=int, default=2000)
    args = p.parse_args()

    sortie = Path(args.sortie)
    (sortie / "recadrages").mkdir(parents=True, exist_ok=True)
    a_faire = [("train", *v) for v in voitures(Path(args.annotations) / "instances_train2017.json", args.max_train, 1)]
    a_faire += [("val", *v) for v in voitures(Path(args.annotations) / "instances_val2017.json", args.max_val, 2)]

    # Une image peut contenir plusieurs voitures : téléchargée une fois.
    par_image = {}
    for jeu, ann, img in a_faire:
        par_image.setdefault(img["coco_url"], []).append((jeu, ann))

    def traiter(url):
        image = telecharger(url)
        lignes = []
        for jeu, ann in par_image[url]:
            nom = f"{ann['id']}.jpg"
            recadrer(image, ann["bbox"]).save(sortie / "recadrages" / nom, quality=92)
            lignes.append({"id": ann["id"], "jeu": jeu, "fichier": nom, "image": url, "boite": ann["bbox"]})
        return lignes

    with ThreadPoolExecutor(32) as pool:
        manifeste = [l for lignes in pool.map(traiter, par_image) for l in lignes]
    (sortie / "manifeste.json").write_text(json.dumps(manifeste))
    n_train = sum(l["jeu"] == "train" for l in manifeste)
    print(f"{n_train} recadrages d'apprentissage, {len(manifeste) - n_train} d'évaluation, {len(par_image)} images")


if __name__ == "__main__":
    main()
