"""Étape 3 : l'élève (MobileNetV3-Small, 128 px, 8 secteurs) apprend à imiter le professeur.

Usage : python entrainer.py <donnees/> <etiquettes/> <sortie/> [--epoques N]
Écrit <sortie>/eleve.onnx, rapport.md, planche.jpg (recadrages rangés par secteur du professeur)
et reference/ (entrées et sorties attendues pour vérifier la conversion TF.js).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from commun import SECTEURS, TAILLE_ELEVE, secteur_de_degre, secteur_miroir

CONFIANCE_MIN = 0.5  # en dessous, le professeur ne sait pas lui-même : écarté
MOYENNE = [0.485, 0.456, 0.406]
ECART = [0.229, 0.224, 0.225]
MIROIR = [secteur_miroir(k) for k in range(SECTEURS)]


class Recadrages(Dataset):
    def __init__(self, dossier, lignes, apprentissage):
        self.dossier, self.lignes, self.apprentissage = dossier, lignes, apprentissage
        if apprentissage:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(TAILLE_ELEVE, scale=(0.75, 1), ratio=(0.85, 1.18)),
                transforms.ColorJitter(0.35, 0.35, 0.3, 0.05),
                transforms.RandomGrayscale(0.05),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([transforms.Resize(TAILLE_ELEVE), transforms.ToTensor()])

    def __len__(self):
        return len(self.lignes)

    def __getitem__(self, i):
        l = self.lignes[i]
        image = Image.open(self.dossier / l["fichier"]).convert("RGB")
        cible = torch.tensor(l["secteurs"])
        if self.apprentissage and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            cible = cible[MIROIR]
        return self.tf(image), cible / cible.sum(), torch.tensor(l["confiance"])


class Exporte(nn.Module):
    """Modèle livré à l'appli : image RVB 0–1 → probabilités des 8 secteurs."""

    def __init__(self, reseau):
        super().__init__()
        self.reseau = reseau
        self.register_buffer("moyenne", torch.tensor(MOYENNE).view(1, 3, 1, 1))
        self.register_buffer("ecart", torch.tensor(ECART).view(1, 3, 1, 1))

    def forward(self, image):
        return torch.softmax(self.reseau((image - self.moyenne) / self.ecart), -1)


def creer_reseau():
    r = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    r.classifier[3] = nn.Linear(r.classifier[3].in_features, SECTEURS)
    return r


def ecart_secteurs(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % SECTEURS
    return np.minimum(d, SECTEURS - d)


def scores(pred, verite):
    e = ecart_secteurs(pred, verite)
    return f"{(e == 0).mean():.1%} exact, {(e <= 1).mean():.1%} à un secteur près"


def planche(dossier, lignes, chemin, par_secteur=12, cote=72):
    img = Image.new("RGB", (cote * (par_secteur + 1), cote * SECTEURS), "white")
    d = ImageDraw.Draw(img)
    for k in range(SECTEURS):
        d.text((6, k * cote + cote // 2 - 6), f"{k} · {k * 45}°", fill="black")
        choix = [l for l in lignes if int(np.argmax(l["secteurs"])) == k][:par_secteur]
        for j, l in enumerate(choix):
            vignette = Image.open(dossier / l["fichier"]).convert("RGB").resize((cote - 4, cote - 4))
            img.paste(vignette, ((j + 1) * cote + 2, k * cote + 2))
    img.save(chemin, quality=88)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("donnees")
    p.add_argument("etiquettes")
    p.add_argument("sortie")
    p.add_argument("--epoques", type=int, default=25)
    args = p.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    donnees, sortie = Path(args.donnees), Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    manifeste = {l["id"]: l for l in json.loads((donnees / "manifeste.json").read_text())}
    lignes = [{**manifeste[e["id"]], **e} for f in sorted(Path(args.etiquettes).glob("*.json"))
              for e in json.loads(f.read_text())]
    sures = [l for l in lignes if l["confiance"] >= CONFIANCE_MIN]
    train = [l for l in sures if l["jeu"] == "train"]
    val = [l for l in sures if l["jeu"] == "val"]
    dossier = donnees / "recadrages"
    rapport = [
        "# Distillation d'Orient Anything en 8 secteurs d'azimut", "",
        f"- Recadrages étiquetés : {len(lignes)} ; confiance du professeur ≥ {CONFIANCE_MIN} : {len(sures)} "
        f"({len(train)} apprentissage, {len(val)} évaluation)",
    ]
    planche(dossier, val, sortie / "planche.jpg")

    reseau = creer_reseau()
    charge = DataLoader(Recadrages(dossier, train, True), batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    charge_val = DataLoader(Recadrages(dossier, val, False), batch_size=256, num_workers=4)
    opt = torch.optim.AdamW(reseau.parameters(), lr=2e-3, weight_decay=1e-4)
    planif = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.epoques * len(charge))
    moyenne, ecart = torch.tensor(MOYENNE).view(1, 3, 1, 1), torch.tensor(ECART).view(1, 3, 1, 1)

    def normaliser(x):
        return (x - moyenne) / ecart

    def evaluer():
        reseau.eval()
        with torch.no_grad():
            pred = torch.cat([reseau(normaliser(x)).argmax(-1) for x, _, _ in charge_val])
        reseau.train()
        return pred.numpy()

    debut = time.time()
    for ep in range(args.epoques):
        total = 0.0
        for x, cible, conf in charge:
            log_p = torch.log_softmax(reseau(normaliser(x)), -1)
            perte = (-(cible * log_p).sum(-1) * conf).mean()
            opt.zero_grad()
            perte.backward()
            opt.step()
            planif.step()
            total += perte.item()
        verite = [int(np.argmax(l["secteurs"])) for l in val]
        print(f"époque {ep + 1} : perte {total / len(charge):.3f}, élève {scores(evaluer(), verite)} "
              f"({time.time() - debut:.0f} s)", flush=True)

    verite = [int(np.argmax(l["secteurs"])) for l in val]
    secteur = secteur_de_degre()
    rapport += [f"- Apprentissage : {args.epoques} époques, {time.time() - debut:.0f} s", "",
                "Accord avec le professeur (ViT-L détouré) sur les voitures de COCO val2017 :", ""]
    rapport.append(f"- **Élève MobileNetV3-Small 128 px** : {scores(evaluer(), verite)}")
    for nom, titre in [("L224_brut", "ViT-L 224 px sans détourage"), ("S224_brut", "ViT-S 224 px sans détourage"),
                       ("S112_brut", "ViT-S 112 px sans détourage")]:
        if all(nom in l for l in val):
            rapport.append(f"- {titre} : {scores([secteur[l[nom]] for l in val], verite)}")
    repartition = np.bincount(verite, minlength=SECTEURS)
    rapport += ["", f"Répartition des secteurs (évaluation) : {repartition.tolist()}"]

    reseau.eval()
    final = Exporte(reseau).eval()
    torch.onnx.export(final, torch.zeros(1, 3, TAILLE_ELEVE, TAILLE_ELEVE), sortie / "eleve.onnx",
                      input_names=["image"], output_names=["secteurs"], opset_version=13)
    torch.save(reseau.state_dict(), sortie / "eleve.pt")

    # Référence pour la conversion TF.js : 16 recadrages d'évaluation, entrée NHWC 0–1.
    ref = sortie / "reference"
    ref.mkdir(exist_ok=True)
    x = torch.stack([Recadrages(dossier, val[:16], False)[i][0] for i in range(min(16, len(val)))])
    with torch.no_grad():
        attendu = final(x)
    (ref / "entree.bin").write_bytes(x.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32).tobytes())
    (ref / "sortie.json").write_text(json.dumps({"forme": list(x.permute(0, 2, 3, 1).shape),
                                                 "secteurs": attendu.numpy().round(5).tolist()}))
    (sortie / "rapport.md").write_text("\n".join(rapport) + "\n")
    print("\n".join(rapport))


if __name__ == "__main__":
    main()
