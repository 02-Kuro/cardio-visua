#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

# Répertoires contenant les fichiers .cfg
train_infos_dir = r"C:\Users\kuro\Desktop\3dUnet\pfe data\training\infos"
test_infos_dir = r"C:\Users\kuro\Desktop\3dUnet\pfe data\testing\patient_infos"

# Fichier de sortie
output_csv = r"C:\Users\kuro\Desktop\3dUnet\info.csv"

# Mapping groupe → label
group_to_label = {
    "NOR": 0,
    "MINF": 1,
    "DCM": 2,
    "HCM": 3,
    "RV": 4
}

def parse_info_file(filepath):
    data = {
        "ED": None,
        "ES": None,
        "Height": None,
        "NbFrame": None,
        "Weight": None,
        "label": -1  # Valeur par défaut si groupe inconnu
    }
    with open(filepath, 'r') as file:
        for line in file:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "ed":
                data["ED"] = int(value)
            elif key == "es":
                data["ES"] = int(value)
            elif key == "height":
                data["Height"] = float(value)
            elif key == "nbframe":
                data["NbFrame"] = int(value)
            elif key == "weight":
                data["Weight"] = float(value)
            elif key == "group":
                group = value.upper()
                if group in group_to_label:
                    data["label"] = group_to_label[group]
                else:
                    print(f"[⚠️ AVERTISSEMENT] Groupe inconnu ou invalide dans {filepath} : '{group}'")

    return data

def process_directory(dir_path):
    rows = []
    for filename in sorted(os.listdir(dir_path)):
        if filename.startswith("info_patient") and filename.endswith(".cfg"):
            patient_id = filename.replace("info_", "").replace(".cfg", "")  # ex: patient101
            full_path = os.path.join(dir_path, filename)
            info = parse_info_file(full_path)
            info["patient_id"] = patient_id
            rows.append(info)
    return rows

# Traitement des deux dossiers
train_rows = process_directory(train_infos_dir)
test_rows = process_directory(test_infos_dir)

# Fusionner les deux
all_rows = train_rows + test_rows

# Convertir en DataFrame avec ordre explicite
df = pd.DataFrame(all_rows)
df = df[["patient_id", "ED", "ES", "Height", "NbFrame", "Weight", "label"]]

# Export
df.to_csv(output_csv, index=False)
print(f"✅ Fichier CSV généré : {output_csv}")
print(f"📦 Patients total traités : {len(df)}")

# Vérifier les entrées avec un label invalide
invalid_rows = df[df["label"] == -1]
if not invalid_rows.empty:
    print(f"\n⚠️ {len(invalid_rows)} patients avec un label invalide (-1) trouvés :")
    print(invalid_rows[["patient_id", "label"]])
else:
    print("✅ Tous les patients ont un label valide.")

# Exemple d'aperçu
print("\n🔍 Aperçu du CSV :")
print(df.head())


import pandas as pd

df = pd.read_csv("info.csv")
print(df.head())
print(df['label'].value_counts())  # Pour voir la distribution des classes


# === Imports ===
import os
import torchio as tio

# === Définir la transformation (rotation, flip, bruit, gamma, etc.) ===
transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),  # Flip aléatoire sur chaque axe
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=0, center='image'),  # Rotation légère
    tio.RandomGamma(log_gamma=(0.7, 1.5), p=0.5),  # Modification gamma (≈ brightness/contrast)
    tio.RandomNoise(mean=0.0, std=0.03, p=0.5),  # Bruit gaussien
    tio.RandomBlur(std=(0, 1.5), p=0.3),  # Flou gaussien
    tio.RandomBiasField(coefficients=0.5, p=0.3),  # Simulation de distorsion IRM
    tio.RandomElasticDeformation(num_control_points=7, max_displacement=5, p=0.3),  # Déformation douce
])

# === Dossiers ===
image_dir = 'C:/Users/kuro/Desktop/3dUnet/pfe data/training/images'
label_dir = 'C:/Users/kuro/Desktop/3dUnet/pfe data/training/labels'
output_image_dir = 'C:/Users/kuro/Desktop/3dUnet/pfe data/training/augmented_images'
output_label_dir = 'C:/Users/kuro/Desktop/3dUnet/pfe data/training/augmented_labels'

# Créer les dossiers de sortie s’ils n’existent pas
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# === Application de l’augmentation sur chaque patient ===
for filename in os.listdir(image_dir):
    if filename.endswith('.nii.gz'):
        image_path = os.path.join(image_dir, filename)

        # Adapter le nom du label : ajouter '_gt'
        label_filename = filename.replace('.nii.gz', '_gt.nii.gz')
        label_path = os.path.join(label_dir, label_filename)

        # Vérifier que le label existe
        if not os.path.exists(label_path):
            print(f"Label manquant pour {filename}, on saute.")
            continue

        # Créer le sujet TorchIO avec image + label
        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path)
        )

        # Appliquer 5 augmentations différentes par patient
        for i in range(5):
            transformed = transform(subject)
            img_aug = transformed['image']
            lbl_aug = transformed['label']

            # Sauvegarde avec suffixe _aug{i}
            img_name = f'{filename.replace(".nii.gz", "")}_aug{i}.nii.gz'
            lbl_name = f'{filename.replace(".nii.gz", "")}_aug{i}.nii.gz'

            img_aug.save(os.path.join(output_image_dir, img_name))
            lbl_aug.save(os.path.join(output_label_dir, lbl_name))

        print(f"✅ {filename} : 5 augmentations sauvegardées.")




