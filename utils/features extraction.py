#!/usr/bin/env python
# coding: utf-8

# Extract Diagnosis Labels from multiple ACDC folders
import os
import pandas as pd

# Liste des dossiers à scanner (tu peux en ajouter d'autres ici)
acdc_folders = [
    "/home/kuro-02/Bureau/ACDC_dataSet/training",
    "/home/kuro-02/Bureau/ACDC_dataSet/testing"  # <-- ajoute ce dossier s'il existe
]

# Map diagnosis groups to integer labels
group_mapping = {
    "NOR": 0,
    "MINF": 1,
    "DCM": 2,
    "HCM": 3,
    "RV": 4
}

data = []

for acdc_root in acdc_folders:
    for patient_folder in os.listdir(acdc_root):
        full_path = os.path.join(acdc_root, patient_folder)
        if not os.path.isdir(full_path):
            continue

        cfg_path = os.path.join(full_path, "Info.cfg")
        if not os.path.isfile(cfg_path):
            continue

        group_label = None
        with open(cfg_path, "r") as f:
            for line in f:
                if line.startswith("Group:"):
                    group_label = line.split(":")[1].strip()
                    break

        if group_label is None or group_label not in group_mapping:
            print(f"Skipping {patient_folder} — unknown or missing group: {group_label}")
            continue

        data.append({
            "patient_id": patient_folder,
            "diagnosis": group_mapping[group_label]
        })
# Save to CSV (trié par patient_id pour plus de lisibilité)
df = pd.DataFrame(data)
df = df.sort_values(by="patient_id")  # <- tri ici
df.to_csv("acdc_diagnosis_labels.csv", index=False)
print("✅ CSV saved: acdc_diagnosis_labels.csv")


import numpy as np
import nibabel as nib

path = "/home/kuro-02/Bureau/ACDC_dataSet/testing/patient101/patient101_frame14_gt.nii.gz"
img = nib.load(path)
array = img.get_fdata()

print("Valeurs uniques dans le volume :", np.unique(array))


#extraire les features sur les volumes (volumes)
import os
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

# Dossier contenant les volumes segmentés (labels)
label_dir = "/home/kuro-02/datasets/labels"  # à adapter
output_csv = "volume_features.csv"

# Fonction pour calculer le volume en mm³
def compute_volume(mask, voxel_volume):
    return np.sum(mask) * voxel_volume

# Liste pour stocker les données
features = []

for fname in sorted(os.listdir(label_dir)):
    if not fname.endswith("_gt.nii"):
        continue

    path = os.path.join(label_dir, fname)
    patient_id = fname.split("_")[0]

    img = nib.load(path)
    data = img.get_fdata()
    voxel_dims = img.header.get_zooms()
    voxel_volume = np.prod(voxel_dims)  # en mm³

    # Classes dans ACDC : 1=VD, 2=Myocarde, 3=VG
    vol_vd = compute_volume(data == 1, voxel_volume)
    vol_myo = compute_volume(data == 2, voxel_volume)
    vol_vg = compute_volume(data == 3, voxel_volume)

    total_heart_volume = vol_vd + vol_myo + vol_vg
    ratio_vg_vd = vol_vg / vol_vd if vol_vd != 0 else 0
    ratio_myo_vg = vol_myo / vol_vg if vol_vg != 0 else 0

    features.append({
        "patient_id": patient_id,
        "volume_vd": vol_vd,
        "volume_myo": vol_myo,
        "volume_vg": vol_vg,
        "total_heart_volume": total_heart_volume,
        "ratio_vg_vd": ratio_vg_vd,
        "ratio_myo_vg": ratio_myo_vg
    })

# Export en CSV
df = pd.DataFrame(features)
df.to_csv(output_csv, index=False)
print(f"✅ Features saved to {output_csv}")

