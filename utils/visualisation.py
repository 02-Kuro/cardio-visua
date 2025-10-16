#!/usr/bin/env python
# coding: utf-8

import os
import nibabel as nib
import numpy as np
import torch

# 📁 Dossiers
seg_dir = "C:/Users/kuro/Desktop/3dUnet/predictions/"
cam_dir = "C:/Users/kuro/Desktop/3dUnet/3d grad cam/"
save_dir = "C:/Users/kuro/Desktop/3dUnet/3d_grad_cam_rgb/"

# 📦 Création du dossier de sortie si besoin
os.makedirs(save_dir, exist_ok=True)

# 📜 Fichiers disponibles dans Grad-CAM
cam_files = [f for f in os.listdir(cam_dir) if f.endswith(".nii.gz")]

# 🔁 Traitement
for file in cam_files:
    try:
        # --- Chargement ---
        cam_path = os.path.join(cam_dir, file)
        seg_path = os.path.join(seg_dir, file.replace("_gradcam", ".nii_pred"))
        
        cam_data = nib.load(cam_path)
        seg_data = nib.load(seg_path)
        affine = seg_data.affine

        cam_tensor = torch.tensor(cam_data.get_fdata(), dtype=torch.float32, device="cuda")
        seg_tensor = torch.tensor(seg_data.get_fdata(), dtype=torch.float32, device="cuda")

        # --- Normalisation segmentation (0-1) ---
        seg_tensor = (seg_tensor - seg_tensor.min()) / (seg_tensor.max() - seg_tensor.min() + 1e-8)

        # --- Construction RGB-like ---
        D, H, W = cam_tensor.shape
        rgb_tensor = torch.zeros((3, D, H, W), dtype=torch.float32, device="cuda")

        rgb_tensor[0] = cam_tensor          # 🔴 Rouge : Grad-CAM
        rgb_tensor[1] = 0.0                 # 🟢 Vert vide
        rgb_tensor[2] = seg_tensor          # 🔵 Bleu : segmentation

        rgb_tensor = rgb_tensor.clamp(0, 1)

        # --- Rearrangement pour Nibabel : (D, H, W, 3)
        rgb_np = rgb_tensor.permute(1, 2, 3, 0).cpu().numpy()

        # --- Sauvegarde NIfTI ---
        rgb_nii = nib.Nifti1Image(rgb_np, affine)
        save_path = os.path.join(save_dir, file.replace("_gradcam", "_rgb"))
        nib.save(rgb_nii, save_path)

        print(f"✅ RGB sauvegardé : {save_path}")

    except Exception as e:
        print(f"❌ Erreur avec {file} : {str(e)}")


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 📂 Chemins
image_path = "C:/Users/kuro/Desktop/3dUnet/pfe data/ACDC_dataSet/testing/patient140/patient140_frame01.nii.gz"
#pred_path  = "C:/Users/kuro/Desktop/3dUnet/predictions/patient140_frame01_pred.nii.gz"
pred_path  = "C:/Users/kuro/Desktop/3dUnet/predictions/patient140_frame01_pred.nii.gz"
gt_path    = "C:/Users/kuro/Desktop/3dUnet/pfe data/ACDC_dataSet/testing/patient140/patient140_frame01_gt.nii.gz"
# 📥 Chargement
img = nib.load(image_path).get_fdata()
pred = nib.load(pred_path).get_fdata()
gt = nib.load(gt_path).get_fdata()

# 📐 Normalisation image pour l'affichage
img = (img - img.min()) / (img.max() - img.min())

# 🔍 Affichage comparatif pour quelques slices
num_slices = img.shape[2]
for i in range(0, num_slices, 10):  # toutes les 10 slices
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img[:, :, i], cmap='gray')
    axs[0].set_title(f'IRM - Slice {i}')
    
    axs[1].imshow(pred[:, :, i], cmap='jet', vmin=0, vmax=3)
    axs[1].set_title('Segmentation Prédite')

    axs[2].imshow(gt[:, :, i], cmap='jet', vmin=0, vmax=3)
    axs[2].set_title('Ground Truth')
    
    for ax in axs:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


import nibabel as nib
import numpy as np
import pyvista as pv
from skimage import measure
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

def visualize_gradcam_segmented_heart(
    gradcam_path,
    seg_path,
    threshold=0.1,
    label_values=(1, 2, 3),
    colormap='hot'
):
    # --- Load Grad-CAM and get voxel spacing ---
    gradcam_nii = nib.load(gradcam_path)
    gradcam = gradcam_nii.get_fdata()
    spacing = gradcam_nii.header.get_zooms()  # (z, y, x) or (D, H, W)
    # Convert to (x, y, z) order for marching_cubes if needed
    spacing = spacing[::-1]

    # --- Load segmentation and resize to Grad-CAM shape ---
    seg = nib.load(seg_path).get_fdata()
    seg_resized = resize(seg, gradcam.shape, order=0, preserve_range=True, anti_aliasing=False)

    # --- Normalize Grad-CAM ---
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)

    # --- Mask and crop to heart region ---
    heart_mask = np.isin(seg_resized, label_values)
    if not np.any(heart_mask):
        print("❌ No heart region found.")
        return

    coords = np.array(np.where(heart_mask))
    minz, miny, minx = coords.min(axis=1)
    maxz, maxy, maxx = coords.max(axis=1) + 1

    gradcam_cropped = gradcam[minz:maxz, miny:maxy, minx:maxx]
    heart_mask_cropped = heart_mask[minz:maxz, miny:maxy, minx:maxx]

    # --- Apply mask and smoothing ---
    masked_gradcam = gradcam_cropped * heart_mask_cropped
    masked_gradcam[masked_gradcam < threshold] = 0
    masked_gradcam = gaussian_filter(masked_gradcam, sigma=1)

    if np.max(masked_gradcam) == 0:
        print("⚠️ No significant activation detected after thresholding.")
        return

    # --- Mesh extraction with correct spacing ---
    # Calculate cropped spacing (in case you want to adjust for crop, but usually spacing is unchanged)
    verts, faces, values, _ = measure.marching_cubes(
        masked_gradcam, 
        level=threshold, 
        spacing=spacing
    )

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32).flatten()
    mesh = pv.PolyData(verts, faces_pv)
    mesh.point_data["Activation"] = values

    # --- Visualization ---
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="Activation",
        cmap=colormap,
        show_edges=False,
        opacity=1.0,
        scalar_bar_args={"title": "Grad-CAM Activation"}
    )
    plotter.background_color = "white"
    plotter.show(title="🔥 Grad-CAM activation on heart (corrected proportions)")

# --- Example call ---
visualize_gradcam_segmented_heart(
    gradcam_path="C:/Users/kuro/Desktop/3dUnet/3d grad cam/patient101_frame01_pred.nii.gz",
    seg_path="C:/Users/kuro/Desktop/3dUnet/pfe data/testing/labels/patient101_frame01_gt.nii.gz",
    threshold=0.2,
    label_values=(1, 2, 3),
    colormap="hot"
)




