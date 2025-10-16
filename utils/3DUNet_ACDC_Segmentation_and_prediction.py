#!/usr/bin/env python
# coding: utf-8

# 📦 Imports
import os
import glob
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn  # <-- L'import manquant
import torch.optim as optim
import torchio as tio
from torch.utils.data import Dataset, random_split, DataLoader


# 📁 Configuration des chemins
image_dir = Path('C:/Users/kuro/Desktop/3dUnet/pfe data/training/images') # Remplace par le chemin correct vers ton dossier d'images
label_dir = Path('C:/Users/kuro/Desktop/3dUnet/pfe data/training/labels')# Remplace par le chemin correct vers ton dossier de labels

# 📦 Chargement des sujets
subjects = []

# On parcourt toutes les images et labels
for image_path in sorted(image_dir.glob('*.nii.gz')):  # On suppose que tous les fichiers sont en .nii.gz
    label_path = label_dir / image_path.name.replace('.nii.gz', '_gt.nii.gz')  # Remplace le nom de l'image par le label

    if label_path.exists():
        # On charge l'image et son label correspondant
        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path)
        )
        subjects.append(subject)
    else:
        print(f"⚠️ Aucun label trouvé pour {image_path.name}")

print(f"✅ {len(subjects)} sujets chargés.")


transform = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),  # Normalise les intensités
    tio.RandomFlip(axes=(0, 1, 2), p=0.5),     # Flip aléatoire sur les 3 axes
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5),  # Petite transformation géométrique
    tio.Resize((64, 64, 64))  # Redimensionne tous les volumes à la même taille (utile pour les réseaux)
])
# 📦 Création du dataset TorchIO avec transformations
dataset = tio.SubjectsDataset(subjects, transform=transform)


from torchio.data import SubjectsLoader
# 🔀 Split train/val (facultatif mais recommandé)
train_ratio = 0.8
n_train = int(len(dataset) * train_ratio)
n_val = len(dataset) - n_train
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

# 🔄 DataLoaders pour entraînement et validation
train_loader = SubjectsLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = SubjectsLoader(val_dataset, batch_size=2)
# 👀 Affichage d’un batch pour vérifier (facultatif)
batch = next(iter(train_loader))
print("✅ Batch shape - image :", batch['image'][tio.DATA].shape)
print("✅ Batch shape - label :", batch['label'][tio.DATA].shape)


# 👀 Affichage d'une coupe d'un volume et de son label
sample = train_dataset[0]
image_tensor = sample['image'].data[0]
label_tensor = sample['label'].data[0]
slice_index = image_tensor.shape[2] // 2

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_tensor[:, :, slice_index], cmap='gray')
plt.title('Image')

plt.subplot(1, 2, 2)
plt.imshow(label_tensor[:, :, slice_index])
plt.title('Label')
plt.show()


# 🧠 Modèle 3D U-Net (version corrigée)
# -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, init_features=32):
        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 4, features * 8)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 4 * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 2 * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)


    

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )


def dice_score(preds, targets, num_classes=4, epsilon=1e-6):
    # pred et target sont des tenseurs (B, D, H, W)
    dice = 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice += (2. * intersection + epsilon) / (union + epsilon)

    return dice / num_classes


import torch

print("CUDA disponible :", torch.cuda.is_available())
print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU détecté")
print("Version CUDA (PyTorch):", torch.version.cuda)
print("Version PyTorch:", torch.__version__)
print("Device utilisé :", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# 🚀 Entraînement
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
#intialisation des valeurs
train_losses = []
val_losses = []
dice_scores = []

best_val_loss = float('inf')
for epoch in range(50):
    # Entraînement
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/100"):
        inputs = batch["image"][tio.DATA].to(device)
        targets = batch["label"][tio.DATA].to(device).long().squeeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"][tio.DATA].to(device)
            targets = batch["label"][tio.DATA].to(device).long().squeeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_dice += dice_score(preds, targets)

    # Affichage des résultats
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    dice_scores.append(avg_val_dice)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Dice Score: {avg_val_dice:.4f}")

    # Sauvegarde du meilleur modèle
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model02.pth")


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.title("Loss")
plt.legend()
plt.subplot(1, 3, 2)
plt.plot([float(ds) for ds in dice_scores])
plt.title("Dice score (val)")
plt.tight_layout()
plt.show()


# 👀 Visualisation (corrigée)
# -------------------------------
sample = next(iter(val_loader))
model.eval()
with torch.no_grad():
    prediction = model(sample["image"][tio.DATA].to(device)).cpu()

image = sample["image"][tio.DATA][0, 0, :, :, 32].cpu().numpy()
label = sample["label"][tio.DATA][0, 0, :, :, 32].cpu().numpy()
pred = torch.argmax(prediction[0], dim=0)[:, :, 32].cpu().numpy()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Image')

plt.subplot(1, 3, 2)
plt.imshow(label)
plt.title('Vérité terrain')

plt.subplot(1, 3, 3)
plt.imshow(pred)
plt.title('Prédiction')
plt.show()


import os
import torchio as tio

image_dir = r"C:\Users\kuro\Desktop\3dUnet\pfe data\testing\images"
label_dir = r"C:\Users\kuro\Desktop\3dUnet\pfe data\testing\labels"

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

test_subjects = []

for image_file in image_files:
    # Générer le nom de fichier attendu du label (en supposant suffixe _gt)
    base_name = image_file.replace('.nii.gz', '').replace('.nii', '')
    label_candidates = [f for f in label_files if base_name in f and '_gt' in f]
    
    if label_candidates:
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_candidates[0])
        
        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path)
        )
        test_subjects.append(subject)
    else:
        print(f"⚠️ Aucun label trouvé pour : {image_file}")

# Transformation (si tu as défini une variable `transform` avant)
test_dataset = tio.SubjectsDataset(test_subjects, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1)

print(f"✅ {len(test_subjects)} sujets chargés pour le test.")


# 🔁 Chargement du modèle entraîné avant inférence
model = UNet3D(in_channels=1, out_channels=4)  # Vérifie bien ces arguments

# D'abord charger les poids, puis envoyer sur le bon device
state_dict = torch.load(r"C:\Users\kuro\Desktop\3dUnet\best_model02.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)  # Très important !

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 🔍 Inférence sur les données de test
def infer(model, loader):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            image = batch['image'][tio.DATA].to(device)
            label = batch['label'][tio.DATA].squeeze(1).cpu().numpy()
            output = model(image)
            pred = torch.argmax(output, dim=1).cpu().numpy()

            preds.extend(pred)
            gts.extend(label)
    return preds, gts

predictions, ground_truths = infer(model, test_loader)


import numpy as np
import matplotlib.pyplot as plt

# Passe le modèle en mode évaluation
model.eval()

# On prend un batch du DataLoader de validation
batch = next(iter(val_loader))
images = batch["image"][tio.DATA].to(device)  # (B, 1, D, H, W)
labels = batch["label"][tio.DATA].to(device).long().squeeze(1)  # (B, D, H, W)

with torch.no_grad():
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)  # (B, D, H, W)

# Affichage + Dice pour chaque volume du batch
num_classes = 4  # à adapter selon ton jeu de données

def dice_per_class(pred, target, num_classes=4, epsilon=1e-6):
    dices = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = pred_cls.sum().item() + target_cls.sum().item()
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dices.append(dice)
    return dices

for i in range(preds.shape[0]):
    # Affichage d'une coupe axiale centrale
    slice_idx = preds.shape[1] // 2
    img = images[i, 0, :, :, slice_idx].cpu().numpy()
    gt = labels[i, :, :, slice_idx].cpu().numpy()
    pred = preds[i, :, :, slice_idx].cpu().numpy()

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image')

    plt.subplot(1, 3, 2)
    plt.imshow(gt)
    plt.title('Label (Vérité terrain)')

    plt.subplot(1, 3, 3)
    plt.imshow(pred)
    plt.title('Prédiction U-Net')
    plt.show()

    # Dice par classe et global
    dices = dice_per_class(preds[i], labels[i], num_classes=num_classes)
    for idx, d in enumerate(dices):
        print(f"Dice classe {idx} : {d:.3f}")
    print(f"Dice global (moyenne) : {np.mean(dices):.3f}")
    print("------")


# 📍 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚙️ Initialisation + chargement du modèle
model = UNet3D(in_channels=1, out_channels=4)
model_path = "C:/Users/kuro/Desktop/3dUnet/best_model02.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


import os
import torch
import nibabel as nib
import numpy as np
from skimage.transform import resize

# 📦 Fonction de prédiction pour un seul volume
def predict_single_volume(image_path, output_path=None):
    print(f"\n🔍 Traitement de : {image_path}")
    
    # 📥 Chargement
    img_nii = nib.load(image_path)
    img = img_nii.get_fdata()
    affine = img_nii.affine
    original_shape = img.shape
    print(f"📐 Taille originale : {original_shape} | Min: {img.min():.2f} | Max: {img.max():.2f}")

    # 🧪 Normalisation et redimensionnement
    img = np.clip(img, 0, np.percentile(img, 99.5))
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    img_resized = resize(img, (64, 64, 64), preserve_range=True)
    print(f"📏 Redimensionné à : {img_resized.shape}")
    print(f"📊 Moyenne : {img_resized.mean():.4f} | Écart-type : {img_resized.std():.4f}")

    # 🎯 Préparation du tenseur
    input_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    print(f"🧠 Tenseur shape : {input_tensor.shape} | Type : {input_tensor.dtype} | Device : {input_tensor.device}")

    # 🔮 Prédiction
    with torch.no_grad():
        output = model(input_tensor)
        print(f"📤 Sortie modèle : {output.shape}")
        print(f"📈 Logits - Min: {output.min().item():.4f} | Max: {output.max().item():.4f} | Mean: {output.mean().item():.4f}")
        
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        print(f"📊 Valeurs uniques dans prédiction : {np.unique(pred)}")

    # 🔁 Resize inverse
    pred_resized = resize(pred, original_shape, order=0, preserve_range=True).astype(np.uint8)

    # 💾 Sauvegarde
    if output_path:
        pred_nii = nib.Nifti1Image(pred_resized, affine)
        nib.save(pred_nii, output_path)
        print(f"✅ Sauvegarde : {output_path}")
    else:
        print("✅ Prédiction terminée (non sauvegardée)")

    return pred_resized

# 📁 Chemins
input_folder = "C:/Users/kuro/Desktop/3dUnet/pfe data/testing/images"
output_folder = "C:/Users/kuro/Desktop/3dUnet/predictions"

# 📂 Créer le dossier de sortie s’il n’existe pas
os.makedirs(output_folder, exist_ok=True)

# 🔁 Boucle sur tous les fichiers du dossier
for filename in os.listdir(input_folder):
    if filename.endswith(".nii.gz"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".nii.gz", "_pred.nii.gz"))

        try:
            predict_single_volume(input_path, output_path)
        except Exception as e:
            print(f"❌ Erreur pour {filename} : {e}")


import nibabel as nib
import numpy as np

# 📍 Fichiers
seg_path = "C:/Users/kuro/Desktop/3dUnet/predictions/patient101_frame01_pred.nii.gz"
gradcam_path = "C:/Users/kuro/Desktop/3dUnet/3d grad cam/patient101_frame01_pred.nii.gz"

# 📥 Chargement
seg = nib.load(seg_path).get_fdata().astype(np.uint8)
cam = nib.load(gradcam_path).get_fdata().astype(np.float32)

# 🧽 Seuillage optionnel du Grad-CAM (pour montrer uniquement les zones significatives)
cam_thresh = cam.copy()
cam_thresh[cam_thresh < 0.5] = 0  # Ajustable


# ... (chargement des données inchangé)

# Lissage
from scipy.ndimage import gaussian_filter
seg_smooth = gaussian_filter(seg.astype(float), sigma=1)

# Vérifiez la plage de valeurs
print(seg_smooth.min(), seg_smooth.max())  # Devrait être entre 0 et 1

# Extraction de surface sur le volume lissé, avec un seuil typique autour de 0.5
verts_seg, faces_seg, _, _ = measure.marching_cubes(seg_smooth, level=0.5)


# Meshs
mesh_seg = go.Mesh3d(
    x=verts_seg[:, 0], y=verts_seg[:, 1], z=verts_seg[:, 2],
    i=faces_seg[:, 0], j=faces_seg[:, 1], k=faces_seg[:, 2],
    color='lightblue', opacity=0.3, name='Myocarde'
)
mesh_cam = go.Mesh3d(
    x=verts_cam[:, 0], y=verts_cam[:, 1], z=verts_cam[:, 2],
    i=faces_cam[:, 0], j=faces_cam[:, 1], k=faces_cam[:, 2],
    intensity=verts_cam[:, 2],
    colorscale='Jet', opacity=0.7, name='Grad-CAM'
)

fig = go.Figure(data=[mesh_seg, mesh_cam])
fig.update_layout(
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    legend=dict(x=0.7, y=0.9),
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data'
    ),
    title="🫀 Cardio-Visua - Visualisation 3D Segment + GradCAM (améliorée)"
)
fig.show()




