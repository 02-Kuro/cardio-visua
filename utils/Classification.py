#!/usr/bin/env python
# coding: utf-8

# 📦 1. Imports & Configuration
import os
import re
import glob
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torchio as tio
from monai.networks.nets import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
# Hardware configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


# 📦 1. Imports & Configuration
import os
import re
import glob
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torchio as tio
from monai.networks.nets import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize

# Set reproducible results
torch.manual_seed(42)
np.random.seed(42)

# 📂 2. Paths & Global Parameters
# =============== USER CONFIG ===============
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = (64, 64, 64)  # Optimized for 3D ResNet

# Update these paths to your local environment
CSV_PATH = "C:/Users/kuro/Desktop/3dUnet/pfe data/info.csv"
IMAGES_DIR = "C:/Users/kuro/Desktop/3dUnet/pfe data/images"
PRETRAINED_WEIGHTS = "C:/Users/kuro/Desktop/3dUnet/pfe data/pretrained/resnet_18_23dataset.pth"
# ===========================================

# Hardware configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 🧠 3. Cardiac Dataset Class with Proper CSV Integration
class CardiacDataset(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Load and process CSV
        self.df = pd.read_csv(csv_path)
        self._process_csv()
        
        # Match NIfTI files with patient data
        self._prepare_dataset()
        
        print(f"✅ Created dataset with {len(self.samples)} valid scans")

    def _process_csv(self):
        """Process CSV to create patient mapping"""
        # Handle both numeric and string patient IDs
        self.df['patient_num'] = self.df['patient_id'].str.extract('(\d+)').astype(int)
        self.df['formatted_id'] = self.df['patient_num'].apply(
            lambda x: f"patient{x:03d}"
        )
        
        # Create label mapping
        self.label_map = {
            row['formatted_id']: row['label']
            for _, row in self.df.iterrows()
        }
        
        # Create ED/ES frame mapping
        self.frame_map = {
            row['formatted_id']: (row['ED'], row['ES'])
            for _, row in self.df.iterrows()
        }

    def _prepare_dataset(self):
        """Match NIfTI files with patient data from CSV"""
        nii_files = glob.glob(os.path.join(self.data_dir, '*.nii.gz'))
        
        for file_path in nii_files:
            file_name = os.path.basename(file_path)
            
            # Extract patient and frame info
            match = re.match(r'patient(\d+)_frame(\d+)', file_name)
            if not match:
                continue
                
            patient_id = f"patient{match.group(1)}"
            frame_num = int(match.group(2))
            
            # Skip if patient not in CSV
            if patient_id not in self.label_map:
                continue
                
            # Get ED/ES frames for this patient
            ed_frame, es_frame = self.frame_map[patient_id]
            
            # Only use ED and ES frames
            if frame_num not in (ed_frame, es_frame):
                continue
                
            # Determine phase and get label
            phase = 'ED' if frame_num == ed_frame else 'ES'
            label = self.label_map[patient_id]
            
            self.samples.append((file_path, label, patient_id, phase))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, _, _ = self.samples[idx]
        
        # Load and preprocess image
        img = nib.load(file_path).get_fdata()
        img = np.clip(img, 0, np.percentile(img, 99.5))
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img = resize(img, IMAGE_SIZE, mode='constant', preserve_range=True)
        
        # Convert to tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return img_tensor, label

# 🔧 4. Data Preparation Pipeline
# Online augmentations (applied during training)
train_transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2)),
    tio.RandomAffine(scales=0.1, degrees=10),
    tio.RandomNoise(std=0.01),
    tio.RandomGamma(log_gamma=(-0.3, 0.3))
])

# Create full dataset
full_dataset = CardiacDataset(
    data_dir=IMAGES_DIR,
    csv_path=CSV_PATH,
    transform=None
)

# Split indices - stratify by labels
labels = [sample[1] for sample in full_dataset.samples]
train_idx, val_idx = train_test_split(
    range(len(full_dataset)), 
    test_size=0.2, 
    stratify=labels,
    random_state=42
)

# Create subset datasets
train_dataset = torch.utils.data.Subset(
    CardiacDataset(IMAGES_DIR, CSV_PATH, transform=train_transform), 
    train_idx
)
val_dataset = torch.utils.data.Subset(
    CardiacDataset(IMAGES_DIR, CSV_PATH, transform=None), 
    val_idx
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                          shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=0)

print(f"\n📊 Dataset Summary:")
print(f"  - Total scans: {len(full_dataset)}")
print(f"  - Training scans: {len(train_dataset)}")
print(f"  - Validation scans: {len(val_dataset)}")
print(f"  - Class distribution: {np.bincount(labels)}")

# 🏗️ 5. Model Initialization with MedicalNet Weights
def load_pretrained_resnet():
    """Initialize ResNet-18 with MedicalNet weights"""
    # Create model
    model = resnet18(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=5,  # 5 pathology classes
        pretrained=False
    )
    
    # Load MedicalNet weights
    checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Remove 'module.' prefix from DataParallel
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    
    # Load compatible weights
    model.load_state_dict(new_state_dict, strict=False)
    
    # Verify weight loading
    loaded_keys = set(new_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    print(f"\n🔍 Weight loading report:")
    print(f"  - Successfully loaded: {len(loaded_keys & model_keys)} layers")
    print(f"  - Missing: {len(model_keys - loaded_keys)} layers (random init)")
    print(f"  - Extra: {len(loaded_keys - model_keys)} unused weights")
    
    return model.to(device)

model = load_pretrained_resnet()

# 🚂 6. Training Setup
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 📈 Training function
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
    return epoch_loss / len(loader.dataset)

# 📊 Validation function
def validate(model, loader, criterion):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
    return val_loss / len(loader.dataset), acc, report_str, report_dict

# 🚀 7. Training Execution
train_losses = []
val_losses = []
val_accuracies = []
best_acc = 0
best_report = ""

# Class names for reporting
class_names = ['NOR', 'MINF', 'DCM', 'HCM', 'RV']

print("\n🚀 Starting training...")
for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*50}")
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    
    # Validate
    val_loss, val_acc, val_report_str, val_report_dict = validate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Update scheduler
    scheduler.step()
    
    # Print epoch summary
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}")
    print(f"  Learning Rate: {current_lr:.2e}")
    
    # Print classification report
    print("\nValidation Classification Report:")
    print(val_report_str)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_report = val_report_str
        torch.save(model.state_dict(), "best_cardiac_model.pth")
        print(f"\n💾 Saved new best model with accuracy: {val_acc:.4f}")

# 📈 8. Visualization & Final Evaluation
# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training History')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, 'g-', label='Val Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Final evaluation
model.load_state_dict(torch.load("best_cardiac_model.pth"))
_, final_acc, final_report_str, _ = validate(model, val_loader, criterion)

print("\n⭐️ Final Evaluation ⭐️")
print(f"Validation Accuracy: {final_acc:.4f}")
print("\nBest Model Classification Report:")
print(best_report)

# 💾 Save full model for deployment
torch.save({
    'model_state': model.state_dict(),
    'class_names': class_names,
    'label_mapping': {i: name for i, name in enumerate(class_names)}
}, "cardiac_pathology_classifier.pth")

print("\n✅ Training complete! Best model saved.")


# 🧪 9. Cellule de Test/Prédiction
import os
import torch
import nibabel as nib
import numpy as np
from skimage.transform import resize

# --------------------------------------------------
# 1. Chargement du modèle entraîné
# --------------------------------------------------
model_path = "C:/Users/kuro/Desktop/3dUnet/cardiac_pathology_classifier.pth"
checkpoint = torch.load(model_path, map_location=device)

# Initialisation du modèle
model = resnet18(
    spatial_dims=3,
    n_input_channels=1,
    num_classes=5,
    pretrained=False
)
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)
model.eval()

# Correspondance classes-pathologies (identique à l'entraînement)
class_mapping = {
    0: 'NOR',   # Sujet normal
    1: 'MINF',  # Infarctus du myocarde
    2: 'DCM',   # Cardiomyopathie dilatée
    3: 'HCM',   # Cardiomyopathie hypertrophique
    4: 'RV'     # Dysfonction ventriculaire droite
}

# --------------------------------------------------
# 2. Fonctions de prédiction
# --------------------------------------------------
def preprocess_scan(scan_path):
    """Prétraitement identique à l'entraînement"""
    img = nib.load(scan_path).get_fdata()
    img = np.clip(img, 0, np.percentile(img, 99.5))  # Suppression des outliers
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)  # Normalisation
    img = resize(img, (64, 64, 64), mode='constant', preserve_range=True)  # Resize
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

def predict_scan(scan_path):
    """Prédiction pour un seul scan"""
    try:
        # Préprocessing
        input_tensor = preprocess_scan(scan_path)
        
        # Prédiction
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0][pred_class].item()
        
        # Formatage des résultats
        return {
            'scan_name': os.path.basename(scan_path),
            'prediction': class_mapping[pred_class],
            'confidence': confidence,
            'probabilities': {class_mapping[i]: float(probs[0][i]) for i in class_mapping}
        }
    
    except Exception as e:
        print(f"Erreur lors du traitement de {scan_path}: {str(e)}")
        return None

# --------------------------------------------------
# 3. Exemple d'utilisation
# --------------------------------------------------
# Test sur un scan individuel
test_file = "C:/Users/kuro/Desktop/3dUnet/pfe data/testing/images/patient101_frame01.nii_pred.nii.gz"

if os.path.exists(test_file):
    result = predict_scan(test_file)
    
    if result:
        print("\n🔍 Résultats de prédiction:")
        print(f"Scan: {result['scan_name']}")
        print(f"Pathologie prédite: {result['prediction']} (Confiance: {result['confidence']:.1%})")
        print("\nDétail des probabilités:")
        for pathology, prob in result['probabilities'].items():
            print(f"- {pathology}: {prob:.2%}")
else:
    print(f"⚠️ Fichier non trouvé: {test_file}")

# --------------------------------------------------
# 4. Évaluation sur un dossier complet (optionnel)
# --------------------------------------------------
def evaluate_test_directory(test_dir):
    """Évaluation sur un ensemble de scans"""
    if not os.path.isdir(test_dir):
        print(f"Le dossier {test_dir} n'existe pas")
        return
    
    nifti_files = [f for f in os.listdir(test_dir) if f.endswith('.nii.gz')]
    if not nifti_files:
        print("Aucun fichier .nii.gz trouvé")
        return
    
    print(f"\n🧪 Évaluation sur {len(nifti_files)} scans...")
    
    for filename in nifti_files:
        filepath = os.path.join(test_dir, filename)
        result = predict_scan(filepath)
        
        if result:
            print(f"\n{filename}:")
            print(f"- Prédiction: {result['prediction']} ({result['confidence']:.1%})")
            print(f"- Probabilités: {', '.join([f'{k}: {v:.1%}' for k, v in result['probabilities'].items()])}")

# Exemple d'utilisation (décommentez pour utiliser)
# test_dir = "C:/chemin/vers/vos/scans/test"
# evaluate_test_directory(test_dir)


# 📦 Imports nécessaires
import os
import torch
import nibabel as nib
import numpy as np
from skimage.transform import resize
import torch.nn.functional as F
from monai.networks.nets import resnet18
import gc

# 📁 Dossiers
pred_dir = "C:/Users/kuro/Desktop/3dUnet/predictions"
gradcam_dir = "C:/Users/kuro/Desktop/3dUnet/3d grad cam"
os.makedirs(gradcam_dir, exist_ok=True)

# ⚙️ Appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Chargement du modèle entraîné
model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=5, pretrained=False)
checkpoint = torch.load("C:/Users/kuro/Desktop/3dUnet/cardiac_pathology_classifier.pth", map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

# 🔁 Classe cible (pour affichage)
class_mapping = {0: 'NOR', 1: 'MINF', 2: 'DCM', 3: 'HCM', 4: 'RV'}

# 🔥 Fonction Grad-CAM 3D
def generate_gradcam(model, input_tensor):
    activations = {}
    gradients = {}

    # Hook pour capturer les activations et gradients
    def forward_hook(module, input, output):
        activations['value'] = output
        output.register_hook(lambda grad: gradients.update({'value': grad}))
    
    # Enregistre le hook sur le bon layer
    hook = model.layer4.register_forward_hook(forward_hook)

    # Forward
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    score = output[0, pred_class]
    
    # Backward
    model.zero_grad()
    score.backward()

    # Extraction des gradients et activations
    grad = gradients['value']       # [1, C, D, H, W]
    fmap = activations['value']     # [1, C, D, H, W]

    # Moyenne pondérée des gradients
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.sum(weights * fmap, dim=1).squeeze(0)  # [D, H, W]
    cam = F.relu(cam)

    # Normalisation
    cam = cam.detach().cpu().numpy()
    cam = cam / (np.max(cam) + 1e-8)

    # Resize vers (64, 64, 64)
    cam_resized = resize(cam, (64, 64, 64), preserve_range=True)
    
    # Nettoyage du hook
    hook.remove()
    return cam_resized, pred_class

# 🧪 Boucle sur tous les patients
nifti_files = [f for f in os.listdir(pred_dir) if f.endswith("_pred.nii.gz")]
print(f"🧠 Génération des cartes Grad-CAM pour {len(nifti_files)} patients...\n")

for file in nifti_files:
    try:
        input_path = os.path.join(pred_dir, file)
        output_path = os.path.join(gradcam_dir, file.replace(".nii_pred.nii.gz", "_gradcam.nii.gz"))
        
        # Chargement
        img_nii = nib.load(input_path)
        img = img_nii.get_fdata()
        affine = img_nii.affine

        # Normalisation robuste
        img = np.clip(img, 0, np.percentile(img, 99.5))
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img_resized = resize(img, (64, 64, 64), preserve_range=True)

        # Tensor
        input_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        # Grad-CAM
        cam_volume, pred_class = generate_gradcam(model, input_tensor)

        # Sauvegarde
        nib.save(nib.Nifti1Image(cam_volume, affine), output_path)
        print(f"✅ {file} → classe prédite : {class_mapping[pred_class]} → {output_path}")
    
    except Exception as e:
        print(f"❌ Erreur avec {file} : {e}")
    
    # Nettoyage mémoire GPU
    torch.cuda.empty_cache()
    gc.collect()




