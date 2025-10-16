import os
import torch
import nibabel as nib
import numpy as np
from skimage.transform import resize
from monai.networks.nets import resnet18  # Assure-toi d’avoir monai installé

# Détection du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle
model_path = "C:/Users/kuro/Desktop/3dUnet/cardiac_pathology_classifier.pth"
checkpoint = torch.load(model_path, map_location=device)

model = resnet18(
    spatial_dims=3,
    n_input_channels=1,
    num_classes=5,
    pretrained=False
)
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)
model.eval()

# Mapping classes → pathologies
class_mapping = {
    0: 'NOR',   # Sujet normal
    1: 'MINF',  # Infarctus du myocarde
    2: 'DCM',   # Cardiomyopathie dilatée
    3: 'HCM',   # Cardiomyopathie hypertrophique
    4: 'RV'     # Dysfonction ventriculaire droite
}

# Prétraitement du scan
def preprocess_scan(scan_path):
    img = nib.load(scan_path).get_fdata()
    img = np.clip(img, 0, np.percentile(img, 99.5))
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    img = resize(img, (64, 64, 64), mode='constant', preserve_range=True)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# Fonction principale utilisée dans l’app
def predict_scan(scan_path):
    try:
        input_tensor = preprocess_scan(scan_path)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0][pred_class].item()

        return {
            'scan_name': os.path.basename(scan_path),
            'prediction': class_mapping[pred_class],
            'confidence': confidence,
            'probabilities': {class_mapping[i]: float(probs[0][i]) for i in class_mapping}
        }

    except Exception as e:
        print(f"Erreur lors du traitement de {scan_path}: {str(e)}")
        return None
