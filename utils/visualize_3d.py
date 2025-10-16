import os
import numpy as np
import nibabel as nib
import trimesh
from skimage import filters, measure
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from monai.networks.nets import resnet18

class_names = ['NOR', 'MINF', 'DCM', 'HCM', 'RV']

def show_heart_3d(img_path, seg_path, gradcam_path=None, threshold=0.2):
    # === Chemins relatifs ===
    root_dir = os.path.dirname(os.path.dirname(__file__))  # dossier parent du dossier utils/
    heart_model_path = os.path.join(root_dir, "data", "heart.obj")
    model_path = os.path.join(root_dir, "models", "cardiac_pathology_classifier.pth")

    # === Chargement modèle
    model_ckpt = torch.load(model_path, map_location="cpu")
    model_ckpt['model_state'] = {k.replace('module.', ''): v for k, v in model_ckpt['model_state'].items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = resnet18(spatial_dims=3, n_input_channels=1, num_classes=5).to(device)
    cnn.load_state_dict(model_ckpt['model_state'])
    cnn.eval()

    # === Chargement données
    img = nib.load(img_path).get_fdata()
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    spacing = nib.load(img_path).header.get_zooms()[:3]
    seg_size = np.array(seg.shape) * np.array(spacing)

    img_input = np.clip(img, 0, np.percentile(img, 99.5))
    img_input = (img_input - img_input.mean()) / (img_input.std() + 1e-8)
    img_resized = resize(img_input, (64, 64, 64), mode='constant', preserve_range=True)
    input_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    output = cnn(input_tensor)
    probs = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
    pred_class = np.argmax(probs)
    pred_class_name = class_names[pred_class]

    print(f"\n🧠 Pathologie prédite : {pred_class_name}")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {probs[i]:.3f}")

        # --- Affichage aussi dans Streamlit ---
    import streamlit as st
    st.success(f"🧠 Pathologie prédite : **{pred_class_name}**")
    st.markdown("### 📊 Probabilités des classes :")
    for i, cls in enumerate(class_names):
     st.markdown(f"- **{cls}** : `{probs[i]:.3f}`")
    # === Cœur réaliste
    heart_mesh = trimesh.load(heart_model_path, process=True)
    vertices = heart_mesh.vertices.copy()
    faces = heart_mesh.faces.copy()
    heart_center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    scale = np.min(seg_size / (vertices.max(axis=0) - vertices.min(axis=0)))
    vertices = (vertices - heart_center) * scale + seg_size / 2

    mesh_heart = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='#B0B0B0', opacity=0.2, name='Cœur 3D réaliste'
    )

    def preprocess_structure(mask, sigma=1.2):
        return filters.gaussian(mask.astype(np.float32), sigma=sigma)

    def get_mesh(mask, level=0.15):
        if np.max(mask) <= level:
            return None, None
        return measure.marching_cubes(mask, level=level, spacing=spacing)[:2]

    def plot_mesh(verts, faces, color, name, opacity=0.9):
        if verts is None or faces is None:
            print(f"⚠️ {name} vide")
            return None
        x, y, z = verts.T
        i, j, k = faces.T
        return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, name=name, showscale=False)

    structures = {
        "VD": (1, "#20B2AA", 0.1),
        "Myocarde": (2, "#B0B0B0", 0.2),
        "Hypertrophie": (3, "#FF4040", 0.1)
    }

    meshes = [mesh_heart]
    for name, (label, color, opacity) in structures.items():
        mask = preprocess_structure(seg == label)
        verts, faces = get_mesh(mask)
        mesh = plot_mesh(verts, faces, color, name, opacity)
        if mesh: meshes.append(mesh)

    # === Grad-CAM
    try:
        cnn.zero_grad()
        score = output[0, pred_class]
        score.backward()

        grad = input_tensor.grad[0][0].cpu().numpy()
        cam = np.maximum(grad, 0)
        cam = gaussian_filter(cam, sigma=1)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cam_resized = resize(cam, seg.shape, order=1, preserve_range=True, anti_aliasing=True)
        heart_mask = np.isin(seg, [1, 2, 3])
        cam_masked = cam_resized * heart_mask
        cam_masked[cam_masked < threshold] = 0
        cam_masked = gaussian_filter(cam_masked, sigma=1)

        print(f"✅ Grad-CAM shape (resized): {cam_resized.shape}")
        print(f"   Max value: {cam_masked.max():.3f}")
        print(f"   Activations > {threshold}: {np.sum(cam_masked > threshold)}")

        if np.max(cam_masked) > 0:
            verts, faces, values, _ = measure.marching_cubes(cam_masked, level=threshold, spacing=spacing)
            i, j, k = faces.T
            mesh_gradcam = go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=i, j=j, k=k,
                intensity=values,
                colorscale="YlOrRd",
                opacity=1.0,
                name="Grad-CAM",
                colorbar_title="Activation"
            )
            meshes.append(mesh_gradcam)
            print("✅ Grad-CAM reconstruit")
        else:
            print("⚠️ Aucune activation significative pour Grad-CAM")
    except Exception as e:
        print(f"❌ Erreur Grad-CAM : {e}")

    fig = go.Figure(data=meshes)
    fig.update_layout(
        title=f"🫀 Visualisation 3D - Pathologie : {pred_class_name}",
        scene=dict(
            xaxis=dict(backgroundcolor="black", color="white", gridcolor="gray"),
            yaxis=dict(backgroundcolor="black", color="white", gridcolor="gray"),
            zaxis=dict(backgroundcolor="black", color="white", gridcolor="gray"),
            aspectmode='data'
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        legend=dict(
            x=0.75, y=0.95,
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="white",
            borderwidth=1,
            font=dict(size=14, color="white")
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    import streamlit as st
    st.plotly_chart(fig, use_container_width=True)

