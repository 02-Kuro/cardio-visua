# CardioVisua App

CardioVisua est une application de visualisation et de prédiction cardiaque basée sur l’intelligence artificielle. 
Elle permet d’analyser des données médicales, d’afficher un modèle 3D du cœur, et de générer des prédictions à partir de modèles entraînés.

---

## 🩺 Objectif du projet

L’objectif de **CardioVisua** est d’aider à la compréhension et à la visualisation de signaux cardiaques à l’aide de modèles prédictifs et de représentations 3D. 
L’application met en avant l’intégration entre **IA médicale** et **visualisation interactive**.

---

## ⚙️ Structure du projet

CardioVisuaApp/
│
├── src/ # Code source de l'application
├── modules/ # Modules d’analyse et de prédiction
├── assets/ # Fichiers statiques (icônes, textures, etc.)
├── data/ # Dossier contenant les modèles 3D et fichiers de prédiction (non inclus)
├── requirements.txt # Dépendances Python
├── app.py # Point d’entrée principal
└── README.md # Documentation du projet


---

## 🚫 Données et modèles non inclus

Le dossier `data/` **n’est pas inclus dans le dépôt GitHub** pour deux raisons principales :

1. **Poids et taille** des fichiers (modèles 3D et données de prédiction volumineux). 
2. **Contraintes de licence** interdisant la redistribution publique de certains modèles et données médicales.

---

## 📂 Données et contenu du dossier `data/`

Le dossier `data/` **n’est pas inclus dans le dépôt public** pour des raisons de licence et de taille.  
Il contient les données et fichiers nécessaires au fonctionnement local de l’application :

- `ECG/` : exemples de signaux ECG provenant du **MIT-BIH Arrhythmia Database**, utilisés pour la démonstration du modèle 1D CNN.  
- `prediction/` : résultats des prédictions générées par les modèles (IRM segmentées, masques, etc.).  
- `testing/` : sous-ensemble des IRM issues du dataset **ACDC** (Automated Cardiac Diagnosis Challenge).  
- `heart.obj` : modèle 3D du cœur utilisé pour la visualisation interactive.

> ⚠️ Les jeux de données **ACDC** et **MIT-BIH Arrhythmia** sont soumis à des licences d’utilisation spécifiques.  
> Ils **ne peuvent pas être redistribués publiquement** dans ce dépôt.  
>  
> 🔗 Sources officielles :  
> - **ACDC Dataset** : [https://www.creatis.insa-lyon.fr/Challenge/acdc](https://www.creatis.insa-lyon.fr/Challenge/acdc)  
> - **MIT-BIH Arrhythmia Database** : [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)

Les utilisateurs souhaitant reproduire les résultats doivent **télécharger eux-mêmes ces datasets** depuis les sites officiels, puis placer les fichiers dans le répertoire `data/` selon la structure indiquée ci-dessus.

---

---

## 🧠 Technologies utilisées

- **Python 3.10+**
- **PyTorch** — pour les modèles de prédiction
- **Open3D / PyVista** — pour la visualisation 3D
- **Tkinter / PyQt** — pour l’interface utilisateur
- **NumPy / Pandas / Matplotlib**

---

## 🚀 Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/02.kuro/CardioVisuaApp.git
   cd CardioVisuaApp

    Installer les dépendances :

pip install -r requirements.txt

Ajouter les fichiers manquants (si autorisé) :

    Créez un dossier data/ à la racine du projet.

    Placez-y les fichiers du modèle 3D et les modèles de prédiction.

Lancer l’application :

    python app.py

📘 Licence

Ce projet est distribué sous licence MIT, sauf pour les éléments du dossier data/ qui sont soumis à des licences distinctes et ne peuvent pas être redistribués publiquement.
👤 Auteur

Hadi Benharrat
📧 Contact : [hadi.bnh00@gmail.com]
🧩 GitHub : https://github.com/02.kuro

