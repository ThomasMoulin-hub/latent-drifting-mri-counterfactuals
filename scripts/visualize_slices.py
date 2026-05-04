import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def visualize_patient_slices(data_dir="data/processed", output_file="patient_slices.png", num_patients=2):
    """
    Charge aléatoirement quelques patients et affiche toutes leurs coupes extraites sous forme de grille
    pour vérifier visuellement la pertinence de la zone d'intérêt (ROI).
    """
    csv_path = Path(data_dir) / "metadata.csv"
    if not csv_path.exists():
        print(f"Erreur: {csv_path} introuvable. Avez-vous lancé le prétraitement ?")
        return
        
    df = pd.read_csv(csv_path)
    patients = df['patient_id'].unique()
    
    if len(patients) == 0:
        print("Aucun patient trouvé dans le CSV.")
        return
        
    # Choisir aléatoirement quelques patients
    selected_patients = random.sample(list(patients), min(num_patients, len(patients)))
    
    fig, axes = plt.subplots(num_patients, 10, figsize=(20, 2 * num_patients))
    fig.suptitle("Vérification Visuelle des Coupes Extraites (ROI)", fontsize=16)
    
    # Si axes est 1D (un seul patient), on le rend 2D pour la boucle
    if num_patients == 1:
        axes = np.array([axes])
        
    for i, p_id in enumerate(selected_patients):
        # Récupérer toutes les coupes du patient, triées par slice_idx
        p_df = df[df['patient_id'] == p_id].sort_values(by='slice_idx')
        label = p_df.iloc[0]['label']
        slices = p_df['slice_path'].tolist()
        
        # On va afficher 10 coupes réparties uniformément sur l'ensemble extrait
        indices_to_show = np.linspace(0, len(slices) - 1, 10, dtype=int)
        
        for j, idx in enumerate(indices_to_show):
            slice_file = slices[idx]
            slice_path = Path(data_dir) / slice_file
            
            if slice_path.exists():
                img = np.load(slice_path)
                # vmin=0, vmax=1 ensures the true absolute contrast is displayed, preventing 
                # matplotlib from artificially brightening dark slices.
                axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[i, j].axis('off')
                slice_num = p_df.iloc[idx]['slice_idx']
                if j == 0:
                    axes[i, j].set_title(f"{p_id} ({label})\nZ={slice_num}", fontsize=10)
                else:
                    axes[i, j].set_title(f"Z={slice_num}", fontsize=10)
            else:
                axes[i, j].axis('off')
                axes[i, j].set_title("Fichier manquant")
                
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_file, dpi=150)
    print(f"L'image de vérification a été sauvegardée sous : {output_file}")
    print("Ouvrez cette image pour vérifier si vous voyez le haut du crâne (inutile) ou bien les ventricules (utile).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="patient_slices.png")
    parser.add_argument("--patients", type=int, default=3, help="Nombre de patients à afficher")
    args = parser.parse_args()
    
    visualize_patient_slices(args.data_dir, args.out, args.patients)
