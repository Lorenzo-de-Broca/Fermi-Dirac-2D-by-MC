import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.animation import FuncAnimation

from main import load_input

os.makedirs("anim_vid", exist_ok=True)

parser = argparse.ArgumentParser(description="Parser for simulation input file.")

# Définition des arguments du parser
parser.add_argument("--file", type=str, required=False, default="input.yaml", \
        help="Nom ou chemin du fichier de configuration YAML")
    
# Lecture des arguments
args = parser.parse_args()
input_file = args.file

# Paramètres de style
FONTSIZE_TITLE = 20
FONTSIZE_LABEL = 18
FONTSIZE_TICKS = 14
iteration_step = 100          # Intervalle entre les frames en nombre d'itérations réelles

# Chargement des données

# Récupération du nom du fichier de configuration
config = load_input("input_anim.yaml")

T = config["T"]
step = config["step"]
num_steps = config["step"]
N = int(config["N"]/2)
freq_save = config.get("freq_save", 100)  # Fréquence de sauvegarde des occupations

data_file = (f"animations_file/energy_distribution_N{N*2:.0f}_T{T:.0e}K_step{step:.1e}_freq_save{freq_save:.0f}.npz")
#data_file = (f"occupations.npz")
data = np.load(data_file)

matrices = [data[key] for key in data]
n_max = (matrices[0].shape[0] - 1) // 2

# Création de la grille de points
n1, n2 = np.mgrid[-n_max:n_max+1, -n_max:n_max+1]

# Configuration de la figure
fig, ax = plt.subplots(figsize=(8, 8))

# Premier scatter plot avec la première matrice
scatter = ax.scatter(n1, n2, c=matrices[0], cmap='viridis', 
                    s=60, marker='s')

# Configuration des axes et labels
ax.set_xlim(-n_max, n_max)
ax.set_ylim(-n_max, n_max)
ax.set_xlabel("n1", fontsize=FONTSIZE_LABEL)
ax.set_ylabel("n2", fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICKS)

# Titre initial
ax.set_title("Occupation des états quantiques (étape 0)", 
             fontsize=FONTSIZE_TITLE, pad=10)

# Colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Occupation / step", fontsize=FONTSIZE_LABEL)
cbar.ax.tick_params(labelsize=FONTSIZE_TICKS)

def update(frame):
    """Mise à jour de l'animation"""
    # Mise à jour des couleurs
    scatter.set_array(matrices[frame].flatten())
    # Mise à jour du titre
    ax.set_title(f"Occupation des états quantiques (étape {frame*freq_save})",
                 fontsize=FONTSIZE_TITLE, pad=10)

# Création de l'animation
anim = FuncAnimation(
    fig, 
    update,
    frames=len(matrices),
    interval=200,  # 200ms entre chaque frame
    repeat=False   # Pas de boucle pour la sauvegarde
)

# Sauvegarder l'animation
print("Sauvegarde de l'animation en cours...")
anim.save(f'anim_vid/energy_distribution_N{N*2:.0f}_T{T:.0e}K_step{step:.1e}_freq_save{freq_save:.0f}.mp4', 
          writer='ffmpeg',
          fps=5,  # 5 images par seconde
          dpi=300,  # Bonne résolution
          bitrate=2000)  # Bonne qualité vidéo

print("Animation sauvegardée dans le dossier 'anim'")
plt.show()