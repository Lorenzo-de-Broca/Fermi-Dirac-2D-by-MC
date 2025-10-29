import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Chargement
data = np.load("occupations.npz")
matrices = [data[key] for key in data]

n_max = (matrices[0].shape[0] - 1) // 2

# Préparation des coordonnées
n1, n2 = np.indices(matrices[0].shape)
n1 = n1.flatten() - n_max
n2 = n2.flatten() - n_max

# Préparation de la figure
fig, ax = plt.subplots(figsize=(6,6))
sc = ax.scatter([], [], c=[], cmap='viridis', s=40, marker='s')
ax.set_xlim(-n_max, n_max)
ax.set_ylim(-n_max, n_max)
ax.set_xlabel("n1")
ax.set_ylabel("n2")
ax.set_title("Occupation des états quantiques (animation)")
cb = plt.colorbar(sc, ax=ax, label="Occupation / step")

# Fonction d'update
def update(frame):
    occ = matrices[frame]
    sc.set_offsets(np.c_[n1, n2])
    sc.set_array(occ.flatten())
    ax.set_title(f"Étape {frame * 100}")  # suppose sauvegarde toutes les 100 itérations
    return (sc,)

# Création de l’animation
ani = FuncAnimation(fig, update, frames=len(matrices), interval=200, blit=True)

plt.show()