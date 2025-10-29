import matplotlib.pyplot as plt
import numpy as np

def plot_occupation(occupation_arr, n_max, step, T):
    """
    Version discrète avec points distincts (scatter plot).
    """
    n1, n2 = np.indices(occupation_arr.shape)-n_max
    occ = occupation_arr / step

    plt.figure(figsize=(8,6))
    plt.scatter(
        n1.flatten(),
        n2.flatten(),
        c=occ.flatten(),
        cmap='viridis',
        marker='s',   # carré
        s=80,         # taille du carré
        edgecolor='k' # bord noir
    )

    plt.colorbar(label='Occupation')
    plt.title(f"Occupation discrète des états quantiques à l'étape {step} (T={T} K)")
    plt.xlabel('n1')
    plt.ylabel('n2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
def plot_energy_distribution(occupation_arr, n_max, step, T):
    """
    Fonction pour tracer la distribution d'énergie des particules dans le système.
    
    Args:
        occupation_arr (array): Liste des occupations des états quantiques
        E0 (float): Constante d'énergie
        n_max (int): Nombre maximum pour les nombres quantiques principaux
        step (int): Numéro de l'étape actuelle de la simulation
        T (float): Température du système
        
    Outputs:
        Un graphique affichant la distribution d'énergie des particules
    """
    
    energy_levels = np.zeros(2*(n_max*n_max)+1)
    degenerescence_levels = np.zeros(2*(n_max*n_max)+1)
    occupation_levels = np.zeros(2*(n_max*n_max)+1)
    
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            energy = int((n1**2 + n2**2))
            occupation_levels[energy] += occupation_arr[n1 + n_max, n2 + n_max]
            degenerescence_levels[energy] += 1
            energy_levels[energy] = energy
            
    plt.figure(figsize=(8,6))
    plt.plot(energy_levels, occupation_levels, "r+", markersize=8, label='Occupation des niveaux d\'énergie')
    plt.title(f'Distribution d\'énergie des particules à l\'étape {step} et T={T}K')
    plt.xlabel('Énergie (adimensionnée)')
    plt.ylabel('Occupation')
    plt.grid()
    plt.show()