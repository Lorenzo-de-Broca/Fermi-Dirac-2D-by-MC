import matplotlib.pyplot as plt
import numpy as np

def plot_occupation(occupation_arr, n_max, step, T):
    """
    Fonction pour tracer la densité de probabilité d'occupation des états quantiques
    et la densité d'energie en fonction des nombres quantiques principaux n1 et n2.
    
    Args:
        occupation_arr (array): Liste des occupations des états quantiques
        n_max (int): Nombre maximum pour les nombres quantiques principaux
        step (int): Numéro de l'étape actuelle de la simulation
        T (float): Température du système
        
    Outputs:
        Un graphique affichant la carte d'occupation des états quantiques
    """
    
    density_occupation_arr = occupation_arr/step  # Transpose pour une meilleure visualisation
    
    plt.figure(figsize=(8,6))
    plt.imshow(density_occupation_arr, origin='lower', extent=[-n_max, n_max, -n_max, n_max], cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Occupation')
    plt.title(f'Carte d\'occupation des états quantiques à l\'étape {step} et T={T}K')
    plt.xlabel('n1')
    plt.ylabel('n2')
    plt.grid()
    plt.show()  
    
def plot_energy_distribution(occupation_arr, E0, n_max, step, T):
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
    
    energy_levels = []
    occupation_levels = []
    
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            energy = (n1**2 + n2**2)
            occupation = occupation_arr[n1 + n_max, n2 + n_max] / step
            energy_levels.append(energy)
            occupation_levels.append(occupation)
    
    plt.figure(figsize=(8,6))
    plt.scatter(energy_levels, occupation_levels, alpha=0.5)
    plt.title(f'Distribution d\'énergie des particules à l\'étape {step} et T={T}K')
    plt.xlabel('Énergie')
    plt.ylabel('Occupation')
    plt.grid()
    plt.show()