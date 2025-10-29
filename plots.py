import matplotlib.pyplot as plt
import numpy as np
import os

from CondInit import kbT_adim, mu_adim_fct, Fermi_Dirac_distribution

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
    plt.title(f"Occupation discrète des états quantiques à l'étape {step:.1e} (T={T} K)")
    plt.xlabel('n1')
    plt.ylabel('n2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
def plot_energy_distribution(occupation_arr, n_max, Ef, step, N, T, L_box):
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
        
    # Dossier où sauvegarder les figures
    output_dir = "fig"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialisation de tableaux d'énergie
    energy_levels = np.zeros(2*(n_max*n_max)+1)
    degenerescence_levels = np.zeros(2*(n_max*n_max)+1)
    occupation_levels = np.zeros(2*(n_max*n_max)+1)

    # Calcul des niveaux d'énergie, de la dégénérescence et de l'occupation    
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            energy = int((n1**2 + n2**2))
            occupation_levels[energy] += occupation_arr[n1 + n_max, n2 + n_max]
            degenerescence_levels[energy] += 1
            energy_levels[energy] = energy

    # mask to remove zero degenerescence levels
    mask = degenerescence_levels > 0
    energy_levels_masked = energy_levels[mask]
    occupation_levels_masked = occupation_levels[mask]
    degenerescence_levels_masked = degenerescence_levels[mask]

    # Pour tracer l'énergie de Fermi
    y_E_Fermi = [0,np.max(occupation_levels_masked/(degenerescence_levels_masked*step))*1.1]
    x_E_Fermi = [Ef, Ef]
    x_E_Fermi_kbT = [Ef + kbT_adim(L_box, T), Ef + kbT_adim(L_box, T)]
    
    # Calcul de la distribution de Fermi-Dirac théorique pour comparer 
    x_Fermi_Dirac = np.linspace(0, np.max(energy_levels_masked), 1000)
    
    mu_adim = mu_adim_fct (L=L_box, T=T, E_f=Ef)
    T_adim = kbT_adim(L_box, T)
    y_Fermi_Dirac = Fermi_Dirac_distribution(x_Fermi_Dirac, mu_adim,T_adim)

    plt.figure(figsize=(8,6))
    #plt.plot(energy_levels, occupation_levels, "r", markersize=8, label='Occupation des niveaux d\'énergie')
    plt.plot(energy_levels_masked, occupation_levels_masked/(degenerescence_levels_masked*step), "b+", markersize=5, label='Occupation par niveau d\'énergie sans dégénérescence')
    plt.plot(x_E_Fermi, y_E_Fermi, 'r--', label=f'Énergie de Fermi : {Ef:.2f} adimensionnée')
    #plt.plot(x_E_Fermi_kbT, y_E_Fermi, 'g--', label=f'Énergie de Fermi + k_b*T : {Ef + T_adim:.2f} adimensionnée')

    plt.plot(x_Fermi_Dirac, y_Fermi_Dirac, 'k-', label='Distribution de Fermi-Dirac théorique')
    plt.legend()
    plt.title(f'Distribution d\'énergie des particules à l\'étape {step:.1e}, pour N={N*2:.0f} e- et T={T}K')
    plt.xlabel('Énergie (adimensionnée)')
    plt.ylabel('Occupation')
    plt.grid()
    
    # Sauvegarde
    filename1 = os.path.join(output_dir, f"energy_distribution_N{N*2:.0f}_T{T:.0e}K_step{step:.1e}.png")
    plt.savefig(filename1, dpi=300, bbox_inches="tight")
    
    plt.show()
      
    # Graphique de la dégénérescence des niveaux d'énergie
    plt.plot(energy_levels_masked, degenerescence_levels_masked, "g+", markersize=8, label='Dégénérescence des niveaux d\'énergie')
    plt.legend()
    plt.title(f'Dégénérescence des niveaux d\'énergie à l\'étape {step:.1e} et T={T}K')
    plt.xlabel('Énergie (adimensionnée)')
    plt.ylabel('Dégénérescence')
    plt.grid()
    
    plt.show()



def plot_energy_distribution_multiT(occupation_arr, n_max, Ef, step, T, Tvalues, L):
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

    # mask to remove zero degenerescence levels
    mask = degenerescence_levels > 0
    energy_levels_masked = energy_levels[mask]
    occupation_levels_masked = occupation_levels[mask]
    degenerescence_levels_masked = degenerescence_levels[mask]

    if T == np.min(Tvalues):
        plt.figure(figsize=(8,6))
        # Pour tracer l'énergie de Fermi
        y_fermi = [0,np.max(occupation_levels_masked/(degenerescence_levels_masked*step))*1.1]
        x_fermi = [Ef, Ef]
        plt.plot(x_fermi, y_fermi, 'r--', label=f'Énergie de Fermi adimensionnée : {Ef:.2f}')
    plt.plot(energy_levels_masked, occupation_levels_masked/(degenerescence_levels_masked*step), \
        label='T = {:.0f}K'.format(T))#, markersize=5, "b+",
    if T == np.max(Tvalues):
        # Pour tracer l'énergie de Fermi
        y_fermi = [0,np.max(occupation_levels_masked/(degenerescence_levels_masked*step))*1.1]
        x_fermi = [Ef, Ef]
        plt.plot(x_fermi, y_fermi, 'r--', label=f'Énergie de Fermi adimensionnée : {Ef:.2f}')
        #affichage des courbes pour chaque T
        plt.legend()
        plt.title(f'Distribution d\'énergie des particules à l\'étape {step} pour différentes températures')
        plt.xlabel('Énergie (adimensionnée)')
        plt.xlim(0, 4*Ef)
        plt.ylabel('Occupation')
        plt.grid()
        plt.show()
     
