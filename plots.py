import matplotlib.pyplot as plt
import numpy as np
import os

from CondInit import kbT_adim, mu_adim_fct, Fermi_Dirac_distribution, L_box_std


# On définit la taille des légendes sur les figures 
title = 20
label = 18
legend = 16
ticks = 14

plt.rcParams['xtick.labelsize'] = ticks
plt.rcParams['ytick.labelsize'] = ticks


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
    plt.title(f"Occupation discrète des états quantiques (moyenne sur {step:.1e} steps) (T={T} K)", fontsize=title)
    plt.xlabel('n1', fontsize=label)
    plt.ylabel('n2', fontsize=label)
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
    y_Fermi_Dirac = Fermi_Dirac_distribution(x_Fermi_Dirac, mu_adim,T_adim) # théorique

    Fermi_Dirac_MC = occupation_levels_masked/(degenerescence_levels_masked*step)
    
    plt.figure(figsize=(8,6))
    #plt.plot(energy_levels, occupation_levels, "r", markersize=8, label='Occupation des niveaux d\'énergie')
    plt.plot(energy_levels_masked, Fermi_Dirac_MC, "b+", markersize=5, label='Occupation par niveau d\'énergie sans dégénérescence')
    plt.plot(x_E_Fermi, y_E_Fermi, 'r--', label=f'Énergie de Fermi : {Ef:.2f} adimensionnée')
    #plt.plot(x_E_Fermi_kbT, y_E_Fermi, 'g--', label=f'Énergie de Fermi + k_b*T : {Ef + T_adim:.2f} adimensionnée')

    plt.plot(x_Fermi_Dirac, y_Fermi_Dirac, 'k-', label='Distribution de Fermi-Dirac théorique')
    plt.legend(fontsize=legend)
    plt.title(f'Distribution d\'énergie des particules (moyenne sur {step:.1e} steps), pour N={N*2:.0f} e- et T={T}K', fontsize=title)
    plt.xlabel('Énergie (adimensionnée)', fontsize=label)
    plt.ylabel('Occupation', fontsize=label)
    plt.grid()
    
    # Sauvegarde
    filename1 = os.path.join(output_dir, f"energy_distribution_N{N*2:.0f}_T{T:.0e}K_step{step:.1e}.png")
    plt.savefig(filename1, dpi=300, bbox_inches="tight")
    
    plt.show()
      
    # Graphique de la dégénérescence des niveaux d'énergie
    plt.plot(energy_levels_masked, degenerescence_levels_masked, "g+", markersize=8, label='Dégénérescence des niveaux d\'énergie')
    plt.legend(fontsize=legend)
    plt.title(f'Dégénérescence des niveaux d\'énergie, T={T}K', fontsize=title)
    plt.xlabel('Énergie (adimensionnée)', fontsize=label)
    plt.ylabel('Dégénérescence', fontsize=label)
    plt.grid()
    
    plt.show()
    return(Fermi_Dirac_MC, energy_levels_masked)

def plot_energy_distribution_multiT(occupation_arr, n_max, Ef, step, N, T, Tvalues, L):
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
        plt.plot(x_fermi, y_fermi, 'r--', label=f'E_F adimensionnée : {Ef:.2f}')
        
    plt.plot(energy_levels_masked, occupation_levels_masked/(degenerescence_levels_masked*step), \
        label='T = {:.0f}K'.format(T))#, markersize=5, "b+",
    
    #estimation de mu_adim 
    distrib_FD = occupation_levels_masked/(degenerescence_levels_masked*step)
    #n_Esup = np.min(np.where(distrib_FD >= 0.5))
    #n_Einf = np.max(np.where(distrib_FD <= 0.5))
    #E_sup = energy_levels_masked[n_Esup]
    #E_inf = energy_levels_masked[n_Einf]
    #on prend la droite passant par ces deux points, puis son abscisse à l'ordonnée 0.5
    #mu_adim_estime = E_inf +  (E_sup - E_inf) * (0.5 - n_Einf)/(n_Esup - n_Einf)
    #print(f"mu_adim(T={T}K) ~= {mu_adim_estime:.4f} (entre E={E_inf} et E={E_sup})")

    # On cherche les indices autour de 1/2
    target = 0.5
    diff = distrib_FD - target
    # indices où la densité passe de >0.5 à <0.5 (ou l’inverse)
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        raise ValueError("Aucune valeur proche de 1/2 trouvée — vérifier les données.")
    # On prend le point juste avant et juste après
    i_below = sign_changes[0]
    i_above = i_below + 1
    
    E_below = energy_levels_masked[i_below]
    E_above = energy_levels_masked[i_above]
    D_below = distrib_FD[i_below]
    D_above = distrib_FD[i_above]
    # Interpolation linéaire pour trouver E où Densité = 1/2
    mu_adim_estime = E_below + (target - D_below) * (E_above - E_below) / (D_above - D_below)
    #print("Énergie juste en dessous :", E_below)
    #print("Énergie juste au-dessus :", E_above)
    print("Estimation du niveau de Fermi (mu) ≈", mu_adim_estime)
    
    if T == np.max(Tvalues):
        #affichage des courbes pour chaque T
        plt.legend()
        plt.title(f'Distribution d\'énergie des particules après {step} étapes pour différentes températures (N = {N*2})')
        plt.xlabel('Énergie (adimensionnée)')
        plt.xlim(0, 4*Ef)
        plt.ylabel('Occupation')
        plt.grid()
        #save figures 
        output_dir = "fig/multiT"
        os.makedirs(output_dir, exist_ok=True)
        Nbre_simus = np.size(Tvalues)
        Tmin = np.min(Tvalues)
        Tmax = np.max(Tvalues)
        filename = os.path.join(output_dir, f"E_distrib_{Nbre_simus}simulations_N={N*2}_T={Tmin}_a_{Tmax}K_steps={step:.1e}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        
        plt.show()
    return mu_adim_estime
     
def plot_energy_distribution_multiN(occupation_arr, n_max, Ef, step, T, N, Nvalues, L, color):
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

    if N == np.min(Nvalues):
        plt.figure(figsize=(8,6))
    # Pour tracer l'énergie de Fermi
    y_fermi = [0,np.max(occupation_levels_masked/(degenerescence_levels_masked*step))*1.1]
    x_fermi = [Ef, Ef]
    plt.plot(x_fermi, y_fermi, color+'--', label=f'Énergie de Fermi adimensionnée : {Ef:.2f}')
    plt.plot(energy_levels_masked, occupation_levels_masked/(degenerescence_levels_masked*step), color, \
        label='N = {}'.format(N))#, markersize=5, "b+",
    if N == np.max(Nvalues):
        #affichage des courbes pour chaque T
        plt.legend()
        plt.title(f'Distribution d\'énergie des particules après {step} étapes pour différents N (T = {T}K)')
        plt.xlabel('Énergie (adimensionnée)')
        plt.xlim(0, 4*Ef)
        plt.ylabel('Occupation', fontsize=label)
        plt.grid()
        #save figures 
        output_dir = "fig/multiN"
        os.makedirs(output_dir, exist_ok=True)
        Nbre_simus = np.size(Nvalues)
        Nmin = np.min(Nvalues)
        Nmax = np.max(Nvalues)
        filename = os.path.join(output_dir, f"E_distrib_{Nbre_simus}simulations_N={Nmin}_a_{Nmax}_T={T}K_steps={step:.1e}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        
        plt.show()
     
def plot_mu_vs_T(T_values, mu_values, L_box, E_f):
    """
    Fonction pour tracer le potentiel chimique adimensionné en fonction de la température.
    
    Args:
        T_values (array): Liste des températures
        mu_values (array): Liste des potentiels chimiques adimensionnés correspondants
        L_box: vraie taille physique
        E_f: adimensionnée
    Outputs:
        Un graphique affichant mu_adim en fonction de T
    """
    
    plt.figure(figsize=(8,6))
    plt.plot(T_values, mu_values, 'r+',label="mu estimés par simulation")
    plt.plot(T_values, mu_adim_fct(L_box,T_values,E_f), label="valeurs théorique")
    plt.title('Potentiel chimique adimensionné en fonction de T', fontsize=title)
    plt.xlabel('T (K)', fontsize=label)
    plt.ylabel('mu_adim', fontsize=label)
    plt.legend(fontsize=legend)
    plt.grid()
    plt.show()