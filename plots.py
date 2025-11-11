import matplotlib.pyplot as plt
import numpy as np
import os

from CondInit import kbT_adim, mu_adim_fct, Fermi_Dirac_distribution, Energy_Fermi_adim, Maxwell_Boltzmann_distribution


# On définit la taille des légendes sur les figures 
title = 14
label = 12
legend = 12
ticks = 14
markersize = 12

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
    y_Maxwelle_Boltzmann = Maxwell_Boltzmann_distribution(x_Fermi_Dirac, mu_adim,T_adim) # théorique
    Fermi_Dirac_MC = occupation_levels_masked/(degenerescence_levels_masked*step)
    
    plt.figure(figsize=(8,6))
    #plt.plot(energy_levels, occupation_levels, "r", markersize=8, label='Occupation des niveaux d\'énergie')
    plt.plot(energy_levels_masked, Fermi_Dirac_MC, "b+", markersize=7, label='Occupation par niveau d\'énergie')
    plt.plot(x_E_Fermi, y_E_Fermi, 'r--', label=f'Énergie de Fermi : {Ef:.2f} (adim)')
    #plt.plot(x_E_Fermi_kbT, y_E_Fermi, 'g--', label=f'Énergie de Fermi + k_b*T : {Ef + T_adim:.2f} adimensionnée')

    plt.plot(x_Fermi_Dirac, y_Fermi_Dirac, 'k-', label='Distribution de Fermi-Dirac théorique')
    plt.plot(x_Fermi_Dirac, y_Maxwelle_Boltzmann, 'g-', label='Distribution de Maxwell-Boltzmann théorique')
    #plt.legend(fontsize=legend)
    plt.legend(fontsize=legend) #, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'Energie des particules (moy. sur {step:.1e} steps), pour N={N*2} e- et T={T}K', fontsize=title)
    plt.xlabel('Énergie (E0)', fontsize=label)
    plt.ylabel('Occupation', fontsize=label)
    # Fixer les limites des axes
    plt.xlim(0, np.max(energy_levels_masked) * 1.1)
    plt.ylim(-0.05, 1.1)
    plt.grid()
    
    # Sauvegarde
    filename1 = os.path.join(output_dir, f"FD_N{N*2}_T{T:.0e}K_step{step:.1e}.png")
    #plt.tight_layout()
    plt.savefig(filename1, dpi=300, bbox_inches="tight")
    
    plt.show()
      
    # Graphique de la dégénérescence des niveaux d'énergie
    plt.plot(energy_levels_masked, degenerescence_levels_masked, "g+", markersize=8, label='Dégénérescence des niveaux d\'énergie')
    plt.legend(fontsize=legend)
    plt.title(f'Dégénérescence des niveaux d\'énergie, T={T}K', fontsize=title)
    plt.xlabel('Énergie (E0)', fontsize=label)
    plt.ylabel('Dégénérescence', fontsize=label)
    plt.grid()
    
    plt.show()
    return(Fermi_Dirac_MC, energy_levels_masked)

def plot_energy_distribution_multiT(occupation_arr, n_max, Ef, step, N, T, Tvalues, L_box, color):
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
    
    if step <= 0:
        raise ValueError("Le paramètre 'step' doit être > 0 pour calculer les occupations (éviter division par zéro).")

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

    distrib_FD = occupation_levels_masked/(degenerescence_levels_masked*step)

    if T == np.min(Tvalues):
        #TODO supprimer plot E_F ?
        plt.figure(figsize=(8,6))
        # Pour tracer l'énergie de Fermi
        y_fermi = [0,np.max(distrib_FD)*1.1]
        x_fermi = [Ef, Ef]
        #plt.plot(x_fermi, y_fermi, 'r--', label=f'E_F = {Ef:.2f} E0')
    
        
    plt.plot(energy_levels_masked, distrib_FD, color+'+', \
        label=f'T = {T:.0f}K')#, markersize=5, "b+",

    # Calcul de la distribution de Fermi-Dirac théorique pour comparer 
    x_Fermi_Dirac = np.linspace(0, np.max(energy_levels_masked), 1000)
    
    mu_adim = mu_adim_fct (L=L_box, T=T, E_f=Ef)
    T_adim = kbT_adim(L_box, T)
    y_Fermi_Dirac = Fermi_Dirac_distribution(x_Fermi_Dirac, mu_adim,T_adim) # théorique
    
    plt.plot(x_Fermi_Dirac, y_Fermi_Dirac, color)#, label=f'F-D théorique')
    
    # On cherche les indices autour de 1/2 (robuste)
    target = 0.5
    diff = distrib_FD - target
    # indices où la densité passe de >0.5 à <0.5 (ou l’inverse)
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_changes) > 0:
        # On prend le premier crossing
        i_below = sign_changes[0]
        i_above = i_below + 1
    else:
        # Pas de crossing clair : on choisit le point le plus proche de 0.5
        i_min = int(np.argmin(np.abs(diff)))
        # choisir un voisin pour interpolation (préférence vers la droite si possible)
        if i_min + 1 < len(distrib_FD):
            i_below, i_above = i_min, i_min + 1
        elif i_min - 1 >= 0:
            i_below, i_above = i_min - 1, i_min
        else:
            # cas extrême : pas assez de points
            # debug info
            print("[plots.plot_energy_distribution_multiT] Debug: distrib_FD min,max =", np.min(distrib_FD), np.max(distrib_FD))
            print("energy_levels_masked:", energy_levels_masked)
            raise ValueError("Impossible d'estimer µ : pas de points voisins pour interpolation.")

    E_below = energy_levels_masked[i_below]
    E_above = energy_levels_masked[i_above]
    D_below = distrib_FD[i_below]
    D_above = distrib_FD[i_above]
    # Interpolation linéaire pour trouver E où Densité = 1/2
    if np.isclose(D_above, D_below):
        mu_adim_estime = 0.5 * (E_below + E_above)
    else:
        mu_adim_estime = E_below + (target - D_below) * (E_above - E_below) / (D_above - D_below)
    #print("Énergie juste en dessous :", E_below)
    #print("Énergie juste au-dessus :", E_above)
    print(f"Estimation du potentiel chimique : μ ≈ {mu_adim_estime:.2f} E0")

    if T == np.max(Tvalues):
        #affichage des courbes pour chaque T
        plt.legend(fontsize=legend)
        #plt.legend(fontsize=legend, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'Energie des particules (moy. sur {step:.0e} steps) pour différents T (N = {N*2})', fontsize=title)
        plt.xlabel('Énergie (E0)', fontsize=label)
        plt.xlim(0, 4*Ef)
        plt.ylabel('Occupation', fontsize=label)
        plt.grid()
        #save figures 
        output_dir = "fig/multiT"
        os.makedirs(output_dir, exist_ok=True)
        Nbre_simus = np.size(Tvalues)
        Tmin = np.min(Tvalues)
        Tmax = np.max(Tvalues)
        filename = os.path.join(output_dir, f"FD_{Nbre_simus}simus_N{N*2}_T{Tmin:.0f}_a_{Tmax:.0f}K_steps{step:.1e}.png")
        #plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        
        plt.show()
    return mu_adim_estime, distrib_FD, energy_levels_masked
     
def plot_energy_distribution_multiN(occupation_arr, n_max, Ef, step, T, N, Nvalues, L_box, color):
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
    
    if step <= 0:
        raise ValueError("Le paramètre 'step' doit être > 0 pour calculer les occupations (éviter division par zéro).")

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

    distrib_FD = occupation_levels_masked/(degenerescence_levels_masked*step)

    if N == np.min(Nvalues):
        plt.figure(figsize=(8,6))
    # Pour tracer l'énergie de Fermi
    #y_fermi = [0,np.max(distrib_FD)*1.1]
    y_fermi = [0,1] #sinon chaque Ef a une taille différente 
    x_fermi = [Ef, Ef]
    #plt.plot(x_fermi, y_fermi, color+'--', label=f'E_F = {Ef:.2f} E0')
    plt.plot(energy_levels_masked, distrib_FD, color+'+', \
        label=f'N = {N*2:.0f}')#, markersize=5, "b+",
        # Calcul de la distribution de Fermi-Dirac théorique pour comparer 
    x_Fermi_Dirac = np.linspace(0, np.max(energy_levels_masked), 1000)
    
    mu_adim = mu_adim_fct (L=L_box, T=T, E_f=Ef)
    T_adim = kbT_adim(L_box, T)
    y_Fermi_Dirac = Fermi_Dirac_distribution(x_Fermi_Dirac, mu_adim,T_adim) # théorique
    
    plt.plot(x_Fermi_Dirac, y_Fermi_Dirac, color)#, label='F-D théorique')
    
    # On cherche les indices autour de 1/2 (robuste)
    target = 0.5
    diff = distrib_FD - target
    # indices où la densité passe de >0.5 à <0.5 (ou l’inverse)
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_changes) > 0:
        # On prend le premier crossing
        i_below = sign_changes[0]
        i_above = i_below + 1
    else:
        # Pas de crossing clair : on choisit le point le plus proche de 0.5
        i_min = int(np.argmin(np.abs(diff)))
        # choisir un voisin pour interpolation (préférence vers la droite si possible)
        if i_min + 1 < len(distrib_FD):
            i_below, i_above = i_min, i_min + 1
        elif i_min - 1 >= 0:
            i_below, i_above = i_min - 1, i_min
        else:
            # cas extrême : pas assez de points
            print("[plots.plot_energy_distribution_multiN] Debug: distrib_FD min,max =", np.min(distrib_FD), np.max(distrib_FD))
            print("energy_levels_masked:", energy_levels_masked)
            raise ValueError("Impossible d'estimer µ : pas de points voisins pour interpolation.")

    E_below = energy_levels_masked[i_below]
    E_above = energy_levels_masked[i_above]
    D_below = distrib_FD[i_below]
    D_above = distrib_FD[i_above]
    # Interpolation linéaire pour trouver E où Densité = 1/2
    if np.isclose(D_above, D_below):
        mu_adim_estime = 0.5 * (E_below + E_above)
    else:
        mu_adim_estime = E_below + (target - D_below) * (E_above - E_below) / (D_above - D_below)
    #print("Énergie juste en dessous :", E_below)
    #print("Énergie juste au-dessus :", E_above)
    print(f"Estimation du potentiel chimique : μ ≈ {mu_adim_estime:.2f} E0")
    
    if N == np.max(Nvalues):
        #affichage des courbes pour chaque T
        plt.legend(fontsize=legend)
        #plt.legend(fontsize=legend, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'Energie des particules (moy. sur {step:.1e} steps) pour différents N (T = {T:.0f}K)', fontsize=title)
        plt.xlabel('Énergie (E0)',fontsize=label)
        plt.xlim(0, 4*Ef)
        plt.ylabel('Occupation', fontsize=label)
        plt.grid()
        #save figures 
        output_dir = "fig/multiN"
        os.makedirs(output_dir, exist_ok=True)
        Nbre_simus = np.size(Nvalues)
        Nmin = np.min(Nvalues)
        Nmax = np.max(Nvalues)
        filename = os.path.join(output_dir, f"FD_{Nbre_simus}simus_N{Nmin*2}_a_{Nmax*2}_T{T:.0f}K_steps{step:.1e}.png")
        #plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        
        plt.show()
    return mu_adim_estime, distrib_FD, energy_levels_masked
     
def plot_mu_vs_T(T_values, mu_values, mu_values_fit, L_box, E_f, step):
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

    valT = np.linspace(np.min(T_values),np.max(T_values),1000)
    plt.plot(valT, mu_adim_fct(L_box,valT,E_f), label="μ(T) théorique")
    plt.plot(T_values, mu_values, 'r+', markersize = markersize, label="μ estimé par recherche de FD(E) = 1/2")
    plt.plot(T_values, mu_values_fit, 'g+', markersize = markersize, label="μ estimé par fit de Fermi-Dirac")
    N = 2*np.pi*E_f
    plt.title(f'Potentiel chimique μ en fonction de T (N = {N})', fontsize=title)
    plt.xlabel('T (K)', fontsize=label)
    plt.ylabel('mu_adim', fontsize=label)
    plt.legend(fontsize=legend)
    plt.grid()
    #save figures
    output_dir = "fig/multiT"
    os.makedirs(output_dir, exist_ok=True)
    Nbre_simus = np.size(T_values)
    Tmin = np.min(T_values)
    Tmax = np.max(T_values)
    filename = os.path.join(output_dir, f"μ_{Nbre_simus}simus_N{N*2}_T{Tmin:.0f}_a_{Tmax:.0f}K_steps{step:.1e}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    
    plt.show()
    

def plot_mu_vs_N(N_values, mu_values, mu_values_fit, L_box, T, step):
    """
    Fonction pour tracer le potentiel chimique adimensionné en fonction de la température.
    
    Args:
        N_values (array): Liste des N
        mu_values (array): Liste des potentiels chimiques adimensionnés correspondants
        L_box: vraie taille physique
        E_f: adimensionnée
    Outputs:
        Un graphique affichant mu_adim en fonction de N pour T fixé
    """
    
    plt.figure(figsize=(8,6))

    valN = np.linspace(np.min(N_values),np.max(N_values),1000) #N sont tjrs divisés par 2 pr le spin
    plt.plot(valN*2, mu_adim_fct(L_box,T,Energy_Fermi_adim(valN*2)), label="μ(N) théorique")
    plt.plot(N_values*2, mu_values, 'r+', markersize = markersize, label="μ estimé par recherche de FD(E) = 1/2")
    plt.plot(N_values*2, mu_values_fit, 'g+', markersize = markersize, label="μ estimé par fit de Fermi-Dirac")
    plt.title(f'Potentiel chimique μ en fonction de N (T = {T} K)', fontsize=title)
    plt.xlabel('N', fontsize=label)
    plt.ylabel('mu_adim', fontsize=label)
    plt.legend(fontsize=legend)
    plt.grid()
    #save figures
    output_dir = "fig/multiN"
    os.makedirs(output_dir, exist_ok=True)
    Nbre_simus = np.size(N_values)
    Nmin = np.min(N_values)
    Nmax = np.max(N_values)
    filename = os.path.join(output_dir, f"μ_{Nbre_simus}simus_N{Nmin*2}_a_{Nmax*2}_T{T:.0f}K_steps{step:.1e}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()