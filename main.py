import numpy as np
import yaml
import matplotlib.pyplot as plt
import argparse
import os

from MC import gen_cfg, accepte_cfg, modif_occupation_arr
from parameters import h, hbar, k_b, m_e, conv_J_eV
from CondInit import CI_lowest_E, create_n_max, Energy_Fermi_adim, wave_vector_Fermi_adim, \
    Energy_unit, wave_vector_unit, lambda_th, L_box_unit, kbT_adim, mu_adim_fct
from plots import plot_occupation, plot_energy_distribution, plot_energy_distribution_multiT, \
    plot_energy_distribution_multiN, plot_mu_vs_T
from fit import fit_fermi_dirac_mu


def load_input(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def simpleMC(input_file = "input.yaml"):
    """
    La fonction principale qui excécute la simulation Monte Carlo
    """
    # Lecture des paramètres depuis le fichier YAML
    print(f"Loading configuration from file : {input_file}")
    config = load_input(input_file)

    T = config["T"]
    num_steps = config["step"]
    L = config["L"]           # Adimensionée
    N = int(config["N"]) // 2 # On divise par 2 pour avoir le nombre effectif d'e- (sans spin)
    freq_save = config.get("freq_save", 100)  # Fréquence de sauvegarde des occupations
    
    # Calcul des grandeurs physiques de la simulation
    L_box = L*L_box_unit(N,100) # Vraie taille de la boite à T=100K (référence)
    E0 = Energy_unit(L_box)
    k0 = wave_vector_unit(L_box)
    #  Calcul des grandeurs adimensionnées de la simulation
    T_adim = kbT_adim(L_box, T) # = k_b * T / E0
    E_f = Energy_Fermi_adim(2*N)  # = Nréel / (2*pi)
    k_f = wave_vector_Fermi_adim(E_f) # = sqrt(E_f)
    n_max = create_n_max(E_f, T_adim) # ~ np.sqrt(E_F + kbT) * 1.5
    mu_adim = mu_adim_fct(L_box, T, E_f)
    
    # Affichage des paramètres pour l'initialisation de la configuration
    print(f"T = {T} K")
    print(f"N = {2*N} (avec spin)")
    print(f"L = {L_box*1e9:.2e} nm")
    print(f"l_th = {lambda_th(T)*1e9:.2e} nm")
    print(f"distance inter-électrons: d = {L_box/np.sqrt(N)*1e9:.2e} nm")
    print(f"E0 = {E0*conv_J_eV:.2e} eV")
    print(f"E_F = {E_f*E0*conv_J_eV:.2e} eV")
    print(f"E_F_adim = {E_f:.2e} ")
    print(f"mu_adim = {mu_adim:.2f}")
    print(f"Maximal quantum number n_max = {n_max:.0f}")
    print(f"Number of MC steps = {num_steps}")
    
    # Initialisation des listes et variables 
    occupation_arr = np.zeros((2 * n_max + 1 , 2 * n_max + 1))
    saved_occuaptions = []       # Liste où on stocke les occupations à chaque étape
    step = 0
    
    # Génération de la configuration initiale
    n1_list, n2_list = CI_lowest_E(N, n_max)
    occupation_step = np.zeros((2 * n_max + 1 , 2 * n_max + 1))
    for i in range(N):
        occupation_step[n1_list[i]+n_max, n2_list[i]+n_max] += 1
    
    print("Initial occupation state:")
    print(occupation_step)  
    
    print("Starting Monte Carlo simulation...")
    # Début de l'algorithme Monte Carlo
    number_of_acceptations = 0
    for i in range(num_steps):
        if i % (num_steps // 10) == 0:
            print(f"Progress: {i / num_steps * 100:.1f}%")
        
        if step % freq_save == 0:
            saved_occuaptions.append(occupation_step.copy())  # on stocke la matrice
        
        # Génération d'une nouvelle configuration proposée
        n1_new, n2_new, particle = gen_cfg(n1_list, n2_list, occupation_step, n_max, N)
        
        # Acceptation ou refus de la nouvelle configuration
        if accepte_cfg(n1_list, n2_list, n1_new, n2_new, particle, T_adim):
            accepted = True
            modif_occupation_arr(occupation_arr, occupation_step, accepted, n1_list, n2_list, n1_new, n2_new, n_max, particle)
            n1_list = n1_new
            n2_list = n2_new
            
            number_of_acceptations += 1
        else:
            accepted = False    
            modif_occupation_arr(occupation_arr, occupation_step, accepted, n1_list, n2_list, n1_new, n2_new, n_max, particle)
        
        step += 1
        
    print("Monte Carlo simulation completed.")
    
    # Sauvegarde finale
    os.makedirs("animations/simpleMC", exist_ok=True)
    np.savez_compressed("animations/simpleMC/occupations.npz", *saved_occuaptions)
    
    print("Final occupation state:")
    print(occupation_step)  
    
    print(f"Simulation completed. Acceptance ratio: {number_of_acceptations / num_steps * 100:.2f}%")
    
    ## Trace les graphiques 
    
    plot_occupation(occupation_arr, n_max, step, T)
    occ, energy = plot_energy_distribution(occupation_arr, n_max, E_f, step, N, T, L_box)

    popt, pcov, mask = fit_fermi_dirac_mu(energy, occ, T_adim, E_f)
    mu_fit = popt
    print(f"mu_fit = {mu_fit:.2e}")

    return()

def MC_multiT(input_file = "input.yaml"):
    """
    exécute plusieurs simulation MC pour différentes T° spécifiées dans l'input
    """
    # Lecture des paramètres depuis le fichier YAML
    print(f"Loading configuration from file : {input_file}")
    config = load_input(input_file)

    Tmin = config["Tmin"]
    Tmax = config["Tmax"]
    deltaT = config["deltaT"]
    T_values = np.arange(Tmin, Tmax + deltaT, deltaT) 
    freq_save = config.get("freq_save", 100)  # Fréquence de sauvegarde des occupations 
    num_steps = config["step"]
    L = config["L"]           # Adimensionée
    N = int(config["N"]) // 2 # On divise par 2 pour avoir le nombre effectif d'e- (sans spin)
    
    list_mu_adim = []  # Liste pour stocker les mu_adim estimés pour chaque T
    list_mu_adim_fit = []
    
    for T in T_values:
        if T == 0:
            T = 0.1 # éviter T=0K strict qui pose pb dans lambda_th
        print(f"\nStarting simulation for T = {T:.0f} K")
        # Calcul des grandeurs physiques pour chaque simulation
        L_box = L*L_box_unit(N,100) # Vraie taille de la boite à T=100K (référence)
        E0 = Energy_unit(L_box)
        k0 = wave_vector_unit(L_box)
        #  Calcul des grandeurs adimensionnées de la simulation
        T_adim = kbT_adim(L_box, T) # = k_b * T / E0
        E_f = Energy_Fermi_adim(2*N)  # = Nréel / (2*pi)
        k_f = wave_vector_Fermi_adim(E_f) # = sqrt(E_f)
        n_max = create_n_max(E_f, T_adim) # ~ np.sqrt(E_F + kbT) * 1.5
        
        # Affichage des paramètres pour l'initialisation de la configuration
        print(f"T = {T} K")
        print(f"N = {2*N} (avec spin)")
        print(f"L = {L_box*1e9:.2e} nm")
        print(f"l_th = {lambda_th(T)*1e9:.2e} nm")
        print(f"distance inter-électrons: d = {L_box/np.sqrt(N)*1e9:.2e} nm")
        print(f"E0 = {E0*conv_J_eV:.2e} eV")
        print(f"E_F = {E_f*E0*conv_J_eV:.2e} eV")
        print(f"E_F_adim = {E_f:.2e} ")
        print(f"Maximal quantum number n_max = {n_max:.0f}")
        print(f"Number of MC steps = {num_steps}")
        
        # Initialisation des listes et variables 
        occupation_arr = np.zeros((2 * n_max + 1 , 2 * n_max + 1))
        saved_occupations = []    # Liste où on stocke les occupations à chaque étape
        step = 0
        
        
        # Génération de la configuration initiale
        n1_list, n2_list = CI_lowest_E(N, n_max)
        occupation_step = np.zeros((2 * n_max + 1 , 2 * n_max + 1))
        for i in range(N):
            occupation_step[n1_list[i]+n_max, n2_list[i]+n_max] += 1
        
        #print("Initial occupation state:")
        #print(occupation_step)  
        
        print(f"Starting simulation for T = {T:.0e}K...")
        # Début de l'algorithme Monte Carlo
        number_of_acceptations = 0
        for i in range(num_steps):
            if i % (num_steps // 10) == 0:
                print(f"Progress: {i / num_steps * 100:.1f}%")
            
            if step % freq_save == 0:
                saved_occupations.append(occupation_step.copy())  # on stocke la matrice
            
            # Génération d'une nouvelle configuration proposée
            n1_new, n2_new, particle = gen_cfg(n1_list, n2_list, occupation_step, n_max, N)
            
            # Acceptation ou refus de la nouvelle configuration
            if accepte_cfg(n1_list, n2_list, n1_new, n2_new, particle, T_adim):
                accepted = True
                modif_occupation_arr(occupation_arr, occupation_step, accepted, n1_list, n2_list, n1_new, n2_new, n_max, particle)
                n1_list = n1_new
                n2_list = n2_new
                
                number_of_acceptations += 1
            else:
                accepted = False    
                modif_occupation_arr(occupation_arr, occupation_step, accepted, n1_list, n2_list, n1_new, n2_new, n_max, particle)
            
            step += 1
        
        print(f"simulation T= {T:.0f} K completed.")
        
        # Sauvegarde finale
        os.makedirs("animations/multiT", exist_ok=True)
        np.savez_compressed(f"animations/multiT/occupations_T={T}K.npz", *saved_occupations)
        
        #print("Final occupation state:")
        #print(occupation_step)  
        print(f"Simulation completed. Acceptance ratio: {number_of_acceptations / num_steps * 100:.2f}%")
        
        ## Trace les graphiques 
        
        #plot_occupation(occupation_arr, n_max, step, T)
        mu_adim_estime, occ, energy = plot_energy_distribution_multiT(occupation_arr, n_max, E_f, step, N, T, T_values, L_box)
        list_mu_adim.append(mu_adim_estime)

        popt, pcov, mask = fit_fermi_dirac_mu(energy, occ, T_adim, E_f, plot_result=False)
        mu_fit = float(popt[0])
        list_mu_adim_fit.append(mu_fit)
        print(list_mu_adim_fit)
        
    plot_mu_vs_T(T_values, list_mu_adim, list_mu_adim_fit, L_box, E_f)

    
    return()

def MC_multiN(input_file = "input.yaml"):
    """
    exécute plusieurs simulation MC pour différentes valeurs de N (spécifiées dans l'input)
    """
    # Lecture des paramètres depuis le fichier YAML
    print(f"Loading configuration from file : {input_file}")
    config = load_input(input_file)

    T = config["T"]
    Nmin = config["Nmin"]
    Nmax = config["Nmax"]
    deltaN = config["deltaN"]
    N_values = np.arange(Nmin, Nmax + deltaN, deltaN) // 2 # On divise par 2 pour avoir le nombre effectif d'e- (sans spin)
    num_steps = config["step"]
    L = config["L"]    # Adimensionée
    
    #definition of plot colors (needed to plot Ef and corresponding distribution w/ same color)
    color_array = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    i_color = 0
    for N in N_values:
        color = color_array[i_color % len(color_array)]
        i_color += 1
        if N == 0:
            print(f"\n N = 0, skipping this simulation.")
            continue
        # Calcul des grandeurs physiques pour chaque simulation
        L_box = L*L_box_unit(N,100) # Vraie taille de la boite à T=100K (référence)
        E0 = Energy_unit(L_box)
        k0 = wave_vector_unit(L_box)
        #  Calcul des grandeurs adimensionnées de la simulation
        T_adim = kbT_adim(L_box, T) # = k_b * T / E0
        E_f = Energy_Fermi_adim(2*N)  # = Nréel / (2*pi)
        k_f = wave_vector_Fermi_adim(E_f) # = sqrt(E_f)
        n_max = create_n_max(E_f, T_adim) # ~ np.sqrt(E_F + kbT) * 1.5
        
        # Affichage des paramètres pour l'initialisation de la configuration
        print(f"T = {T} K")
        print(f"N = {2*N} (avec spin)")
        print(f"L = {L_box*1e9:.2e} nm")
        print(f"l_th = {lambda_th(T)*1e9:.2e} nm")
        print(f"distance inter-électrons: d = {L_box/np.sqrt(N)*1e9:.2e} nm")
        print(f"E0 = {E0*conv_J_eV:.2e} eV")
        print(f"E_F = {E_f*E0*conv_J_eV:.2e} eV")
        print(f"E_F_adim = {E_f:.2e} ")
        print(f"Maximal quantum number n_max = {n_max:.0f}")
        print(f"Number of MC steps = {num_steps}")
        
        # Initialisation des listes et variables 
        occupation_arr = np.zeros((2 * n_max + 1 , 2 * n_max + 1))
        saved_occupations = []    # Liste où on stocke les occupations à chaque étape
        step = 0
        
        # Génération de la configuration initiale
        n1_list, n2_list = CI_lowest_E(N, n_max)
        occupation_step = np.zeros((2 * n_max + 1 , 2 * n_max + 1))
        for i in range(N):
            occupation_step[n1_list[i]+n_max, n2_list[i]+n_max] += 1
        
        #print("Initial occupation state:")
        #print(occupation_step)  
        
        print(f"Starting simulation for N = {N:.0e}...")
        # Début de l'algorithme Monte Carlo
        number_of_acceptations = 0
        for i in range(num_steps):
            if i % (num_steps // 10) == 0:
                print(f"Progress: {i / num_steps * 100:.1f}%")
            
            if step % 100 == 0:
                saved_occupations.append(occupation_step.copy())  # on stocke la matrice
            
            # Génération d'une nouvelle configuration proposée
            n1_new, n2_new, particle = gen_cfg(n1_list, n2_list, occupation_step, n_max, N)
            
            # Acceptation ou refus de la nouvelle configuration
            if accepte_cfg(n1_list, n2_list, n1_new, n2_new, particle, T_adim):
                accepted = True
                modif_occupation_arr(occupation_arr, occupation_step, accepted, n1_list, n2_list, n1_new, n2_new, n_max, particle)
                n1_list = n1_new
                n2_list = n2_new
                
                number_of_acceptations += 1
            else:
                accepted = False    
                modif_occupation_arr(occupation_arr, occupation_step, accepted, n1_list, n2_list, n1_new, n2_new, n_max, particle)
            
            step += 1
        
        print(f"simulation N = {N:.0f} completed.")
        
        # Sauvegarde finale
        os.makedirs("animations/multiN", exist_ok=True)
        np.savez_compressed(f"animations/multiN/occupations_N={N}K.npz", *saved_occupations)
        
        #print("Final occupation state:")
        #print(occupation_step)  
        print(f"Simulation completed. Acceptance ratio: {number_of_acceptations / num_steps * 100:.2f}%")
        
        ## Trace les graphiques 
        
        #plot_occupation(occupation_arr, n_max, step, T)
        plot_energy_distribution_multiN(occupation_arr, n_max, E_f, step, T, N, N_values, L_box, color)
    
    return()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for simulation input file.")
    os.makedirs("animations", exist_ok=True)
    # Définition des arguments du parser
    parser.add_argument("--file", type=str, required=False, default="input.yaml", \
        help="Nom ou chemin du fichier de configuration YAML")
    
    parser.add_argument("--simu", type=str, required=True, default="simple", \
        help="Une seule simu ? ou bien plusieurs T, N, V en parallèle ? (options: simple / multiT / multiN / multiV )")
    
    # Lecture des arguments
    args = parser.parse_args()
    input_file = args.file
    simu_type = args.simu
    
    if simu_type.lower() == "simple":
        simpleMC(input_file)
    elif simu_type.lower() == "multit":
        MC_multiT(input_file)
    elif simu_type.lower() == "multin":
        MC_multiN(input_file)
    elif simu_type.lower() == "multiv":
        print("Option multiV pas encore implémentée.")
    else:
        print("Choisir le paramètre simu parmi: simple / multiT / multiN / multiV.")