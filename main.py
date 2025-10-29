import numpy as np
import yaml
import matplotlib.pyplot as plt
import argparse

from MC import gen_cfg, accepte_cfg, modif_occupation_arr
from parameters import h, hbar, k_b, m_e, conv_J_eV
from CondInit import CI_lowest_E, create_n_max, Energy_Fermi_adim, wave_vector_Fermi_adim, \
    Energy_unit, wave_vector_unit, lambda_th, L_box_unit, kbT_adim
from plots import plot_occupation, plot_energy_distribution

def load_input(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(input_file):
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
    
    # Calcul des grandeurs physiques de la simulation
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
    saved_occuaptions = []                                           # Liste où on stocke les occupations à chaque étape
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
        
        if step % 100 == 0:
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
    np.savez_compressed("occupations.npz", *saved_occuaptions)
    
    #print("Final occupation state:")
    #print(occupation_step)  
    print(f"Simulation completed. Acceptance ratio: {number_of_acceptations / num_steps * 100:.2f}%")
    
    ## Trace les graphiques 
    
    plot_occupation(occupation_arr, n_max, step, T)
    plot_energy_distribution(occupation_arr, n_max, E_f, step, T, L_box)
    
    return()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parser for simulation input file.")
    
    # Définition des arguments du parser
    parser.add_argument("--file", type=str, required=True, help="Nom ou chemin du fichier de configuration YAML")

    # Lecture des arguments
    args = parser.parse_args()

    input_file = args.file

    main(input_file)