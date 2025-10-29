import numpy as np
import yaml
import matplotlib.pyplot as plt

from MC import gen_cfg, accepte_cfg, modif_occupation_arr
from parameters import h, hbar, k_b, m_e, eV
from CondInit import CI, create_n_max, Energy_Fermi, wave_vector_Fermi, Energy_unit, L_box_std, kbT_adim
from plots import plot_occupation, plot_energy_distribution

def load_input(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    La fonction principale qui excécute la simulation Monte Carlo
    """
    # Lecture des paramètres depuis le fichier YAML
    input_file = input("Choose the YAML config file: ")
    config = load_input(input_file)

    T = config["T"]
    num_steps = config["step"]
    L = config["L"]                 # A dimensioné
    N = int(config["N"]/2)   # On divise par 2 pour avoir le nombre de particules par spin 

    # Calcul des grandeurs physiques de la simulation
    L_box = L*L_box_std(config["N"], T)                     # Vrai taille de la boite
    E0 = Energy_unit(L_box)
    T_adim = kbT_adim(L_box, T)
    E_f = Energy_Fermi(N)
    k_f = wave_vector_Fermi(E_f)
    n_max = create_n_max(E_f, L_box, T)
    
    # Affichage des paramètres pour l'initialisation de la configuration
    print(f"Température T = {T} K")
    print(f"Number of particules N = {N*2} (avec spin)")
    print(f"Size of the box L = {L_box*1e9:.2e} nm")
    print(f"Fermi energy E_f = {E_f*E0*eV:.2e} eV")
    print(f"Adim Fermi energy E_f = {E_f:.2e} adim")
    print(f"Maximal quantum number n_max = {n_max:.0f}")
    print(f"Number of MC steps = {num_steps}")
    
    # Initialisation des listes et variables 
    occupation_arr = np.zeros((2 * n_max + 1 , 2 * n_max + 1))
    step = 0
    
    # Génération de la configuration initiale
    n1_list, n2_list = CI(N, n_max)
    occupation_step = np.zeros((2 * n_max + 1 , 2 * n_max + 1))
    for i in range(N):
        occupation_step[n1_list[i]+n_max, n2_list[i]+n_max] += 1
    
    print("Initial occupation state:")
    print(occupation_step)  
    
    print("Starting Monte Carlo simulation...")
    # Début de l'algorithme Monte Carlo
    for i in range(num_steps):
        if i % (num_steps // 10) == 0:
            print(f"Progress: {i / num_steps * 100:.1f}%")
        step += 1
        
        # Génération d'une nouvelle configuration proposée
        n1_new, n2_new, particle = gen_cfg(n1_list, n2_list, occupation_step, n_max, N)
        
        # Acceptation ou refus de la nouvelle configuration
        if accepte_cfg(n1_list, n2_list, n1_new, n2_new, particle, T_adim):
            accepted = True
            modif_occupation_arr(occupation_arr, occupation_step, accepted, n1_list, n2_list, n1_new, n2_new, n_max, particle)
            n1_list = n1_new
            n2_list = n2_new
        
        else:
            accepted = False    
            modif_occupation_arr(occupation_arr, occupation_step, accepted, n1_list, n2_list, n1_new, n2_new, n_max, particle)

    ## Trace les graphiques 
    
    plot_occupation(occupation_arr, n_max, step, T)
    plot_energy_distribution(occupation_arr, n_max, E_f, step, T)
    
    return()


if __name__ == "__main__":
    
    main()