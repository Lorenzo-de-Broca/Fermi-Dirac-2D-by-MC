import numpy as np
import yaml
import matplotlib.pyplot as plt

from MC import gen_cfg, accepte_cfg, modif_occupation_arr
from parameters import h, hbar, k_b, m_e
from CondInit import CI, create_n_max, Energy_Fermi, wave_vector_Fermi, Energy_unit
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
    L = config["L"]
    N = config["N"]/2   # On divise par 2 pour avoir le nombre de particules par spin 

    # Calcul des grandeurs physiques de la simulation
    E_f = Energy_Fermi(N)
    k_f = wave_vector_Fermi(E_f)
    n_max = create_n_max(E_f, L, T)
    E0 = Energy_unit(L)
    
    # Initialisation des listes et variables 
    print(f"N_max = {n_max:.0f}")
    occupation_arr = np.zeros((2 * n_max +1 , 2 * n_max + 1))
    step = 0
    
    # Génération de la configuration initiale
    n1_list, n2_list = CI(N, L, T, E_f, k_f)

    # Début de l'algorithme Monte Carlo
    for i in range(num_steps):
        step += 1
        n1_new, n2_new, particle = gen_cfg(n1_list, n2_list, n_max, N)
        
        if accepte_cfg(n1_list, n2_list, n1_new, n2_new, particle, E0, T):
            n1_list = n1_new
            n2_list = n2_new
            modif_occupation_arr(occupation_arr, n1_new, n2_new)
        
        else:
            modif_occupation_arr(occupation_arr, n1_list, n2_list)

    ## Trace les graphiques 
    
    plot_occupation(occupation_arr, n_max, step, T)
    plot_energy_distribution(occupation_arr, E0, n_max, step, T)
    
    return()


    
    
    


if __name__ == "__main__":
    
    main()