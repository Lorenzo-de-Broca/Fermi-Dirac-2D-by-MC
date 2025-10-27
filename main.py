import numpy as np
import yaml
import matplotlib.pyplot as plt
from MC import gen_cfg, accepte_cfg, modif_occupation_list
from parameters import h, hbar, k_b, m_e, E0, lambda_th
from CondInit import CI, create_n_max, Energy_Fermi, wave_vector_Fermi

def load_input(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    La fonction principale qui excécute la simulation Monte Carlo
    """
    
    input_file = input("Choose the YAML config file: ")
    config = load_input(input_file)
    
    # Lecture des paramètres depuis config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    T = config["T"]
    num_steps = config["step"]
    L = config["L"]
    N = config["N"] 

    E_f = Energy_Fermi(N)
    k_f = wave_vector_Fermi(E_f)
    n_max = create_n_max(N, E_f)
    occupation_list = np.zeros((2 * n_max, 2 * n_max))

    # Génération de la configuration initiale
    n1_list, n2_list = CI(N, L, T, E_f, k_f)

    # Début de l'algorithme Monte Carlo
    for step in range(num_steps):
        n1_new, n2_new, particle = gen_cfg(n1_list, n2_list, n_max, N)
        if accepte_cfg(n1_list, n2_list, n1_new, n2_new, particle, E0, T):
            n1_list = n1_new
            n2_list = n2_new
            modif_occupation_list(occupation_list, n1_new, n2_new)
        else:
            pass  # À compléter selon la logique souhaitée


    
    
    


if __name__ == "__main__":
    main()
    #module physical_parameters pour mettre toutes les grandeurs physiques (H, kb etc)

    #def CI(N,T,L=??):


    #def plot(...):