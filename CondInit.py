import numpy as np
from parameters import h, hbar, k_b, m_e
import sys
import os

# paramètres constants lors d'une simu
# N, L et T seront choisis dans le fichier input

"""/!\ grandeurs physiques NON adimensionnées /!\ """

def lambda_th(T):
    """ Retourne la longueur thermique de De Broglie à T > 0K """
    try:
        return np.sqrt(h*hbar)/np.sqrt(m_e * k_b * T)
    except:
        print("Warning: lambda_th(T) is defined only for T > 0K. In this simulation T = 0K so lambda_th is None.")
        return None
    
def L_box_unit(N,T = 100):
    """ Retourne la taille de boîte physique dite "de transition" pour N particules à T = 100K 
    telle que d_inter_e- ~ lambda (transition entre régime classique et quantique) """
    #pour T >> 100K, régime classique ; pour T <~ 100K, régime quantique 
    return np.sqrt(N) * lambda_th(T)

def L_box_std(N,T): # useless at the moment
    """ Retourne l'unité de taille de boîte, telle que lambda = d (distance interatomique) """
    return np.sqrt(N) * lambda_th(T)

def wave_vector_unit(Lbox):
    """ Retourne la valeur de l'unité de vecteur d'onde pour une boîte de taille L """
    return 2*np.pi/Lbox # unité de vecteur d'onde

def Energy_unit(Lbox):
    """ Retourne la valeur de l'unité d'énergie E0 (Joules) pour une boîte de taille L """
    return hbar**2/(2*m_e) * wave_vector_unit(Lbox)**2 # unité d'énergie

# -----------------------------------------------------------------------------------
"""/!\ grandeurs numériques ADIMENSIONNEES /!\ """

def kbT_adim(L,T):
    """ Retourne la valeur adimensionnée de k_b * T 
    inputs : L (longueur physique) et T (temp. physique) """
    E_0 = Energy_unit(L)
    return (k_b * T) / E_0

def mu_adim_fct(L,T,E_f):
    """Retourne le potentiel chimique adimensionné à la température T (en K)
    inputs: 
        - L : taille physique de la boîte
        - T : température physique en K   
        - E_f : énergie de Fermi adimensionnée
    """
    T_adim = kbT_adim(L,T)
    print("T_adim in mu_adim_fct:", T_adim)
    print("E_f in mu_adim_fct:", E_f)
    print("ln(exp(E_f/T_adim)):", np.log(1-np.exp(-E_f/T_adim)))
    mu_adim = E_f + T_adim * np.log(1-np.exp(-E_f/T_adim))
    return (mu_adim)

def Energy_Fermi_adim(N):
    """ Retourne l'énergie de Fermi adimensionnée """
    return N/(2*np.pi)

def wave_vector_Fermi_adim(N):
    """ Retourne le vecteur d'onde de Fermi adimensionné """
    E_F = Energy_Fermi_adim(N)
    return np.sqrt(E_F)

def create_n_max(E_F,kbT): # grandeurs ADIM
    """ Retourne la valeur maximale de n1 et n2 (tq E(n_max,0) = E_F + kbT) """
    """ CLP --> N = g_s * pi * n_F^2""" # spin pris en compte dans l'input atm : N := N/2
    n_F = int(np.ceil(np.sqrt(E_F)))
    n_max = int(np.ceil(np.sqrt(E_F + kbT) * 1.5)) #facteur 1.5 = marge de sécurité
    N = 2*np.pi*E_F
    # on veut n_max tq (2*n_max+1)^2 >= N (nb d'états possibles >= N)
    if (2*n_max+1)**2 >= N: 
        return n_max
    else: 
        return (np.sqrt(N)-1)/2
    
def CI_random(N, n_max): #useless at the moment
    """ Retourne les listes n1, n2 de N états générés uniformément (pas à 0K) """
    #tirage aléatoire sans remise de N couples (n1,n2), avec 0 <= n1,n2 <= 2*n_max
    try: 
        values = np.random.choice((2*n_max+1) * (2*n_max+1), size = N, replace = False) 
    except:
        print("Error in the generation of random initial conditions (CI_random): \
            N is too large compared to n_max. Please restart with")
        sys.exit()
    n1_list, n2_list = np.unravel_index(values, (2*n_max+1, 2*n_max+1))
    n1_list -= n_max #on passe de [0,2n_max] à [-n_max,n_max]
    n2_list -= n_max #on passe de [0,2n_max] à [-n_max,n_max]
    return n1_list, n2_list

def CI_lowest_E(N, n_max):
    """ Retourne les listes n1, n2 des états de + basse énergie (initialement occupés) """
    # liste tous les états possibles et les énergies correspondantes
    possible_n1 = np.arange(-n_max, n_max + 1)
    possible_n2 = np.arange(-n_max, n_max + 1)
    n1_grid, n2_grid = np.meshgrid(possible_n1, possible_n2, indexing='ij')
    E_grid = n1_grid**2 + n2_grid**2
    # regroupe toutes les infos de l'état dans un seul tableau (n1, n2, E)
    state = np.stack((n1_grid.ravel(), n2_grid.ravel(), E_grid.ravel()), axis=-1)
    # trie les états par E croissante
    state_sorted_by_E = state[np.argsort(state[:, 2])]
    # on garde les premiers états uniquement
    n1_list = state_sorted_by_E[:N, 0]
    n2_list = state_sorted_by_E[:N, 1]
    return n1_list, n2_list

def Fermi_Dirac_distribution(E_adim, mu_adim, T_adim):
    """ 
    Calcule la distribution de Fermi Dirac pour une énergie et une température donnée.
    
    Args : 
        - E_adim (float) : énergie adimensionnée
        - mu_adim (float) : potentiel chimique adimensionné  
        - T_adim (float) : température adimensionnée
    """
    return(1/(np.exp((E_adim-mu_adim)/T_adim)+1))
    
    
def occupations_0(n_max):
    return np.zeros((2*n_max+1, 2*n_max+1))