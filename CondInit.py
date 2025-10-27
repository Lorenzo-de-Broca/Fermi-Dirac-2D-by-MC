import numpy as np
from parameters import *
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
    
def L_box_std(N,T):
    """ Retourne l'unité de taille de boîte, telle que d = lambda (distance interatomique) """
    return np.sqrt(N) * lambda_th(T)

def Energy_unit(L):
    """ Retourne la valeur de l'unité d'énergie pour une boîte de taille L """
    return hbar**2/(2*m_e) * (2*np.pi/L)**2 # unité d'énergie

def wave_vector_unit(L):
    """ Retourne la valeur de l'unité de vecteur d'onde pour une boîte de taille L """
    return 2*np.pi/L # unité de vecteur d'onde

# -----------------------------------------------------------------------------------
"""/!\ grandeurs numériques ADIMENSIONNEES /!\ """

def kbT_adim(L,T):
    """ Retourne la valeur adimensionnée de k_b * T """
    E_0 = Energy_unit(L)
    return (k_b * T) / E_0

def wave_vector_Fermi(N):
    """ Retourne le vecteur d'onde de Fermi adimensionné """
    return np.sqrt(N/(2*np.pi))

def Energy_Fermi(N):
    """ Retourne l'énergie de Fermi adimensionnée """
    k_F = wave_vector_Fermi(N)
    return k_F**2

def create_n_max(E_F,L,T):
    """ Retourne la valeur maximale de n1 et n2 (tq E(n1,n2) = E_F) """
    """ E_max ~= E_F + k_b * T """ 
    """ CLP --> N = g_s * pi * n_F^2""" # spin pris en compte dans l'input atm : N := N/2
    # n_F = int(np.ceil(np.sqrt(E_F)))
    n_max = int(np.ceil(np.sqrt(E_F + kbT_adim(L,T)) * 1.5)) #facteur 1.5 = marge de sécurité
    return n_max
    
def CI(N, n_max):
    """ Retourne les listes n1,n2 des vecteurs d'onde ADIMENSIONNES des N électrons """
    #tirage aléatoire sans remise de N couples (n1,n2), avec 0 <= n1,n2 <= 2*n_max
    values = np.random.choice(2*n_max+1, 2**n_max+1, size = N, replace = False) 
    n1_list, n2_list = np.unravel_index(values, (2**n_max+1, 2**n_max+1))
    n1_list -= n_max #on passe de [0,2n_max] à [-n_max,n_max]
    n2_list -= n_max #on passe de [0,2n_max] à [-n_max,n_max]
    return n1_list, n2_list

def occupations_0(n_max):
    return np.zeros(2*n_max+1, 2*n_max+1)