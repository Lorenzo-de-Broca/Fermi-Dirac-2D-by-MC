import numpy as np
import matplotlib.pyplot as plt

# constantes physiques
h = 6.67e-34
hbar = h/(2*np.pi)
k_b = 1.38e-23
m_e = 9.11e-31

# constantes du problème
E0 = hbar**2/(2*m_e) # énergie 
def lambda_th(T):
    return np.sqrt(h*hbar)/np.sqrt(m_e * k_b * T)
