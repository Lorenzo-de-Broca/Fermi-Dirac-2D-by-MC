import numpy as np

def CI_random(N, n_max):
    # TODO Modifier pour commencer avec un état T = 0K
    """ Retourne les listes n1, n2 de N états générés uniformément (pas à 0K) """
    #tirage aléatoire sans remise de N couples (n1,n2), avec 0 <= n1,n2 <= 2*n_max
    values = np.random.choice((2*n_max+1) * (2*n_max+1), size = N, replace = False) 
    n1_list, n2_list = np.unravel_index(values, (2*n_max+1, 2*n_max+1))
    n1_list -= n_max #on passe de [0,2n_max] à [-n_max,n_max]
    n2_list -= n_max #on passe de [0,2n_max] à [-n_max,n_max]
    return n1_list, n2_list


def occupations_0(n_max):
    return np.zeros((2*n_max+1, 2*n_max+1))


N=7
n_max=5

print(CI_random(N, n_max))
#print(occupations_0(n_max))