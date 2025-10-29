import numpy as np
from parameters import h, hbar, k_b, m_e


def gen_cfg(n1_list, n2_list, occupation_step, n_max, N):
    """
    Génère une nouvelle configuration initiale aléatoire en partant de la configuration (n1,n2),
    où l'on a choisit que l'on tirait les nombres quantiques dans l'interval [-n_max , n-max]^2) 
    
    Args:
        n1_list (array): Tableau des premiers nombres quantiques principaux des particules
        n2_list (array): Tableau des seconds nombres quantiques principaux des particules
        occupation_step (array): Liste des occupations des états quantiques à cette étape
        n_max (int) :  Valeur maximale des nombres quantiques principaux
        N (int): Nombre de particules dans le système
    
    Outputs:
        n1_list_new (array): Nouvelle configuration des nombres quantiques principaux de la première particule
        n2_list_new (array): Nouvelle configuration des nombres quantiques principaux de la deuxième particule
    """
    
    # Choix de la particule à modifier 
    particle = np.random.randint(N)
    
    # Choix de la nouvelle configuration proposée parmi les états non occupés
    indices_non_occupes = np.argwhere(occupation_step == 0)
    if len(indices_non_occupes) > 0:
        prop_n1, prop_n2 = indices_non_occupes[np.random.choice(len(indices_non_occupes))]-n_max
    else:
        # Aucun état libre
        print("Warning: No unoccupied states available. The proposed configuration is None.")
        prop_n1, prop_n2 = None, None
       
    #print("The new configuration is not occupied yet. Let's try to accept it.")
    
    # Création des nouvelles configurations
    n1_new = np.copy(n1_list)
    n2_new = np.copy(n2_list)
    n1_new[particle] = prop_n1
    n2_new[particle] = prop_n2  
    
    return(n1_new, n2_new, particle)


def accepte_cfg(n1_list, n2_list, n1_new, n2_new, particle, T_adim):
    """
    Fonction qui accepte ou refuse une nouvelle configuration en fonction de l'énergie
    
    Args:
        n1_list (array): Tableau des premiers nombres quantiques principaux des particules
        n2_list (array): Tableau des seconds nombres quantiques principaux des particules
        n1_new (array): Nouvelle configuration des premiers nombres quantiques principaux des particules
        n2_new (array): Nouvelle configuration des seconds nombres quantiques principaux des particules
        particle (int): Indice de la particule modifiée
        E_0 (float): Unité d'énergie du problème
        T (float): Température du système en Kelvin
    """
    
    # Calcul de l'énergie de l'ancienne configuration
    n1_old_particle = n1_list[particle]
    n2_old_particle = n2_list[particle]
    E_old = n1_old_particle**2 + n2_old_particle**2
    
    # Calcul de l'énergie de la nouvelle configuration
    n1_new_particle = n1_new[particle]
    n2_new_particle = n2_new[particle]
    
    E_new = n1_new_particle**2 + n2_new_particle**2
    
    # Calcul de la proba d'acceptation
    delta_E = E_new - E_old
    
    if delta_E <= 0:
        #print("New configuration has a lower energy than before. The new configuraiton is accepted.")
        return True

    if delta_E > 0:
        prob = np.exp(- delta_E / T_adim)
        rand = np.random.rand()
        if rand < prob:
            #print("New configuration accepted based on Metropolis criterion.")
            return True
        else:
            #print("New configuration rejected based on Metropolis criterion.")
            return False
    
    
def modif_occupation_arr (occupation_arr, occupation_step, accepted, n1_list, n2_list, n1_new, n2_new, n_max, particle):
    """
    Modifie la liste des occupations en fonction de la nouvelle configuration acceptée
    
    Args:
        occupation_list (array): Liste des occupations des états quantiques
        n1_new (array): Nouvelle configuration des premiers nombres quantiques principaux des particules
        n2_new (array): Nouvelle configurationd seconds nombres quantiques principaux des particules
        n_max (int): Valeur maximale des nombres quantiques principaux
        
    Outputs: 
        occupation_arr (array): Liste des occupations des états quantiques mise à jour
    """
    if accepted:
        # Mise à jour de l'occupation à cette étape
        n1_old_particle = n1_list[particle]+n_max
        n2_old_particle = n2_list[particle]+n_max
        
        n1_new_particle = n1_new[particle]+n_max
        n2_new_particle = n2_new[particle] +n_max 
        
        occupation_step[n1_old_particle, n2_old_particle] -= 1
        occupation_step[n1_new_particle, n2_new_particle] += 1
            
    occupation_arr += occupation_step
    
    return occupation_arr