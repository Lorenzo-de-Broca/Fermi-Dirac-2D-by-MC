# Fermi-Dirac-2D-by-MC

## Description du projet

Ce projet simule la **distribution de Fermi-Dirac en 2D** à l'aide de l'**algorithme de Monte Carlo (Metropolis)**. 

La distribution de Fermi-Dirac décrit la probabilité qu'un état énergétique soit occupé par un électron dans un système quantique dégénéré en équilibre thermique. Cette simulation modélise un gaz d'électrons bidimensionnel et observe comment les électrons se distribuent entre les niveaux d'énergie à différentes températures.

## Installation et lancement rapide

### Prérequis
```bash
pip install numpy scipy matplotlib pyyaml
```

### Lancer une simulation

Le code se lance avec `main.py` en spécifiant le type de simulation souhaité :

```bash
# Simulation simple (une seule température, un seul nombre de particules)
python main.py --file input.yaml --simu simple

# Simulation pour plusieurs températures
python main.py --file input.yaml --simu multiT

# Simulation pour plusieurs nombres de particules
python main.py --file input.yaml --simu multiN
```

## Configuration des paramètres

Les paramètres de simulation se définissent dans le fichier `input.yaml` :

### Pour une simulation simple (`--simu simple`)
```yaml
step: 100000          # Nombre d'itérations Monte Carlo
N: 100                # Nombre d'électrons (avec spin)
T: 100                # Température en Kelvin
L: 1                  # Facteur d'échelle pour la boîte de simulation
freq_save: 100        # Fréquence de sauvegarde des occupations
```

### Pour plusieurs températures (`--simu multiT`)
```yaml
step: 1000000         # Itérations par température
Tmin: 1               # Température minimale (K)
Tmax: 300             # Température maximale (K)
deltaT: 50            # Pas en température (K)
N: 100                # Nombre d'électrons
L: 1                  # Facteur d'échelle
freq_save: 100
```

### Pour plusieurs nombres de particules (`--simu multiN`)
```yaml
step: 1000000         # Itérations par simulation
T: 100                # Température fixe (K)
Nmin: 10              # Nombre min d'électrons
Nmax: 200             # Nombre max d'électrons
deltaN: 20            # Pas en nombre de particules
L: 1                  # Facteur d'échelle
freq_save: 100
```

## Structure du projet

### Fichiers principaux

| Fichier | Description |
|---------|-------------|
| **main.py** | Point d'entrée : exécute les simulations Monte Carlo (simple, multiT, multiN) |
| **MC.py** | Algorithme de Monte Carlo : génération de configurations, acceptation Metropolis |
| **CondInit.py** | Conditions initiales : conversion d'unités, distribution de Fermi-Dirac théorique |
| **parameters.py** | Constantes physiques (h, ℏ, k_B, m_e, etc.) |
| **fit.py** | Ajustement de courbes : fit des données par une distribution de Fermi-Dirac |
| **plots.py** | Visualisation : graphiques d'occupation, distributions d'énergie |
| **anim.py** | Animation : crée une vidéo MP4 montrant l'évolution temporelle |

### Fichiers de configuration

| Fichier | Description |
|---------|-------------|
| **input.yaml** | Paramètres pour une simulation simple |
| **input_anim.yaml** | Paramètres pour générer une animation à partir des données sauvegardées |

### Dossiers

| Dossier | Contenu |
|---------|---------|
| **fig/** | Figures PNG générées (subdivisions : `multiT/`, `multiN/`) |
| **animations_file/** | Données sauvegardées pour chaque simulation (format .npz) |
| **anim_vid/** | Vidéos MP4 de l'évolution temporelle |
| **__pycache__/** | Fichiers Python compilés (ignorés par Git) |

## Flux de travail typique

1. **Éditer `input.yaml`** avec les paramètres souhaités
2. **Lancer la simulation** :
   ```bash
   python main.py --file input.yaml --simu simple
   ```
3. **Les résultats s'affichent** automatiquement :
   - Graphiques interactifs matplotlib
   - Fichiers PNG dans `fig`
   - Données sauvegardées dans `animations_file`

4. **(Optionnel) Créer une animation** :
   ```bash
   python anim.py
   ```
   Cela génère une vidéo MP4 dans `anim_vid` montrant l'occupation des états quantiques en fonction du temps.

   Il faut remplir le fichier input_anim.yaml avec les données correspondant à une simulation déjà effectuée dont on souhaite réaliser l'animation

## Exemple d'exécution

```bash
# Simuler 100 électrons à 50K pendant 100 000 étapes
python main.py --file input.yaml --simu simple

# Comparer le comportement pour T = 1K, 100K, 300K
python main.py --file input.yaml --simu multiT
```

## Notes physiques

- Les énergies sont **adimensionnées** par rapport à $E_0 = \frac{\hbar^2}{2m_e}\left(\frac{2\pi}{L}\right)^2$
- La température est **adimensionnée** par $k_B T / E_0$
- Le potentiel chimique $\mu$ est estimé de deux façons : interpolation et ajustement numérique
- Les résultats sont comparés à la **distribution de Fermi-Dirac théorique**