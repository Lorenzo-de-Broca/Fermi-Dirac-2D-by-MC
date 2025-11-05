import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
import matplotlib.pyplot as plt

# tailles d'affichage
title = 20
label = 18
legend = 16
ticks = 14
plt.rcParams['xtick.labelsize'] = ticks
plt.rcParams['ytick.labelsize'] = ticks

def fermi_dirac_stable(E, beta, mu):
    """Fermi-Dirac numériquement stable via expit."""
    return expit(-beta * (E - mu))


def fit_fermi_dirac_mu(energies, occupancy, kb_T_adim, Ef=None,
                       p0=None, bounds=None, sigma=None, plot_result=True):
    """
    Ajuste occupancy(E) = fermi_dirac(E; beta=1/kb_T_adim, mu) en ne faisant varier
    que mu (beta fixé à 1/kb_T_adim).
    Retour: popt (mu,), pcov, valid_mask
    """
    E = np.asarray(energies).flatten()
    y = np.asarray(occupancy).flatten()

    beta_adim = 1.0 / kb_T_adim

    # mask des valeurs valides
    valid = np.isfinite(E) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError("Aucune donnée valide pour le fit (tout NaN/inf).")

    Esel = E[valid]
    ysel = y[valid]
    sigsel = None if sigma is None else np.asarray(sigma).flatten()[valid]

    # trier par énergie (utile pour interpolation)
    order = np.argsort(Esel)
    Esorted = Esel[order]
    ysorted = ysel[order]

    # Estimer mu0 comme l'énergie où y croise 0.5 (interpolation linéaire)
    try:
        # si ysorted est monotone (typique pour FD), on interpole
        mu0_est = np.interp(0.5, ysorted[::-1], Esorted[::-1])  # on inverse si décroissant
    except Exception:
        mu0_est = np.median(Esorted)

    if p0 is None:
        p0 = [mu0_est]

    # bornes pour mu : par défaut dans la plage d'energie (avec marge)
    Emin, Emax = np.min(Esorted), np.max(Esorted)
    if bounds is None:
        margin = 0.1 * max(1.0, (Emax - Emin))  # petite marge relative
        lower = Emin - margin
        upper = Emax + margin
        bounds = ([lower], [upper])
    else:
        # s'assurer que bounds a la forme ([low], [high]) pour 1 paramètre
        # si utilisateur a passé deux scalaires, on les convertit
        if (np.isscalar(bounds[0]) and np.isscalar(bounds[1])):
            bounds = ([bounds[0]], [bounds[1]])

    # fonction à un paramètre mu (beta fixé)
    def fermi_mu(Earr, mu):
        return fermi_dirac_stable(Earr, beta_adim, mu)

    # appel à curve_fit
    try:
        # si sigma est fournie et représente des écarts-types réels, on peut vouloir absolute_sigma=True
        popt, pcov = curve_fit(fermi_mu, Esorted, ysorted, p0=p0, bounds=bounds,
                               sigma=sigsel, maxfev=200000)
    except Exception as e:
        raise RuntimeError(f"curve_fit a échoué : {e}")

    mu_fit = popt[0]

    if plot_result:
        E_plot = np.linspace(Emin, Emax, 800)
        y_fit = fermi_mu(E_plot, mu_fit)

        plt.figure(figsize=(7,4))
        plt.plot(E, y, "+b", label='données (brutes)')
        plt.plot(E_plot, y_fit, "r-", lw=2, label=f"fit FD (β={beta_adim:.3g}, μ={mu_fit:.4g})")
        if Ef is not None:
            plt.axvline(Ef, linestyle='--', color='k', label=f"E_F = {Ef:.3g}")
        plt.axvline(mu_fit, linestyle='--', color='g', label=f"μ_fit = {mu_fit:.3g}")
        plt.title("Fit de la distribution Fermi-Dirac (μ fixé)", fontsize=title)
        plt.xlabel('Énergie', fontsize=label)
        plt.ylabel('Occupation', fontsize=label)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    return popt, pcov, valid
