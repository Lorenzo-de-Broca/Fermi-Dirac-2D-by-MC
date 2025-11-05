import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def fermi_dirac(E, beta, mu):
    """Fonction Fermi-Dirac 1/(exp(beta*(E-mu)) + 1)."""
    # éviter overflow numérique pour grandes valeurs
    x = beta * (E - mu)
    # stable: pour grandes valeurs positives exp(x) peut overflow, use np.where
    out = np.empty_like(x, dtype=float)
    pos = x > 50
    neg = x < -50
    mid = ~(pos | neg)
    out[pos] = 1.0 / (np.exp(x[pos]) + 1.0)         # close to 0
    out[neg] = 1.0                                 # close to 1
    out[mid] = 1.0 / (np.exp(x[mid]) + 1.0)
    return out

def fit_fermi_dirac(energies, occupancy, p0=None, bounds=None, sigma=None, plot_result=True):
    """
    Ajuste occupancy = fermi_dirac(energies; beta, mu).
    energies, occupancy : 1D arrays (ou masked) de même longueur.
    p0 : initial guess [beta, mu] (optionnel)
    bounds : bounds for curve_fit, e.g. ([0, E_min], [np.inf, E_max])
    sigma : optional uncertainties for weighting (same length as data)
    Returns (popt, pcov, mask) where popt = [beta, mu].
    """
    # Flatten and mask invalid values (nan or inf)
    E = np.asarray(energies).flatten()
    y = np.asarray(occupancy).flatten()

    valid = np.isfinite(E) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError("Aucune donnée valide pour le fit (tout NaN/inf).")

    Esel = E[valid]
    ysel = y[valid]
    sigsel = None if sigma is None else np.asarray(sigma).flatten()[valid]

    # initial guesses if not fournies
    if p0 is None:
        # beta ~ 1/(k_B T) if you know T; otherwise use 1.0
        beta0 = 1.0
        mu0 = np.median(Esel)
        p0 = [beta0, mu0]

    # default bounds: beta>0, mu inside data range
    if bounds is None:
        Emin, Emax = np.min(Esel), np.max(Esel)
        bounds = ([0.0, Emin-abs(Emax-Emin)], [np.inf, Emax+abs(Emax-Emin)])

    # call curve_fit
    try:
        popt, pcov = curve_fit(fermi_dirac, Esel, ysel, p0=p0, bounds=bounds, sigma=sigsel, maxfev=100000)
    except Exception as e:
        # fallback: try looser options or raise helpful error
        raise RuntimeError(f"curve_fit a échoué : {e}")

    if plot_result:
        # Plot data and fitted curve
        E_plot = np.linspace(np.min(Esel), np.max(Esel), 400)
        y_fit = fermi_dirac(E_plot, *popt)

        plt.figure(figsize=(6,4))
        plt.scatter(Esel, ysel, s=20, alpha=0.6, label='données (sélectionnées)')
        plt.plot(E_plot, y_fit, 'r-', lw=2, label=f'fit FD\nbeta={popt[0]:.4g}, mu={popt[1]:.4g}')
        plt.xlabel('Énergie')
        plt.ylabel('Occupation')
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    return popt, pcov, valid

