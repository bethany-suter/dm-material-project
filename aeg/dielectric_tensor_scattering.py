import h5py
import csv
from math import pi
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.signal import find_peaks, find_peaks_cwt, peak_widths
import scipy.integrate as integrate
import tensorflow as tf

GeV = 1e9               # eV
rho_chi = 0.3 * GeV     # GeV/cm^3
hbar = 6.582e-16        # eV s
g = 5.6175e23 * GeV
year = 3.1536e7          # s
me = 0.511e6            # eV
rho_srtio3 = 4.81 * g   # g/cm^3
rho_sil = 2.33 * g      # g/cm^3
alpha = 1./137
# e = np.sqrt(4 * np.pi * alpha)
e = np.sqrt(alpha)
c = 2.99792458e10            # cm/s
q_0 = alpha * me        # reference momentum eV
yeartoInveV = year / hbar # year to 1/eV
kgtoeV = 1000 * g         # kg to eV
eVtoInvcm = 1/hbar/c      # 1 eV to cm^-1

def Lindhard(q, w, Gam_p, kF, vF, wp):
    Qplu = tf.cast(q/(2*kF) + (w + 1j * Gam_p)/(q * vF), tf.complex128)
    Qmin = tf.cast(q/(2*kF) - (w + 1j * Gam_p)/(q * vF), tf.complex128)

    if Gam_p != 0:
        logQplu = tf.cast(tf.math.log((Qplu + 1)/(Qplu - 1)), tf.complex128)
        logQmin = tf.cast(tf.math.log((Qmin + 1)/(Qmin - 1)), tf.complex128)
        return tf.cast(1 + 3 * wp**2 / (q**2 * vF**2) * (1/2 + kF/(4*q)*(1 - Qmin**2)*logQmin + kF/(4*q)*(1 - Qplu**2)*logQplu), tf.complex128)

    logQplu = tf.cast(tf.math.log(tf.math.abs((Qplu + 1)/(Qplu - 1))), tf.complex128)
    logQmin = tf.cast(tf.math.log(tf.math.abs((Qmin + 1)/(Qmin - 1))), tf.complex128)

    re_eps = tf.cast(1 + 3 * wp**2 / (q**2 * vF**2) * (1/2 + kF/(4*q)*(1 - Qmin**2)*logQmin + kF/(4*q)*(1 - Qplu**2)*logQplu), tf.complex128)
    if tf.math.real(Qplu) < 1:
        im_eps = tf.cast(3 * pi * wp**2 * w / (2 * q**3 * vF**3), tf.complex128)
    elif tf.math.abs(Qmin) < 1 < tf.math.real(Qplu):
        im_eps = tf.cast(3 * pi * wp**2 * kF * (1 - Qmin**2) / (4 * q**3 * vF**2), tf.complex128)
    else:
        im_eps = tf.cast(0, tf.complex128)
    return tf.cast(re_eps + 1j * im_eps, tf.complex128)

def Lindhard_qto0(w, Gam_p, wp):
    return 1 + wp**2 / (Gam_p - 1j * w)**2

def loss_function(eps):
    return tf.math.imag(-1 / eps)

def W_lind(w, Gam_p, wp):
    return loss_function(Lindhard_qto0(w, Gam_p, wp))

def find_eps(formula):
    # formula: str; should be a chemical formula
    filename = "../dielectrics.h5"
    with h5py.File(filename, 'r') as fh:
        # Find the group with this formula
        formula_found = False
        for _, grp in fh.items():
            if grp.attrs['formula'] == formula:
                formula_found = True
                break
        if formula_found == False:
            print("Formula not found. Please make sure there is no typo and try again")
            return float("nan")
        energy = grp['energy'][:]
        er = pd.DataFrame(grp['epsilon_real'][:])
        ei = pd.DataFrame(grp['epsilon_imag'][:])
    eps = er + 1j*ei
    eps["energy"]=energy
    return eps

def find_density(formula):
    # finds the density for any chemical formula
    # based on material project data
    materials = pd.read_csv("../metadata.csv")
    for i in materials['index']:
        if formula == materials['formulae'][i]:
            return materials['density'][i]*5.6175E32 #g/cm^3 -> eV/cm^3
    print('material ' + formula + ' not found')
    return np.nan

def find_fermi_properties(formula):
    # finds fermi energy for any chemical formula
    # based on material project data
    # calculates other fermi properties: fermi momentum, fermi velocity, w_p
    materials = pd.read_csv("../metadata.csv")
    for i in materials['index']:
        if formula == materials['formulae'][i]:
            efermi = materials['efermi'][i] # eV
            if efermi < 0:
                efermi = np.nan
            kfermi = np.sqrt(2*efermi*me) # eV
            vfermi = np.sqrt(2*efermi/me)
            wp = vfermi / np.sqrt(3) * (e/np.pi * (2 * efermi * me**3)**(1/4))
            return (efermi, kfermi, vfermi, wp)
    print('material ' + formula + ' not found')
    return (np.nan, np.nan, np.nan, np.nan)

def find_BZ_size(formula):
    # finds the size of the Brillioun Zone for any
    # chemical formula based on material project data
    materials = pd.read_csv("metadata.csv")
    for i in materials['index']:
        if formula == materials['formulae'][i]:
            return 2*np.pi/(materials['volume'][i])**(1/3) / eVtoInvA # Volume is in A^3

    print('material ' + formula + ' not found')
    return np.nan

def find_band_gap(formula):
    # finds the band gap energy for any
    # chemical formula based on material project data
    materials = pd.read_csv("metadata.csv")
    for i in materials['index']:
        if formula == materials['formulae'][i]:
            return materials['bandgap'][i] # band gap is in eV

    print('material ' + formula + ' not found')
    return np.nan

def find_lindhard_params(material, formula, show_plots=False, use_mean_eps=True, eps_dir='xx', compare_dir=False):

    if use_mean_eps:
        eps = (material['xx']+material['yy']+material['zz'])/3
    else:
        eps = material[eps_dir]
    omega = np.array(material['energy'])

    W = loss_function(eps)

    peaks, properties = find_peaks(W, prominence=0.3, height=0.1, width=1)

    peak_width = np.array(peak_widths(W, peaks))
    peak_width = peak_width[0] * (omega[1]-omega[0])
    peak_heights = properties['peak_heights']

    if show_plots:
        total_W = np.zeros(len(W))
        for i in range(len(peaks)):
            total_W += W_lind(material["energy"], peak_width[i], omega[peaks][i])
        total_W /= len(peaks)

        plt.loglog(omega, W, '-')
        plt.plot(omega[peaks], W[peaks], 'x')
    #     plt.loglog(material["energy"],  W_lind(material["energy"], peak_width[0], omega[peaks][0]))
    #     plt.loglog(material["energy"],  W_lind(material["energy"], peak_width[1], omega[peaks][1]))
        plt.loglog(omega, total_W)
        plt.xlabel("$\omega (eV)$")
        plt.ylabel("$W(q=0,w)$")
        plt.title(formula)
        plt.show()

    parameters = np.array([peak_width, omega[peaks], peak_heights]).T
    return parameters


def W_total(q, w, kfermi, vfermi, parameters):
    q, w, kfermi, vfermi, parameters = [tf.cast(x, tf.complex128) for x in [q, w, kfermi, vfermi, parameters]]
    total_W = tf.cast(tf.zeros_like(q), tf.complex128)
    for i in range(len(parameters)):
        total_W += tf.cast(loss_function(Lindhard(q, w, parameters[i][0], kfermi, vfermi, parameters[i][1])), tf.complex128)
    total_W /= len(parameters)
    return total_W
