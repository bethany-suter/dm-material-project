"""
rate.py

This script calculates the daily modulation in dark matter experiments with 
anisotropic sensitivity, focusing on dielectric response and anisotropic 
effective electron mass tensor. It includes functions for interaction rates, 
energy loss functions, and various rate integrations under different scenarios.

Utilizing TensorFlow and VegasFlow, the script allows efficient computation 
for complex physics simulations. It features the Scaler class for scaling 
physics constants and parameters, functions for momentum transfer limits 
(q_min, q_max), loss functions considering bare 1-loop and RPA approaches, 
and rate calculation functions (rate, rate_extremes, pt_ratio, mean_rate, 
delta_r).

Usage:
Import as a module in Python environments with TensorFlow and VegasFlow 
installed. Customize dark matter interaction parameters to compute rates and 
analyze temporal modulations in detection experiments.

Note:
Ensure TensorFlow and VegasFlow dependencies are installed and configured 
in your Python environment.
"""


# Importing necessary libraries
import numpy as np  # For numerical operations
import tensorflow as tf  # For ML and computational graphs
from scipy import special  # For special mathematical functions
from vegasflow import VegasFlow, float_me  # For MC integration in physics

# Importing useful functions from previous code for scattering
import aeg.dielectric_tensor_scattering as dts

# Setting up logging for vegasflow
import logging
logger_vegasflow = logging.getLogger('vegasflow')
logger_vegasflow.setLevel(logging.ERROR)  # Log only errors

# Defining fundamental constants and derived quantities
me = 1  # Electron mass (in arbitrary units)
kF = 1  # Fermi wave number (in arbitrary units)
EF = 1  # Fermi energy (in arbitrary units)
hbar = np.sqrt(EF / (kF**2 / (2*me)))  # Reduced Planck constant
vF = hbar * kF / me  # Fermi velocity
e = np.sqrt(4*np.pi/137.)  # Elementary charge (natural units)
second_eV = 1.519e15  # Seconds to electronvolts conversion
kms_c = 3.34e-6  # km/s to speed of light units conversion
gram_eV = 5.61e32  # Grams to electronvolts conversion
me_eV = 511e3  # Electron mass in electronvolts
theta_earth = np.radians(42)  # Earth's tilt angle in radians

# Defining velocities relevant to dark matter experiment
V_0_KMS = 220  # Local standard of rest velocity in km/s
V_ESC_KMS = 550  # Escape velocity in km/s
V_EARTH_KMS = 232  # Earth's velocity in km/s


def mass_vector(r):
    """
    Calculate the mass vector for an electron in an anisotropic effective 
    mass tensor.

    Args:
        r (float): Parameter related to anisotropy direction/magnitude.

    Returns:
        numpy.ndarray: Anisotropic effective mass in three dimensions.
    """
    sr = r**(2./3.)  # Derived parameter for anisotropy
    # Return anisotropic effective mass components in 3D
    return np.asarray([1/np.sqrt(sr), 1/np.sqrt(sr), sr])


class Scaler(object):
    """
    Class for scaling parameters and constants in dark matter experiment 
    simulations.

    Attributes:
        EF_eV (float): Fermi energy in electronvolts.
        kF_eV (float): Fermi wave number in electronvolts.
        vF_c (float): Fermi velocity in speed of light units.
        c (float): Conversion factor for velocity.
        eV, kms, second, km, cm, gram (float): Conversion factors for energy,
            velocity, time, and mass.
        v_0, v_esc, s_earth (float): Scaled velocities for local standard of 
            rest, escape velocity, and Earth's velocity.
        N0 (float): Normalization constant related to velocity distribution.
        Delta (float): Parameter for anisotropic dielectric response.
    """

    def __init__(self, EF_eV=None, v0_over_vF=None):
        """
        Initialize Scaler object with optional Fermi energy and velocity ratio.

        Args:
            EF_eV (float, optional): Fermi energy in electronvolts. Defaults to
                calculated value based on electron mass, local standard of rest
                velocity, and velocity ratio.
            v0_over_vF (float, optional): Ratio of local standard of rest
                velocity to Fermi velocity. Required if EF_eV not provided.
        """
        if EF_eV:
            self.EF_eV = EF_eV  # Set Fermi energy if provided
        else:
            # Calculate Fermi energy from parameters
            self.EF_eV = (1/4)*me_eV*(V_0_KMS*kms_c)**2*hbar**2 / v0_over_vF**2
        self.kF_eV = np.sqrt(2*me_eV*self.EF_eV) / hbar  # Fermi wave number
        self.vF_c = hbar*self.kF_eV / me_eV  # Fermi velocity in c units
        self.c = 1/(self.vF_c/vF)  # Velocity conversion factor
        self.eV = EF/self.EF_eV  # Energy conversion factor
        self.kms = kms_c * self.c  # Velocity in km/s conversion factor
        self.second = second_eV*hbar/self.eV  # Time in seconds conversion
        self.km = self.kms * self.second  # Distance in kilometers conversion
        self.cm = 1e-5 * self.km  # Distance in centimeters conversion
        self.gram = gram_eV*self.eV  # Mass in grams conversion
        self.v_0 = V_0_KMS*self.kms  # Scaled local standard of rest velocity
        self.v_esc = V_ESC_KMS*self.kms  # Scaled escape velocity
        self.s_earth = V_EARTH_KMS*self.kms  # Scaled Earth's velocity
        # Calculate normalization constant N0
        self.N0 = np.pi**(3./2.) * self.v_0**3 * (
            special.erf(self.v_esc/self.v_0)
            - 2/np.sqrt(np.pi)*self.v_esc/self.v_0 * np.exp(
                -self.v_esc**2 / self.v_0**2
            )
        )
        # Calculate Delta for anisotropic dielectric response
        self.Delta = (
            2**(3./4.) * e * (me_eV/self.EF_eV)**(1./4.)
        ) / (np.sqrt(3.)*np.pi)


def f_0(scaler, v):
    """
    Calculate the normalized velocity distribution function.

    Args:
        scaler (Scaler object): An object containing scaling factors and constants.
        v (float): Velocity.

    Returns:
        float: Normalized velocity distribution value.
    """
    # Normalized Maxwell-Boltzmann distribution, considering escape velocity
    return 1./scaler.N0 * tf.math.exp(-v**2 / scaler.v_0**2) \
        * float_me(tf.greater(scaler.v_esc, v))


def v_earth(scaler, t, beta=0.):
    """
    Calculate Earth's velocity vector in the galactic frame.

    Args:
        scaler (Scaler object): An object containing scaling factors and constants.
        t (float): Time variable.
        beta (float, optional): Additional angle for rotation. Defaults to 0.

    Returns:
        tf.Tensor: Earth's velocity vector at time t.
    """
    t = float_me(t)
    t = tf.reshape(t, (-1,))
    psi = float_me(2*np.pi*t)
    beta = float_me(beta)
    # Rotation matrix for Earth's motion around the Sun
    bm = tf.convert_to_tensor([
        [tf.math.cos(beta), -tf.math.sin(beta), 0],
        [tf.math.sin(beta),  tf.math.cos(beta), 0],
        [0, 0, 1]
    ])
    # Earth's velocity in the galactic frame
    ve0 = scaler.s_earth * tf.convert_to_tensor([
        [tf.math.sin(psi)*tf.math.sin(theta_earth)],
        [(-1 + tf.math.cos(psi))*tf.math.cos(theta_earth)*tf.math.sin(theta_earth)],
        [tf.math.cos(theta_earth)**2 + tf.math.cos(psi)*tf.math.sin(theta_earth)**2]
    ])
    ve0 = tf.transpose(ve0, perm=(2, 0, 1))
    return bm @ ve0  # Applying rotation matrix to Earth's velocity


def v_minus(scaler, q, w, t, mX):
    """
    Calculate the minimum velocity for a given recoil energy.

    Args:
        scaler (Scaler object): An object containing scaling factors and constants.
        q (float): Recoil momentum.
        w (float): Recoil energy.
        t (float): Time variable.
        mX (float): Dark matter particle mass.

    Returns:
        tf.Tensor: Minimum velocity tensor.
    """
    q = float_me(q)
    qq = float_me(tf.expand_dims(tf.norm(q, axis=-1), axis=-1))
    w = float_me(w)
    r1 = tf.squeeze(tf.expand_dims(tf.squeeze(w)/tf.squeeze(qq), axis=-1))
    r2 = tf.squeeze(qq/(2*float_me(mX)))
    q2 = tf.squeeze(tf.expand_dims(q / qq, axis=1))
    ve = tf.squeeze(float_me(v_earth(scaler, t)))
    r3 = tf.reduce_sum( tf.multiply(q2, ve), 1)
    return tf.math.minimum(
        scaler.v_esc,
        r1 + r2 + r3
    )  # Calculating the minimum velocity


def g_0(scaler, q, w, t, mX):
    """
    Calculate the modulation amplitude for the velocity distribution.

    Args:
        scaler (Scaler object): An object containing scaling factors and constants.
        q (float): Recoil momentum.
        w (float): Recoil energy.
        t (float): Time variable.
        mX (float): Dark matter particle mass.

    Returns:
        tf.Tensor: Modulation amplitude tensor.
    """
    # Calculating the modulation amplitude
    return float_me(
        np.pi*scaler.v_0**2 / (tf.squeeze(tf.norm(q, axis=-1))*scaler.N0)
    ) * float_me(
        tf.math.exp(-v_minus(scaler, q, w, t, mX)**2 / scaler.v_0**2)
        - tf.exp(-scaler.v_esc**2 / scaler.v_0**2)
    )


def q_min(w_min, mX):
    """
    Calculate the minimum momentum transfer.

    Args:
        w_min (float): Minimum recoil energy.
        mX (float): Dark matter particle mass.

    Returns:
        tf.Tensor: Minimum momentum transfer.
    """
    return tf.sqrt(2*mX*w_min)  # Minimum momentum transfer formula


def q_max(scaler, w_min, mX):
    """
    Calculate the maximum momentum transfer.

    Args:
        scaler (Scaler object): An object containing scaling factors and constants.
        w_min (float): Minimum recoil energy.
        mX (float): Dark matter particle mass.

    Returns:
        tf.Tensor: Maximum momentum transfer.
    """
    # Maximum momentum transfer calculation
    max_val = mX * (
        (scaler.v_esc + scaler.s_earth)
        + tf.math.sqrt(-((2*w_min)/mX) + (scaler.v_esc + scaler.s_earth)**2)
    )
    min_val = float_me(q_min(w_min, mX))
    max_val = float_me(max_val)
    # Handling invalid or non-physical values
    bad_mask = tf.math.logical_or(
        tf.math.logical_not(tf.math.equal(tf.math.imag(max_val), 0)),
        tf.math.less(tf.math.real(max_val), 0)
    )
    max_val = tf.where(bad_mask, min_val, max_val)
    return tf.math.maximum(min_val, max_val)


def loss(v, m, d, D, mX, q):
    """
    Implement the energy loss function of the detector.

    Args:
        v (tf.Tensor): Velocity tensor.
        m (float): Mass parameter.
        d (float): Detector characteristic parameter.
        D (float): Another detector characteristic parameter.
        mX (float): Dark matter particle mass.
        q (tf.Tensor): Momentum tensor.

    Returns:
        tf.Tensor: Calculated loss tensor.
    """
    # Casting inputs to complex128 for the calculations
    v, m, d, D, mX, q = [tf.cast(x, tf.complex128) for x in [v, m, d, D, mX, q]]

    
    return loss


def V(q, m_med):
    """
    Calculate the potential function.

    Args:
        q (tf.Tensor): Momentum tensor.
        m_med (float): Mediator mass.

    Returns:
        tf.Tensor: Potential function value.
    """
    # Potential function calculation
    return 1/(tf.norm(q, axis=-1)**2 + m_med**2)


def make_rate_integrand(scaler, **kwargs):
    """
    Create the rate integrand function based on provided parameters, 
    differentiating between the bare 1-loop and RPA loss functions.

    The bare 1-loop loss function considers only the primary loop contribution 
    in the interaction, without accounting for higher-order loop corrections or 
    collective excitations. In contrast, the RPA loss function involves 
    resumming all 1-loop diagrams, which takes into account the many-body 
    effects and collective excitations within the system. This resummation 
    typically leads to a more accurate representation of the dielectric 
    response in many-body systems, especially important in dense mediums.

    Args:
        scaler (Scaler object): An object containing scaling factors and constants.
        **kwargs: Arbitrary keyword arguments for various parameters.

    Returns:
        function: Rate integrand function.
        function: Loss function, either bare 1-loop or RPA based on 'ph_only' flag.
    """
    # Extracting parameters from kwargs
    w_min = kwargs.get('w_min')
    w_max = kwargs.get('w_max')
    t = kwargs.get('t', 0.)
    mX = kwargs.get('mX', 1.)
    m_med = kwargs.get('m_med', 0.)
    d = kwargs.get('delta', 0.1)
    D = tf.cast(kwargs.get('Delta', scaler.Delta), tf.complex128)
    m = kwargs.get('m')
    v = kwargs.get('v')
    spherical = kwargs.get('spherical', True)
    lind_params_x = tf.cast(kwargs.get('lind_params_x', []), tf.complex128)
    lind_params_y = tf.cast(kwargs.get('lind_params_y', []), tf.complex128)
    lind_params_z = tf.cast(kwargs.get('lind_params_z', []), tf.complex128)
    kfermi = tf.cast(kwargs.get('kfermi', np.nan), tf.complex128)
    vfermi = tf.cast(kwargs.get('vfermi', np.nan), tf.complex128)
    
    def _loss_function(q, w):
        q = tf.cast(q, tf.complex128)
        w = tf.cast(w, tf.complex128)
        
        Wx = dts.W_total(q[:,0], w, kfermi, vfermi, lind_params_x)
        Wy = dts.W_total(q[:,0], w, kfermi, vfermi, lind_params_y)
        Wz = dts.W_total(q[:,0], w, kfermi, vfermi, lind_params_z)
        qhat_W_qhat = Wx + Wy + Wz
        return tf.cast(qhat_W_qhat, tf.float64)
    
    
    # Defining cartesian and spherical functions based on parameters
    if v is not None:
        def _cartesian(q):
            # Cartesian calculation
            q_dot_v = tf.reduce_sum(tf.multiply(q, v), 1)
            w = q_dot_v - tf.linalg.norm(q, axis=-1)**2 / (2*mX)
            return tf.squeeze(
                V(q, m_med)**2
                * tf.norm(q, axis=-1)**2
                * _loss_function(q, w)
            )
    elif t is None:
        def _cartesian(qwt):
            # Cartesian calculation for different case
            q = qwt[:, :3]
            w = qwt[:, 3]
            tt = qwt[:, 4]
            tt = tf.reshape(tt, (-1,))
            return tf.squeeze(
                tf.squeeze(g_0(scaler, q, w, tt, mX))
                * V(q, m_med)**2
                * tf.norm(q, axis=-1)**2
                * _loss_function(q, w)
            )
    else:
        def _cartesian(qw):
            # Another cartesian calculation case
            q = qw[:, :3]
            w = qw[:, 3]
            return (
                tf.squeeze(g_0(scaler, q, w, tf.reshape(t, (-1,)), mX))
                * V(q, m_med)**2
                * tf.norm(q, axis=-1)**2
                * _loss_function(q, w)
            )

    if spherical:
        def _spherical(qplus):
            # Spherical coordinates calculation
            # q = [rq, theta, phi]
            q = qplus[:, :3]
            qx = q[:, 0] * tf.math.sin(q[:, 1]) * tf.math.cos(q[:, 2])
            qy = q[:, 0] * tf.math.sin(q[:, 1]) * tf.math.sin(q[:, 2])
            qz = q[:, 0] * tf.math.cos(q[:, 1])
            qc = tf.concat(
                [qx[:, None], qy[:, None], qz[:, None], qplus[:, 3:]], axis=1
            )
            return _cartesian(qc) * (
                q[:, 0]**2 * tf.math.sin(q[:, 1])
            )
        _integrand = _spherical
    else:
        _integrand = _cartesian

    return _integrand, _loss_function


def rate(*args, n_iter=5, **kwargs):
    """
    Calculate the rate of dark matter interactions.

    This function dynamically adjusts the number of points for the Monte Carlo
    integration to achieve a specified error tolerance.

    Args:
        *args: Variable length argument list.
        n_iter (int, optional): Number of iterations for integration. Defaults to 5.
        **kwargs: Keyword arguments including 'mX', 'w_max', 'w_min', etc.

    Returns:
        tuple: Result of the integration and its error estimate.
    """
    # Extract parameters and set up the integration range
    s = args[0]
    mX = kwargs.get('mX', 1.)
    w_max = kwargs.get('w_max', 1)
    w_min = kwargs.get('w_min', 0)
    rq_max = kwargs.get('q_max', q_max(s, w_min, mX))
    rq_min = kwargs.get('q_min', q_min(w_min, mX))
    formula = kwargs.get('chem_formula', 'Si')

    # Adaptive point sampling for Monte Carlo integration
    n_points = kwargs.get('n_points')
    if n_points is None:
        err_tol = kwargs.pop('err_tol', 1e-2)
        n_points_init = kwargs.pop('n_points_init', int(1e6))
        n_points_step = kwargs.pop('n_points_step', 3)
        n_points_max = kwargs.pop('n_points_max',
                                  n_points_init * n_points_step**5)
        n_points = n_points_init
        above_tol = True
        while above_tol and n_points <= n_points_max:
            res, err = rate(*args, n_points=n_points, **kwargs)
            above_tol = (not np.isfinite(res)) or (np.abs(err / res) > err_tol)
            n_points *= n_points_step

    # Set up and run VegasFlow instance for integration
    
    integrand, _ = make_rate_integrand(*args, **kwargs)
    v = kwargs.get('v')
    spherical = kwargs.get('spherical', True)
    if spherical:
        xmin = [rq_min, 0, 0]
        xmax = [rq_max, np.pi, 2*np.pi]
    else:
        xmin, xmax = [-rq_max]*3, [rq_max]*3
    if v is not None:
        vegas_instance = VegasFlow(3, n_points, xmin=xmin, xmax=xmax)
    else:
        xmin.append(w_min)
        xmax.append(w_max)
        vegas_instance = VegasFlow(4, n_points, xmin=xmin, xmax=xmax)
    vegas_instance.compile(integrand)
    result, err = vegas_instance.run_integration(n_iter)
    return result, err


def rate_extremes(*args, **kwargs):
    """
    Calculate rate at two extreme points in time, t=0 and t=0.5.

    Args:
        *args: Variable length argument list.
        **kwargs: Keyword arguments.

    Returns:
        tuple: Rates and their errors at t=0 and t=0.5.
    """
    # Calculate rates at two different times for comparison
    result_1, err_1 = rate(*args, t=0.0, **kwargs)
    result_2, err_2 = rate(*args, t=0.5, **kwargs)
    return result_1, err_1, result_2, err_2


def pt_ratio(*args, **kwargs):
    """
    Calculate the ratio of rates at two points in time.

    This function computes the ratio of the highest to lowest rates at different
    times to assess the time variation in the rate.

    Args:
        *args: Variable length argument list.
        **kwargs: Keyword arguments.

    Returns:
        tuple: Ratio of rates and the associated error.
    """
    # Calculate rate extremes and then the ratio
    hi, hi_err, lo, lo_err = rate_extremes(*args, **kwargs)
    return hi/lo, np.sqrt(hi_err**2/lo**2 + hi**2 * lo_err**2/lo**4)


def mean_rate(*args, n_iter=5, n_points=int(1e6), **kwargs):
    """
    Calculate the mean rate over a period of time.

    This function performs a Monte Carlo integration to compute the average rate
    over time.

    Args:
        *args: Variable length argument list.
        n_iter (int, optional): Number of iterations for integration. Defaults to 5.
        n_points (int, optional): Number of points for integration. Defaults to 1e6.
        **kwargs: Keyword arguments.

    Returns:
        tuple: Mean rate and its error.
    """
    # Setup and computation of mean rate using VegasFlow
    s = args[0]
    mX = kwargs.get('mX', 1.)
    w_max = kwargs.get('w_max', 1)
    w_min = kwargs.get('w_min', 0)
    rq_max = kwargs.get('q_max', q_max(s, w_min, mX))
    rq_min = kwargs.get('q_min', q_min(w_min, mX))
    integrand, _ = make_rate_integrand(*args, t=None, **kwargs)
    spherical = kwargs.get('spherical', True)
    if spherical:
        xmin = [rq_min, 0, 0]
        xmax = [rq_max, np.pi, 2*np.pi]
    else:
        xmin, xmax = [-rq_max]*3, [rq_max]*3
    xmin += [w_min, 0]
    xmax += [w_max, 1]
    vegas_instance = VegasFlow(5, n_points, xmin=xmin, xmax=xmax)
    vegas_instance.compile(integrand)
    result, err = vegas_instance.run_integration(n_iter)
    return result, err


def delta_r(*args, **kwargs):
    """
    Calculate the relative difference in rates.

    This function computes the relative difference between the highest and lowest
    rates, normalized to the mean rate, to understand the extent of variability.

    Args:
        *args: Variable length argument list.
        **kwargs: Keyword arguments.

    Returns:
        tuple: Relative difference in rates and the associated error.
    """
    # Compute rates at extremes and mean rate
    hi, hi_err, lo, lo_err = rate_extremes(*args, **kwargs)
    mean, mean_err = mean_rate(*args, **kwargs)

    # Calculate relative difference and error
    err = np.sqrt(
        ((hi-lo)**2 * mean_err**2  +  mean**2 * (hi_err**2 + lo_err**2)) /
            mean**4
    )
    return (hi-lo)/mean, err
