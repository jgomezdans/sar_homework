#!/usr/bin/env python
"""SAR retrieval strategy. Version 15. Lost my will to live"""

from pathlib import Path
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
import scipy.stats


def wcm_jac_(A, V1, B, V2, R, alpha, C, theta=23):
    """WCM model and jacobian calculations. The main
    assumption here is that we only consider first
    order effects. The vegetation backscatter contribution
    is given by `A*V1`, which is often related to scatterer
    (e.g. leaves, stems, ...) properties. The attenuation
    due to the canopy is controlled by `B*V2`, which is
    often related to canopy moisture content (this is polarisation
    and frequency dependent). The soil backscatter is modelled as
    an additive model (in dB units, multiplicative in linear), with
    a roughness term and a moisture-controlled term. The soil moisture
    term can be interpreted in weird and wonderful manners once retrieved
    (eg inverting the dielectric constant)
    This function returns the gradient for all parameters (A, B,
    V1, V2 and C)."""
    mu = np.cos(np.deg2rad(theta))
    tau = np.exp(-2 * B * V2 / mu)
    veg = A * V1 * (1 - tau)
    sigma_soil = R+alpha
    soil = tau * sigma_soil + C

    der_dA = V1 - V1 * tau
    der_dV1 = A - A * tau
    der_dB = (-2 * V2 / mu) * tau * (-A * V1 + sigma_soil)
    der_dV2 = (-2 * B / mu) * tau * (-A * V1 + sigma_soil)
    der_dC = 1
    der_dR = tau
    der_dalpha = tau

    # Also returns der_dV1 and der_dV2
    return (
        veg + soil,
        [der_dA, der_dB, der_dC, der_dR, der_dalpha, der_dV1, der_dV2],
    )

def fwd_model_(x, svh, svv, theta):
    """Running the model forward to predict backscatter"""
    n_obs = len(svv)
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh = x[:6]
    alpha = x[6 : (6 + n_obs)]
    R = x[(6 + n_obs):(6 + 2*n_obs)]
    lai = x[(6 + 2*n_obs) :]
    sigma_vv, dvv = wcm_jac_(A_vv, lai, B_vv, lai, C_vv, R, alpha, theta=theta)
    sigma_vh, dvh = wcm_jac_(A_vh, lai, B_vh, lai, C_vh, R, alpha, theta=theta)
    return sigma_vv, sigma_vh

def cost_obs_(x, svh, svv, theta, unc=0.5):
    """Cost function. Order of parameters is
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh,
    vsm_0, ..., vsm_N,
    LAI_0, ..., LAI_N

    We assume that len(svh) == N
    Uncertainty is the uncertainty in backscatter, and
    assume that there are two polarisations (VV and VH),
    although these are just labels!
    """
    n_obs = svh.shape[0]
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh = x[:6]
    alpha = x[6 : (6 + n_obs)]
    R = x[(6 + n_obs):(6 + 2*n_obs)]
    lai = x[(6 + 2*n_obs) :]
    sigma_vv, dvv = wcm_jac_(A_vv, lai, B_vv, lai, C_vv, R, alpha, theta=theta)
    sigma_vh, dvh = wcm_jac_(A_vh, lai, B_vh, lai, C_vh, R, alpha, theta=theta)
    diff_vv = svv - sigma_vv
    diff_vh = svh - sigma_vh
    #NOTE!!!!! Only fits the VV channel!!!!
    # Soil misture in VH is complicated
    diff_vh = 0.
    cost = 0.5 * (diff_vv ** 2 + diff_vh ** 2) / (unc ** 2)
    jac = np.concatenate(
        [##[der_dA, der_dB, der_dC, der_dR, der_dalpha, der_dV1, der_dV2]
            np.array(
                [
                    np.sum(dvv[0] * diff_vv),  # A_vv
                    np.sum(dvv[1] * diff_vv),  # B_vv
                    np.sum(dvv[2] * diff_vv),  # C_vv
                    np.sum(dvh[0] * diff_vh),  # A_vh
                    np.sum(dvh[1] * diff_vh),  # B_vh
                    np.sum(dvh[2] * diff_vh),
                ]
            ),  # C_vh
            dvv[3] * diff_vv + dvh[3] * diff_vh,  # R
            dvv[4] * diff_vv + dvh[4] * diff_vh,  # alpha
            (dvv[5] + dvv[6]) * diff_vv + (dvh[5] + dvh[6]) * diff_vh,  # LAI
        ]
    )
    return np.nansum(cost), -jac / (unc ** 2)


def cost_prior_(x, svh, svv, theta, prior_mean, prior_unc):
    """A Gaussian cost function prior. We assume no correlations
    between parameters, only mean and standard deviation.
    Cost function. Order of parameters is
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh,
    alpha_0, ..., alpha_N,
    ruff_0, ..., ruff_N,
    LAI_0, ..., LAI_N

    We assume that len(svh) == N
    """
    n_obs = len(svh)
    prior_cost = 0.5 * (prior_mean - x) ** 2 / prior_unc ** 2
    dprior_cost = -(prior_mean - x) / prior_unc ** 2
    dprior_cost[:6] = 0.0
    # Ruff->No prior!
    dprior_cost[(6 + n_obs):(6 + 2*n_obs)] = 0.
    cost0 = prior_cost[6:(6+n_obs)].sum() # alpha cost
    cost1 = prior_cost[(6+2*n_obs):].sum() # LAI cost
    return cost0 + cost1, dprior_cost


def cost_smooth_(x, gamma):
    """A smoother for one parameter (e.g. LAI or whatever).
    `gamma` controls the magnitude of the smoothing (higher
    `gamma`, more smoothing)
    """
    # Calculate differences
    p_diff1 = x[1:-1] - x[2:]
    p_diff2 = x[1:-1] - x[:-2]
    # Cost function
    xcost_model = 0.5 * gamma * np.sum(p_diff1 ** 2 + p_diff2 ** 2)
    # Jacobian
    xdcost_model = 1 * gamma * (p_diff1 + p_diff2)
    # Note that we miss the first and last elements of the Jacobian
    # They're zero!
    return xcost_model, xdcost_model


def cost_function_(x, svh, svv, theta, gamma, prior_mean, prior_unc, unc=0.8):
    """A combined cost function that calls the prior, fit to the observations
    """
    # Fit to the observations
    cost1, dcost1 = cost_obs_(x, svh, svv, theta, unc=unc)
    # Fit to the prior
    cost2, dcost2 = cost_prior_(x, svh, svv, theta, prior_mean, prior_unc)
    # Smooth evolution of LAI
    n_obs = len(svv)
    lai = x[(6 + 2*n_obs) :]
    cost3, dcost3 = cost_smooth_(lai, gamma[1])
    tmp = np.zeros_like(dcost1)
    tmp[(7 + 2*n_obs) : -1] = dcost3
    # Smooth evolution of ruffness
    R = x[(6 + n_obs):(6 + 2*n_obs)]
    cost4, dcost4 = cost_smooth_(R, gamma[0])
    tmp[(7 + n_obs) : (5 + 2*n_obs)] = dcost4
    return cost1 + cost2 + cost3 + cost4, dcost1 + dcost2 + tmp



def invert_field_(svv, svh, theta, prior_mean, prior_sd, gamma, s2_lai):
    n_obs = len(svv)
    # Do some dodgy starting point guessing
    sigma_soil_vv_mu = np.mean(svv[s2_lai < 1])
    sigma_soil_vh_mu = np.mean(svh[s2_lai < 1])
    xvv = np.array([1, 0.5, sigma_soil_vv_mu])
    xvh = np.array([1, 0.5, sigma_soil_vh_mu])
    sm0 = prior_mean[6 : (6 + n_obs)]
    ruff = sm0*0 + 1.
    # In reality, this should come from a sensible prior mean, but for the
    # time being...
    x0 = np.concatenate([xvv, xvh, sm0, ruff, s2_lai])

    # Put some parameter bounds so we don't end up with crazy numbers
    bounds = (
        [[None, None]] * 6
        + [[0.7, 1.3]] * s2_lai.shape[0]
        + [[None, None]]* s2_lai.shape[0]
        + [[0, 8]] * s2_lai.shape[0]
    )
    # Minimise the log-posterior
    retval = scipy.optimize.minimize(
        cost_function_,
        x0,
        bounds=bounds,
        jac=True,
        args=(svh, svv, theta, gamma, prior_mean, prior_sd),
        tol=1e-10,
        options={"disp": True},
    )
    print(
        f"Initial cost {cost_function_(x0,svh, svv, theta,gamma, prior_mean, prior_sd)[0]:g}"
    )
    print(f"Final cost {retval.fun:g}")
    fwd_vv, fwd_vh = fwd_model_(retval.x, svh, svv, theta)
    residuals = np.array([fwd_vv - svv,
                              fwd_vh - svh])
    return retval, residuals

def fresnel(eps, theta):
    """Fresnel reflection coefficient for VV"""
    theta = np.deg2rad(theta)
    num = (eps-1)*(np.sin(theta)**2 - eps*(1+np.sin(theta)**2))
    den = eps*np.cos(theta) + np.sqrt(eps - np.sin(theta)**2)
    den = den**2
    return np.abs(num/den)

def mv2eps(a, b, c, mv):
    eps = a + b * mv + c * mv**2
    return eps


def quad_approx_solver(a, b, c, theta, alphas):
    x = np.arange(0.01, 0.9, 0.01)
    p = np.polyfit(x, fresnel(mv2eps(a, b, c, x),theta.mean()), 2)
    # 2nd order polynomial
    #solve
    solutions = [np.roots([p[2]-aa, p[1], p[0]]) for aa in alphas]
    return solutions



def do_plots_(field, retval, svv, svh, theta, doy, df, s2_lai):
    n_obs = len(svv)
    fwd_vv, fwd_vh = fwd_model_(retval.x, svh, svv, theta)

    fig, axs = plt.subplots(
        nrows=4, ncols=1, sharex=True, squeeze=True, figsize=(15, 14)
    )
    axs = axs.flatten()
    l1 = axs[0].plot(doy, fwd_vv, "o", label="Solved sigma VV")
#    l2 = axs[0].plot(doy, fwd_vh, "o", label="Solved sigma VH")

    l3 = axs[0].plot(doy, svv, "o-", mfc="None", lw=0.2, label="Obs sigma VV")
#    l4 = axs[0].plot(doy, svh, "o-", mfc="None", lw=0.2, label="Obs sigma VH")
    legends = l1 + l3
    labels = [l.get_label() for l in legends]
    axs[0].legend(legends, labels, loc="best", frameon=False)
    axs[1].plot(
        doy,
        fwd_vv - svv,
        "o",
        mfc="None",
        label=f"Residual VV (RMSE: {np.std(fwd_vv-svv):g})",
    )
#    axs[1].plot(
#        doy,
#        fwd_vh - svh,
#        "o",
#        mfc="None",
#        label=f"Residual VH (RMSE: {np.std(fwd_vh-svh):g})",
#    )
    axs[1].axhspan(-0.5, 0.5, color="0.8")
    axs[1].legend(loc="best")
    l1 = axs[2].plot(doy, retval.x[6:(6+n_obs)], "r-o", label="sigma soil")
    axx = axs[2].twinx()
    l2 = axx.plot(
        df[f"doy_{field:s}"], df[f"SM_{field:s}"], "s-g", label="Sigma SM"
    )
    legends = l1 + l2
    labels = [l.get_label() for l in legends]
    axs[2].set_ylabel(r"$\sigma_{soil}\; [dB]$")
    axx.set_ylabel("Vol soil moisture [\%]")
    axx.legend(legends, labels, loc="best")
    l1 = axs[3].plot(doy, s2_lai, "-", label="KaSKA LAI")
    l2 = axs[3].plot(
        df[f"doy_{field:s}"], df[f"LAI_{field:s}"], "o", label="In situ"
    )
    l3 = axs[3].plot(doy, retval.x[(2*n_obs + 6) :], label="Analysis")
    axs[0].set_ylabel("Backscatter [dB]")
    axs[1].set_ylabel("Residual [dB]")
    axs[3].set_ylabel("Leaf area index $[m^2\,m^{-2}]$")
    legends = l1 + l2 + l3
    labels = [l.get_label() for l in legends]
    axs[3].legend(legends, labels, loc="best")
