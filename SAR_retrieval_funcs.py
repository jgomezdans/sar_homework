#!/usr/bin/env python
"""SAR Water Cloud Model utility functions"""
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats


def oh2004(mv, ks, theta):
    """Oh 2004 soil model as a function of volumetric soil moisture,
    rms height and angle of incidence. 
    Parameters
    ----------
    mv : float, ndarray
        volumetric soil moisture m3/m3
    ks : float
        product of wavenumber and rms height
        be aware that both need to have the same units
    theta : float, ndarray
        incidence angle [rad]
    
    Returns
    -------
    Backscatter (linear units) in HH, VV and HV polarisations

    """
    p1 = (2.0 * theta / np.pi) ** (0.35 * mv ** (-0.65))
    p2 = np.exp(-0.4 * ks ** 1.4)
    p = 1 - p1 * p2
    # p = 1 - (2.*theta/np.pi)**(0.35*mv**(-0.65)) * np.exp(-0.4 * ks**1.4)
    dp_dmv = 0.2275 * p1 * np.log(2.0 * theta / np.pi) * p2 / (mv ** 1.65)
    dp_dks = 0.56 * p1 * np.exp(-0.4 * ks ** 1.4) * ks ** 0.4
    q0 = 0.095 * (0.13 + np.sin(1.5 * theta)) ** 1.4
    q = q0 * (1 - np.exp(-1.3 * ks ** 0.9))
    dq_dmv = 0.0
    dq_dks = q0 * (-1.17 * np.exp(-1.3 * ks ** 0.9) / ks ** 0.1)
    a = 0.11 * mv ** 0.7 * np.cos(theta) ** 2.2
    da_dmv = 0.7 * 0.11 * np.cos(theta) ** 2.2 / mv ** 0.3
    da_dks = 0.0
    b = 1 - np.exp(-0.32 * ks ** 1.8)
    db_mv = 0.0
    db_mks = 0.576 * np.exp(-0.32 * ks ** 1.8) * (ks ** 0.8)

    hv = a * b
    dhv_dmv = da_dmv * b
    dhv_dks = db_mks * a
    vv = a * b / q
    dvv_dmv = (da_dmv * b * q) / (q * q)
    dvv_dks = (db_mks * a * q + dq_dks * a * b) / (q * q)
    hh = hv / q * p

    return [
        [hh, None],
        [vv, np.array([dvv_dmv, dvv_dks])],
        [hv, np.array([dhv_dmv, dhv_dks])],
    ]


def wcm(A, V1, B, V2, mv, ks, theta=23, pol="VV"):
    """WCM model with Oh2004 model. The main
    assumption here is that we only consider first
    order effects. The vegetation backscatter contribution
    is given by `A*V1`, which is often related to scatterer
    (e.g. leaves, stems, ...) properties. The attenuation
    due to the canopy is controlled by `B*V2`, which is
    often related to canopy moisture content (this is polarisation
    and frequency dependent). The soil backscatter is modelled as
    using an Oh2004 model.
    """
    mu = np.cos(np.deg2rad(theta))
    tau = np.exp(-2 * B * V2 / mu)
    veg = A * V1 * (1 - tau)
    isel = ["HH", "VV", "HV"].index(pol.upper())
    (sigma_soil, (dsoil_dmv, dsoil_dks)) = oh2004(mv, ks, np.deg2rad(theta))[
        isel
    ]
    soil = tau * sigma_soil

    der_dA = V1 - V1 * tau
    der_dV1 = A - A * tau
    der_dB = (-2 * V2 / mu) * tau * (-A * V1 + sigma_soil)
    der_dV2 = (-2 * B / mu) * tau * (-A * V1 + sigma_soil)
    der_dmv = tau * dsoil_dmv
    der_dks = tau * dsoil_dks

    return (
        veg + soil,
        np.array([der_dA, der_dV1, der_dB, der_dV2, der_dmv, der_dks]),
    )


def cost_obs(x, svh, svv, theta, unc=0.5):
    """Cost function. Order of parameters is
    A_vv, B_vv, A_vh, B_vh, ks,
    vsm_0, ..., vsm_N,
    LAI_0, ..., LAI_N
    
    We assume that len(svh) == N
    Uncertainty is the uncertainty in backscatter, and
    assume that there are two polarisations (VV and VH),
    although these are just labels!
    """
    n_obs = svh.shape[0]
    A_vv, B_vv, A_vh, B_vh, ks = x[:5]
    vsm = x[5 : (5 + n_obs)]
    lai = x[(5 + n_obs) :]
    sigma_vv, dvv = wcm(A_vv, lai, B_vv, lai, vsm, ks, pol="VV", theta=theta)
    sigma_vh, dvh = wcm(A_vh, lai, B_vh, lai, vsm, ks, pol="HV", theta=theta)
    diff_vv = svv - sigma_vv
    diff_vh = svh - sigma_vh
    cost = 0.5 * (diff_vv ** 2 + diff_vh ** 2) / (unc ** 2)
    jac = np.concatenate(
        [
            np.array(
                [
                    np.sum(dvv[0] * diff_vv),  # A_vv
                    np.sum(dvv[2] * diff_vv),  # B_vv
                    np.sum(dvh[0] * diff_vh),  # A_vh
                    np.sum(dvh[2] * diff_vh),  # B_vh
                    np.sum(dvv[5] * diff_vv + dvh[5] * diff_vh),  # ks
                ]
            ),
            dvv[4] * diff_vv + dvh[4] * diff_vh,  # vsm
            (dvv[1] + dvv[3]) * diff_vv + (dvh[1] + dvh[3]) * diff_vh,  # LAI
        ]
    )
    return cost.sum(), -jac / (unc ** 2)


def cost_prior(x, svh, svv, theta, prior_mean, prior_unc):
    """A Gaussian cost function prior. We assume no correlations
    between parameters, only mean and standard deviation.
    Cost function. Order of parameters is
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh,
    vsm_0, ..., vsm_N,
    LAI_0, ..., LAI_N
    
    We assume that len(svh) == N
    """
    n_obs = len(svh)
    prior_cost = 0.5 * (prior_mean - x) ** 2 / prior_unc ** 2
    dprior_cost = -(prior_mean - x) / prior_unc ** 2
    dprior_cost[:5] = 0.0
    return (prior_cost[5:]).sum(), dprior_cost


def cost_smooth(x, gamma):
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


def cost_function(x, svh, svv, theta, gamma, prior_mean, prior_unc, unc=0.8):
    """A combined cost function that calls the prior, fit to the observations
    """
    # Fit to the observations
    cost1, dcost1 = cost_obs(x, svh, svv, theta, unc=unc)
    # Fit to the prior
    cost2, dcost2 = cost_prior(x, svh, svv, theta, prior_mean, prior_unc)
    # Smooth evolution of LAI
    n_obs = len(svv)
    lai = x[(5 + n_obs) :]
    cost3, dcost3 = cost_smooth(lai, gamma)
    tmp = np.zeros_like(dcost1)
    tmp[(6 + n_obs) : -1] = dcost3
    return cost1 + cost2 + cost3, dcost1 + dcost2 + tmp


def fwd_model(x, svh, svv, theta):
    """Running the model forward to predict backscatter"""
    n_obs = svh.shape[0]
    A_vv, B_vv, A_vh, B_vh, ks = x[:5]
    vsm = x[5 : (5 + n_obs)]
    lai = x[(5 + n_obs) :]
    sigma_vv, _ = wcm(A_vv, lai, B_vv, lai, vsm, ks, pol="VV", theta=theta)
    sigma_vh, _ = wcm(A_vh, lai, B_vh, lai, vsm, ks, pol="HV", theta=theta)
    return sigma_vv, sigma_vh


############ Some general functions for inversions #####
def prepare_field_data(field, df, df_s2):
    """Extracts and prepares data for a single field"""
    svv = 10 * np.log(df[f"sigma_sentinel_vv_{field:s}"])
    svh = 10 * np.log(df[f"sigma_sentinel_vh_{field:s}"])
    passer = np.isfinite(svv)
    #svv = df[f"sigma_sentinel_vv_{field:s}"][passer]
    #svh = df[f"sigma_sentinel_vh_{field:s}"][passer]

    svv = svv[passer]
    svh = svh[passer]
    n_obs = len(svv)
    theta = df[f"theta_{field:s}"][passer]
    s2_lai = np.interp(
        df[f"doy_{field:s}"][passer], df_s2.doy, df_s2[f"lai_{field:s}"]
    )
    s2_cab = np.interp(
        df[f"doy_{field:s}"][passer], df_s2.doy, df_s2[f"cab_{field:s}"]
    )
    s2_cbrown = np.interp(
        df[f"doy_{field:s}"][passer], df_s2.doy, df_s2[f"cbrown_{field:s}"]
    )
    doy = df[f"doy_{field:s}"][passer]
    return doy, passer, n_obs, svv, svh, theta, s2_lai, s2_cab, s2_cbrown


def do_plots(field, retval, svv, svh, theta, doy, df, s2_lai):
    n_obs = len(svv)
    fwd_vv, fwd_vh = fwd_model(retval.x, svh, svv, theta)

    fig, axs = plt.subplots(
        nrows=4, ncols=1, sharex=True, squeeze=True, figsize=(15, 14)
    )
    axs = axs.flatten()
    l1 = axs[0].plot(doy, fwd_vv, "o", label="Solved sigma VV")
    l2 = axs[0].plot(doy, fwd_vh, "o", label="Solved sigma VH")

    l3 = axs[0].plot(doy, svv, "o-", mfc="None", lw=0.2, label="Obs sigma VV")
    l4 = axs[0].plot(doy, svh, "o-", mfc="None", lw=0.2, label="Obs sigma VH")
    legends = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in legends]
    axs[0].legend(legends, labels, loc="best", frameon=False)
    axs[1].plot(
        doy,
        fwd_vv - svv,
        "o",
        mfc="None",
        label=f"Residual VV (RMSE: {np.std(fwd_vv-svv):g})",
    )
    axs[1].plot(
        doy,
        fwd_vh - svh,
        "o",
        mfc="None",
        label=f"Residual VH (RMSE: {np.std(fwd_vh-svh):g})",
    )
    axs[1].axhspan(-0.5, 0.5, color="0.8")
    axs[1].legend(loc="best")
    l1 = axs[2].plot(doy, retval.x[5 : (n_obs + 5)], "r-o", label="sigma soil")
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
    l3 = axs[3].plot(doy, retval.x[(n_obs + 5) :], label="Analysis")
    axs[0].set_ylabel("Backscatter [dB]")
    axs[1].set_ylabel("Residual [dB]")
    axs[3].set_ylabel("Leaf area index $[m^2\,m^{-2}]$")
    legends = l1 + l2 + l3
    labels = [l.get_label() for l in legends]
    axs[3].legend(legends, labels, loc="best")


def invert_field(svv, svh, theta, prior_mean, prior_sd, gamma, s2_lai, ks0=4):
    n_obs = len(svv)
    xvv = np.array([1, 0.5])
    xvh = np.array([1, 0.5, ks0])
    sm0 = prior_mean[5 : (5 + n_obs)]
    # In reality, this should come from a sensible prior mean, but for the
    # time being...
    x0 = np.concatenate([xvv, xvh, sm0, s2_lai])

    # Put some parameter bounds so we don't end up with crazy numbers
    bounds = (
        [[None, None]] * 4
        + [0.1, 20]
        + [[0, 0.5]] * s2_lai.shape[0]
        + [[0, 8]] * s2_lai.shape[0]
    )
    # Minimise the log-posterior
    retval = scipy.optimize.minimize(
        cost_function,
        x0,
        bounds=bounds,
        jac=True,
        args=(svh, svv, theta, gamma, prior_mean, prior_sd),
        tol=1e-10,
        options={"disp": True},
    )
    print(
        f"Initial cost {cost_function(x0,svh, svv, theta,gamma, prior_mean, prior_sd)[0]:g}"
    )
    print(f"Final cost {retval.fun:g}")
    return retval
