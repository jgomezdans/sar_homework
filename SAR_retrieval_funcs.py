#!/usr/bin/env python
"""SAR Water Cloud Model utility functions"""
import datetime as dt
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats


def wcm_jac(A, V1, B, V2, C, sigma_soil, theta=23):
    """WCM model and jacobian calculations. The main
    assumption here is that we only consider first
    order effects. The vegetation backscatter contribution
    is given by `A*V1`, which is often related to scatterer
    (e.g. leaves, stems, ...) properties. The attenuation
    due to the canopy is controlled by `B*V2`, which is
    often related to canopy moisture content (this is polarisation
    and frequency dependent). The soil backscatter is modelled as
    a linear function of volumetric soil moisture.

    This function returns the gradient for all parameters (A, B,
    V1, V2 and C)."""
    mu = np.cos(np.deg2rad(theta))
    tau = np.exp(-2 * B * V2 / mu)
    veg = A * V1 * mu * (1 - tau)
    soil = tau * sigma_soil + C

    der_dA = V1 * mu - V1 * mu * tau
    der_dV1 = A * mu - A * mu * tau
    der_dB = (-2 * V2 / mu) * tau * (-A * V1 + sigma_soil)
    der_dV2 = (-2 * B / mu) * tau * (-A * V1 + sigma_soil)
    der_dC = 1  # CHECK!!!!!!
    der_dsigmasoil = tau

    # Also returns der_dV1 and der_dV2
    return (
        veg + soil,
        [der_dA, der_dB, der_dC, der_dsigmasoil, der_dV1, der_dV2],
    )


def cost_obs_OLD(x, svh, svv, theta, unc=0.5):
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
    vsm = x[6 : (6 + n_obs)]
    lai = x[(6 + n_obs) :]
    sigma_vv, dvv = wcm_jac(A_vv, lai, B_vv, lai, C_vv, vsm, theta=theta)
    sigma_vh, dvh = wcm_jac(A_vh, lai, B_vh, lai, C_vh, vsm, theta=theta)
    diff_vv = svv - sigma_vv
    diff_vh = svh - sigma_vh
    cost = 0.5 * (diff_vv ** 2 + diff_vh ** 2) / (unc ** 2)
    jac = np.concatenate(
        [
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
            dvv[3] * diff_vv + dvh[3] * diff_vh,  # vsm
            (dvv[4] + dvv[5]) * diff_vv + (dvh[4] + dvh[5]) * diff_vh,  # LAI
        ]
    )
    return cost.sum(), -jac / (unc ** 2)


def wcm(A, V1, B, V2, mvs, R, theta=23, pol="VH"):
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
    theta = np.deg2rad(theta)
    mu = np.cos(theta)

    K = 10*np.log10(0.11*(mu**2.2))
    if pol.upper() == "VV":
        K = K - 10*np.log10(0.095*(0.13 + np.sin(1.5*theta))**1.4)

    tau = np.exp(-2 * B * V2 / mu)
    veg = A * V1 * mu * (1 - tau)
    sigma_soil = K + R + mvs


    der_dA = V1 * mu - V1 * mu * tau
    der_dV1 = A * mu - A * mu * tau
    der_dB = (-2 * V2 / mu) * tau * (-A * V1 + sigma_soil)
    der_dV2 = (-2 * B / mu) * tau * (-A * V1 + sigma_soil)
    der_dmvs = tau
    der_dR = tau

    return (
        veg + tau*sigma_soil,
        np.array([der_dA, der_dV1, der_dB, der_dV2, der_dmvs, der_dR]),
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
    A_vv, B_vv, R_vv, A_vh, B_vh, R_vh = x[:6]
    vsm = x[6 : (6 + n_obs)]
    lai = x[(6 + n_obs) :]
    sigma_vv, dvv = wcm(A_vv, lai, B_vv, lai, vsm, R_vv,
                        pol="VV", theta=theta)
    sigma_vh, dvh = wcm(A_vh, lai, B_vh, lai, vsm, R_vh,
                        pol="HV", theta=theta)
    diff_vv = svv - sigma_vv
    diff_vh = svh - sigma_vh
    cost = 0.5 * (diff_vv ** 2 + diff_vh ** 2) / (unc ** 2)
    jac = np.concatenate(
        [
            np.array(
                [
                    np.sum(dvv[0] * diff_vv),  # A_vv
                    np.sum(dvv[2] * diff_vv),  # B_vv
                    np.sum(dvv[5] * diff_vv),  # R_vv
                    np.sum(dvh[0] * diff_vh),  # A_vh
                    np.sum(dvh[2] * diff_vh),  # B_vh
                    np.sum(dvh[5] * diff_vh),  # R_vh
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
    A_vv, B_vv, R_vv, A_vh, B_vh, R_vh,
    vsm_0, ..., vsm_N,
    LAI_0, ..., LAI_N

    We assume that len(svh) == N
    """
    n_obs = len(svh)
    prior_cost = 0.5 * (prior_mean - x) ** 2 / prior_unc ** 2
    dprior_cost = -(prior_mean - x) / prior_unc ** 2
    dprior_cost[:6] = 0.0
    return (prior_cost[6:]).sum(), dprior_cost



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


def cost_function(x, svh, svv, theta, gamma, prior_mean, prior_unc,
                  unc=0.8):
    """A combined cost function that calls the prior, fit to the observations
    """
    # Fit to the observations
    cost1, dcost1 = cost_obs(x, svh, svv, theta, unc=unc)
    # Fit to the prior
    cost2, dcost2 = cost_prior(x, svh, svv, theta, prior_mean, prior_unc)
    # Smooth evolution of LAI
    n_obs = len(svv)
    lai = x[(6 + n_obs) :]
    cost3, dcost3 = cost_smooth(lai, gamma)
    tmp = np.zeros_like(dcost1)
    tmp[(7 + n_obs) : -1] = dcost3
    return cost1 + cost2 + cost3, dcost1 + dcost2 + tmp


def fwd_model(x, svh, svv, theta):
    """Running the model forward to predict backscatter"""
    n_obs = svh.shape[0]
    A_vv, B_vv, R_vv, A_vh, B_vh, R_vh = x[:6]
    vsm = x[6 : (6 + n_obs)]
    lai = x[(6 + n_obs) :]
    sigma_vv, _ = wcm(A_vv, lai, B_vv, lai, vsm, R_vv,
                        pol="VV", theta=theta)
    sigma_vh, _ = wcm(A_vh, lai, B_vh, lai, vsm, R_vh,
                        pol="HV", theta=theta)

    return sigma_vv, sigma_vh


def extract_data():
    chunk = """;301;301;301;301;301;301;301;301;301;301;301;301;508;508;508;508;508;508;508;508;508;508;508;508;542;542;542;542;542;542;542;542;542;542;542;542;319;319;319;319;319;319;319;319;319;319;319;319;515;515;515;515;515;515;515;515;515;515;515;515
;date;sigma_sentinel_vv;sigma_sentinel_vh;theta;relativeorbit;orbitdirection;satellite;LAI;SM;Height;VWC;vh/vv;date;sigma_sentinel_vv;sigma_sentinel_vh;theta;relativeorbit;orbitdirection;satellite;LAI;SM;Height;VWC;vh/vv;date;sigma_sentinel_vv;sigma_sentinel_vh;theta;relativeorbit;orbitdirection;satellite;LAI;SM;Height;VWC;vh/vv;date;sigma_sentinel_vv;sigma_sentinel_vh;theta;relativeorbit;orbitdirection;satellite;LAI;SM;Height;VWC;vh/vv;date;sigma_sentinel_vv;sigma_sentinel_vh;theta;relativeorbit;orbitdirection;satellite;LAI;SM;Height;VWC;vh/vv
"""
    fields = chunk.split("\n")[0].split(";")[1:]
    col_names = [f"{col:s}_{fields[i]:s}" for i,
             col in enumerate(chunk.split("\n")[1].split(";")[1:])]

    df = pd.read_csv("multi.csv", skiprows=2, sep=";", names=col_names)
    fields = ["301", "508", "542", "319", "515"]

    for field in fields:
        df[f"doy_{field:s}"] = pd.to_datetime(df[f'date_{field:s}']).dt.dayofyear

    df_s2 = pd.read_csv("LMU_S2_field_retrievals.csv", sep=";")
    df_s2['doy'] = pd.to_datetime(df_s2.dates).dt.dayofyear
    return df, df_s2, fields


############ Some general functions for inversions #####
def prepare_field_data(field, df, df_s2, ignore_orbits=True):
    svvx = 10 * np.log10(df[f"sigma_sentinel_vv_{field:s}"])
    svhx = 10 * np.log10(df[f"sigma_sentinel_vh_{field:s}"])
    thetax = df[f"theta_{field:s}"]
    passer1 = np.isfinite(svvx)

    if ignore_orbits:
        """Extracts and prepares data for a single field"""

        svv = svvx[passer1].values
        svh = svhx[passer1].values
        theta = thetax[passer1].values
        n_obs = len(svv)
        s2_lai = np.interp(
            df[f"doy_{field:s}"][passer1], df_s2.doy, df_s2[f"lai_{field:s}"]
        )
        s2_cab = np.interp(
            df[f"doy_{field:s}"][passer1], df_s2.doy, df_s2[f"cab_{field:s}"]
        )
        s2_cbrown = np.interp(
            df[f"doy_{field:s}"][passer1], df_s2.doy, df_s2[f"cbrown_{field:s}"]
        )
        doy = df[f"doy_{field:s}"][passer1].values
        return (doy, passer1, n_obs, svv, svh, theta,
                             s2_lai, s2_cab, s2_cbrown)

    else:
        orbits = np.unique(df[f"relativeorbit_{field:s}"].values)
        orbit_data = {}
        for orbit in orbits:
            passer = df[f"relativeorbit_{field:s}"] == orbit
            passer = passer*passer1
            """Extracts and prepares data for a single field"""

            svv = svvx[passer].values
            svh = svhx[passer].values
            theta = thetax[passer].values
            n_obs = len(svv)
            s2_lai = np.interp(
                df[f"doy_{field:s}"][passer], df_s2.doy, df_s2[f"lai_{field:s}"]
            )
            s2_cab = np.interp(
                df[f"doy_{field:s}"][passer], df_s2.doy, df_s2[f"cab_{field:s}"]
            )
            s2_cbrown = np.interp(
                df[f"doy_{field:s}"][passer], df_s2.doy, df_s2[f"cbrown_{field:s}"]
            )
            doy = df[f"doy_{field:s}"][passer].values
            orbit_data[orbit] = [doy, passer, n_obs, svv, svh, theta,
                                 s2_lai, s2_cab, s2_cbrown]
        return orbit_data


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
    l1 = axs[2].plot(doy, retval.x[6 : (n_obs + 6)], "r-o", label="sigma soil")
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
    l3 = axs[3].plot(doy, retval.x[(n_obs + 6) :], label="Analysis")
    axs[0].set_ylabel("Backscatter [dB]")
    axs[1].set_ylabel("Residual [dB]")
    axs[3].set_ylabel("Leaf area index $[m^2\,m^{-2}]$")
    legends = l1 + l2 + l3
    labels = [l.get_label() for l in legends]
    axs[3].legend(legends, labels, loc="best")


def invert_field(svv, svh, theta, prior_mean, prior_sd, gamma, s2_lai):
    n_obs = len(svv)
    xvv = np.array([1, 0.5, 0.])
    xvh = np.array([1, 0.5, 0.])
    sm0 = prior_mean[6 : (6 + n_obs)]
    # In reality, this should come from a sensible prior mean, but for the
    # time being...
    x0 = np.concatenate([xvv, xvh, sm0, s2_lai])

    # Put some parameter bounds so we don't end up with crazy numbers
    bounds = (
        [
        [None, None],
        [None, None],
        [-4, 0],
        [None, None],
        [None, None],
        [-4, 0]]
        + [[0.05, 0.4]] * s2_lai.shape[0]
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






def test_wcm_vv():
    """Test the WCM mode forward operation. Assume R_pq=0 and mvs=0"""
    A, B, V1, V2, mvs = -12,  0.05, np.ones(2)*4, np.ones(2)*4, np.zeros(2)
    R = np.zeros(2)
    retval_vv = wcm(A, V1, B, V2, mvs, R, theta=36, pol="VV")

    expected = np.array([-18.7851694, -18.7851694])
    assert np.allclose( retval_vv[0], expected)


def test_wcm_vh():
    """Test the WCM mode forward operation. Assume R_pq=0 and mvs=0"""
    A, B, V1, V2, mvs = -5,  0.1, np.ones(2)*4, np.ones(2)*4, np.zeros(2)
    R = np.zeros(2)
    retval_vh = wcm(A, V1, B, V2, mvs, R, theta=36, pol="VH")
    expected = np.array([-12.99188003, -12.99188003])
    assert np.allclose( retval_vh[0], expected)


def test_cost_wcm():
    sigma_vv = np.array([-18.7851694, -18.7851694])
    sigma_vh = np.array([-12.99188003, -12.99188003])
    theta = np.ones(2)*36.
    A_vv, B_vv, R_vv, A_vh, B_vh, R_vh = (-12, 0.05, 0., -5, 0.1, 0)


    x = np.concatenate([np.array([A_vv, B_vv, R_vv, A_vh, B_vh, R_vh]),
                       np.ones(2)*0.,np.ones(2)*4.])
    retval = cost_obs(x, sigma_vh, sigma_vv, theta, unc=0.5)
    assert np.allclose(retval[0], 0., atol=1e-3)


def test_cost_wcm_jac0():
    sigma_vv = np.array([-18.7851694, -18.7851694])
    sigma_vh = np.array([-12.99188003, -12.99188003])
    theta = np.ones(2)*36.
    A_vv, B_vv, R_vv, A_vh, B_vh, R_vh = (-12, 0.05, 0., -5, 0.1, 0)


    x = np.concatenate([np.array([A_vv, B_vv, R_vv, A_vh, B_vh, R_vh]),
                       np.ones(2)*0.,np.ones(2)*4.])
    retval = cost_obs(x, sigma_vh, sigma_vv, theta, unc=0.5)
    assert np.allclose(retval[1], np.zeros_like(retval[1]), atol=1e-3)



def test_cost_wcm_jac1():
    sigma_vv = np.array([-18.7851694, -18.7851694])
    sigma_vh = np.array([-12.99188003, -12.99188003])
    theta = np.ones(2)*36.
    A_vv, B_vv, R_vv, A_vh, B_vh, R_vh = (-10, 0.06, 0.1, -4, 0.11, 0.2)

    func = lambda xx: cost_obs(xx, sigma_vh, sigma_vv, theta, unc=0.5)[0]

    x = np.concatenate([np.array([A_vv, B_vv, R_vv, A_vh, B_vh, R_vh]),
                       np.ones(2)*0.,np.ones(2)*4.])
    approx_jac = scipy.optimize.approx_fprime(x, func, 1e-9)
    true_jac = cost_obs(x, sigma_vh, sigma_vv, theta, unc=0.5)[1]

    assert np.allclose(approx_jac, true_jac, atol=0.1)

