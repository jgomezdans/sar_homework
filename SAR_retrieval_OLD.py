#!/usr/bin/env python
"""SAR Water Cloud Model utility functions"""
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats
from osgeo import gdal, ogr, osr


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
    veg = A * V1 * (1 - tau)
    soil = tau * sigma_soil + C

    der_dA = V1 - V1 * tau
    der_dV1 = A - A * tau
    der_dB = (-2 * V2 / mu) * tau * (-A * V1 + sigma_soil)
    der_dV2 = (-2 * B / mu) * tau * (-A * V1 + sigma_soil)
    der_dC = 1  # CHECK!!!!!!
    der_dsigmasoil = tau

    # Also returns der_dV1 and der_dV2
    return (
        veg + soil,
        [der_dA, der_dB, der_dC, der_dsigmasoil, der_dV1, der_dV2],
    )


def cost_obs(x, svh, svv, theta, unc=0.5):
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


def cost_function(x, svh, svv, theta, gamma, prior_mean, prior_unc, unc=0.8):
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
    n_obs = len(svv)
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh = x[:6]
    vsm = x[6 : (6 + n_obs)]
    lai = x[(6 + n_obs) :]
    sigma_vv, dvv = wcm_jac(A_vv, lai, B_vv, lai, C_vv, vsm, theta=theta)
    sigma_vh, dvh = wcm_jac(A_vh, lai, B_vh, lai, C_vh, vsm, theta=theta)
    return sigma_vv, sigma_vh


def reproject_data(
    source_img,
    target_img=None,
    dstSRS=None,
    srcSRS=None,
    srcNodata=np.nan,
    dstNodata=np.nan,
    outputType=None,
    output_format="MEM",
    verbose=False,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xRes=None,
    yRes=None,
    xSize=None,
    ySize=None,
    resample=1,
):

    """
    A method that uses a source and a target images to
    reproject & clip the source image to match the extent,
    projection and resolution of the target image.

    """

    outputType = gdal.GDT_Unknown if outputType is None else outputType
    if srcNodata is None:
        try:
            srcNodata = " ".join(
                [
                    i.split("=")[1]
                    for i in gdal.Info(source_img).split("\n")
                    if " NoData" in i
                ]
            )
        except RuntimeError:
            srcNodata = None
    # If the output type is intenger and destination nodata is nan
    # set it to 0 to avoid warnings
    if outputType <= 5 and np.isnan(dstNodata):
        dstNodata = 0

    if srcSRS is not None:
        _srcSRS = osr.SpatialReference()
        try:
            _srcSRS.ImportFromEPSG(int(srcSRS.split(":")[1]))
        except:
            _srcSRS.ImportFromWkt(srcSRS)
    else:
        _srcSRS = None

    if (target_img is None) & (dstSRS is None):
        raise IOError(
            "Projection should be specified ether from "
            + "a file or a projection code."
        )
    elif target_img is not None:
        try:
            g = gdal.Open(target_img)
        except RuntimeError:
            g = target_img
        geo_t = g.GetGeoTransform()
        x_size, y_size = g.RasterXSize, g.RasterYSize

        if xRes is None:
            xRes = abs(geo_t[1])
        if yRes is None:
            yRes = abs(geo_t[5])

        if xSize is not None:
            x_size = 1.0 * xSize * xRes / abs(geo_t[1])
        if ySize is not None:
            y_size = 1.0 * ySize * yRes / abs(geo_t[5])

        xmin, xmax = (
            min(geo_t[0], geo_t[0] + x_size * geo_t[1]),
            max(geo_t[0], geo_t[0] + x_size * geo_t[1]),
        )
        ymin, ymax = (
            min(geo_t[3], geo_t[3] + y_size * geo_t[5]),
            max(geo_t[3], geo_t[3] + y_size * geo_t[5]),
        )
        dstSRS = osr.SpatialReference()
        raster_wkt = g.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)
        gg = gdal.Warp(
            "",
            source_img,
            format=output_format,
            outputBounds=[xmin, ymin, xmax, ymax],
            dstNodata=dstNodata,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            xRes=xRes,
            yRes=yRes,
            dstSRS=dstSRS,
            outputType=outputType,
            srcNodata=srcNodata,
            resampleAlg=resample,
            srcSRS=_srcSRS,
        )

    else:
        gg = gdal.Warp(
            "",
            source_img,
            format=output_format,
            outputBounds=[xmin, ymin, xmax, ymax],
            xRes=xRes,
            yRes=yRes,
            dstSRS=dstSRS,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            copyMetadata=True,
            outputType=outputType,
            dstNodata=dstNodata,
            srcNodata=srcNodata,
            resampleAlg=resample,
            srcSRS=_srcSRS,
        )
    if verbose:
        print(
            "There are %d bands in this file, use "
            + "g.GetRasterBand(<band>) to avoid reading the whole file."
            % gg.RasterCount
        )
    return gg


############ Some general functions for inversions #####
def prepare_field_data(field, df, df_s2):
    """Extracts and prepares data for a single field"""
    svv = 10 * np.log10(df[f"sigma_sentinel_vv_{field:s}"])
    svh = 10 * np.log10(df[f"sigma_sentinel_vh_{field:s}"])
    passer = np.isfinite(svv)
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
    # Do some dodgy starting point guessing
    sigma_soil_vv_mu = np.mean(svv[s2_lai < 1])
    sigma_soil_vh_mu = np.mean(svh[s2_lai < 1])
    xvv = np.array([1, 0.5, sigma_soil_vv_mu])
    xvh = np.array([1, 0.5, sigma_soil_vh_mu])
    sm0 = prior_mean[6 : (6 + n_obs)]
    # In reality, this should come from a sensible prior mean, but for the
    # time being...
    x0 = np.concatenate([xvv, xvh, sm0, s2_lai])

    # Put some parameter bounds so we don't end up with crazy numbers
    bounds = (
        [[None, None]] * 6
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
    fwd_vv, fwd_vh = fwd_model(retval.x, svh, svv, theta)
    residuals = np.array([fwd_vv - svv,
                              fwd_vh - svh])
    return retval, residuals
