#!/usr/bin/env python
"""SAR Water Cloud Model utility functions"""
import numpy as np

from pathlib import Path
import datetime as dt

import numpy as np

from osgeo import gdal,ogr
from osgeo import osr

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
    tau = np.exp(-2*B*V2/mu)
    veg = A*V1*(1-tau)
    soil = tau*sigma_soil + C

    der_dA = V1 - V1*tau
    der_dV1 = A - A*tau
    der_dB = (-2*V2/mu)*tau*(-A*V1 + sigma_soil)
    der_dV2 = (-2*B/mu)*tau*(-A*V1 + sigma_soil)
    der_dC = 1 # CHECK!!!!!!
    der_dsigmasoil = tau
    
    # Also returns der_dV1 and der_dV2
    return veg+soil, [der_dA, der_dB, der_dC,
                      der_dsigmasoil, der_dV1, der_dV2]


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
    vsm = x[6:(6+n_obs)]
    lai = x[(6+n_obs):]
    sigma_vv, dvv = wcm_jac(A_vv, lai, B_vv, lai,  C_vv, vsm, theta=theta)
    sigma_vh, dvh  = wcm_jac(A_vh, lai, B_vh, lai, C_vh, vsm, theta=theta)
    diff_vv = (svv - sigma_vv)
    diff_vh = (svh - sigma_vh)
    cost = 0.5*(diff_vv**2 + diff_vh**2)/(unc**2)
    jac = np.concatenate([np.array([np.sum(dvv[0]*diff_vv), #A_vv
                             np.sum(dvv[1]*diff_vv), # B_vv
                             np.sum(dvv[2]*diff_vv), # C_vv
                             np.sum(dvh[0]*diff_vh), # A_vh
                             np.sum(dvh[1]*diff_vh), # B_vh
                             np.sum(dvh[2]*diff_vh)]), #C_vh
                             dvv[3]*diff_vv + dvh[3]*diff_vh, # vsm
                             (dvv[4] + dvv[5])*diff_vv + (dvh[4] + dvh[5])*diff_vh # LAI
                             ])  
    return cost.sum(), -jac/(unc**2)


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
    prior_cost = 0.5*(prior_mean-x)**2/prior_unc**2
    dprior_cost = -(prior_mean-x)/prior_unc**2
    dprior_cost[:6] = 0.
    return (prior_cost[6:]).sum(),dprior_cost 


def cost_smooth(x, gamma):
    """A smoother for one parameter (e.g. LAI or whatever).
    `gamma` controls the magnitude of the smoothing (higher
    `gamma`, more smoothing)
    """
    # Calculate differences
    p_diff1 = x[1:-1] - x[2:]
    p_diff2 = x[1:-1] - x[:-2]
    # Cost function
    xcost_model = 0.5*gamma*np.sum(p_diff1**2 + p_diff2**2)
    # Jacobian
    xdcost_model = 1*gamma*(p_diff1 + p_diff2)
    # Note that we miss the first and last elements of the Jacobian
    # They're zero!
    return xcost_model, xdcost_model


def cost_function(x, svh, svv, theta, gamma, prior_mean, prior_unc, unc=.8):
    """A combined cost function that calls the prior, fit to the observations
    """
    # Fit to the observations
    cost1, dcost1 = cost_obs(x, svh, svv,  theta, unc=unc)
    # Fit to the prior
    cost2, dcost2 = cost_prior(x, svh, svv, theta, prior_mean, prior_unc)
    # Smooth evolution of LAI 
    n_obs = len(svv)
    lai = x[(6+n_obs):]
    cost3, dcost3 = cost_smooth(lai, gamma)
    tmp = np.zeros_like(dcost1)
    tmp[(7+n_obs):-1] = dcost3
    return cost1+cost2+cost3, dcost1+dcost2+tmp


def fwd_model(x, svh, svv, theta):
    """Running the model forward to predict backscatter"""
    n_obs = len(svv)
    A_vv, B_vv, C_vv, A_vh, B_vh, C_vh = x[:6]
    vsm = x[6:(6+n_obs)]
    lai = x[(6+n_obs):]
    sigma_vv, dvv = wcm_jac(A_vv, lai, B_vv, lai,  C_vv, vsm, theta=theta)
    sigma_vh, dvh  = wcm_jac(A_vh, lai, B_vh, lai, C_vh, vsm, theta=theta)
    return sigma_vv, sigma_vh



def reproject_data(source_img,
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

    outputType = (
        gdal.GDT_Unknown if outputType is None else outputType
        )
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
                srcSRS=_srcSRS
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
                srcSRS=_srcSRS
            )
    if verbose:
        LOG.debug("There are %d bands in this file, use "
                + "g.GetRasterBand(<band>) to avoid reading the whole file."
                % gg.RasterCount
            )
    return gg
