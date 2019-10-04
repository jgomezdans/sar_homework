#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import gp_emulator

class EpsModel(object):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        clay : float
            clay content as fractional volume
        sand : float
            sand content as fractional volume
        bulk : float
            bulk density [g/cm**3]; default: 1.65
        mv : float
            volumetric soil moisture content [cm**3/cm**3] = [g/cm**3]
        freq : float
            frequency [GHz]
        t : float
            temperature [°C]
        """

        self.clay = kwargs.get('clay', None)
        self.sand = kwargs.get('sand', None)

        self.bulk = kwargs.get('bulk', 1.65)

        self.mv = kwargs.get('mv', None)

        self.f = kwargs.get('freq', None)

        self.t = kwargs.get('temp', 23.)

        self._check()

    def _check(self):
        assert self.clay is not None, 'Clay needs to be provided!'
        assert self.clay >=0.
        assert self.clay <=1.
        assert self.sand is not None, 'Sand needs to be provided!'
        assert self.sand >=0.
        assert self.sand <=1.
        assert self.mv is not None, 'volumetric soil moisture needs to be given'
        assert self.f is not None, 'Frequency needs to be given!'
        if isinstance(self.f, np.ndarray):
            assert np.all(self.f> 0.)
        else:
            assert self.f > 0.

class Dobson85(EpsModel):
    def __init__(self, **kwargs):
        super(Dobson85, self).__init__(**kwargs)

        self.debye = kwargs.get('debye', False)
        self.single_debye = kwargs.get('single_debye', False)
        self._init_model_parameters()
        self.ew = self._calc_ew()
        self.eps = self._calc_eps()

    def _calc_ew(self):
        """
        calculate dielectric permittivity of free water
        using either the Debye model or a more simplistic approach
        """
        if self.debye:
            # single Debye dielectric model for pure water. Eqs. 4.14 or Debye model with conductivity term for e2. Eqs. 4.67
            return self._debye()
        else:
            # default setting
            # simplistic approach using Eq. 4.69
            return self._simple_ew()

    def _simple_ew(self):
        """
        eq. 4.69
        simplistic approach with T=23°C, bulk density = 1.7 g/cm3
        """
        f0 = 18.64   # relaxation frequency [GHz]
        hlp = self.f/f0
        e1 = 4.9 + (74.1)/(1.+hlp**2.)
        e2 =(74.1*hlp)/(1.+hlp**2.) + 6.46 * self.sigma/self.f
        return e1 + 1.j * e2

    def _debye(self):
        """
        Debye model
        1) single Debye dielectric model for pure water. Eqs. 4.14
        2) (default) Debye model with conductivity term for e2. Eqs. 4.67
        """


        f = self.f *10**9
        ew_inf = 4.9 # determined by Lane and Saxton 1952 (E.4.15)
        ew_0 = 88.045 - 0.4147 * self.t + 6.295*10**-4 * self.t**2 + 1.075*10**-5 * self.t**3
        tau_w = (1.1109*10**-10 - 3.824*10**-12*self.t + 6.938*10**-14*self.t**2 - 5.096*10**-16*self.t**3)/2./np.pi
        e1 = ew_inf +(ew_0-ew_inf)/(1 + (2*np.pi*f*tau_w)**2)

        if self.single_debye:
            # single Debye dielectric model for pure water. Eqs. 4.14
            e2 = 2*np.pi*f*tau_w * (ew_0-ew_inf) / (1 + (2*np.pi*f*tau_w)**2)
        else:
            # Debye model with conductivity term for e2. Eqs. 4.67
            e2 = 2*np.pi*f*tau_w * (ew_0-ew_inf) / (1 + (2*np.pi*f*tau_w)**2) + (2.65-self.bulk)/2.65/self.mv * self.sigma/(2*np.pi*8.854*10**-12*f)
        return e1 + 1.j *e2

    def _init_model_parameters(self):
        """
        model parameters, eq. 4.68, Ulaby (2014)
        """
        self.alpha = 0.65
        self.beta1 = 1.27-0.519*self.sand - 0.152*self.clay
        self.beta2 = 2.06 - 0.928*self.sand -0.255*self.clay
        self.sigma = -1.645 + 1.939*self.bulk - 2.256*self.sand + 1.594*self.clay

    def _calc_eps(self):
        """
        calculate dielectric permittivity
        Eq. 4.66 (Ulaby et al., 2014)
        """

        e1 = (1.+0.66*self.bulk+self.mv**self.beta1*np.real(self.ew)**self.alpha - self.mv)**(1./self.alpha)
        e2 = np.imag(self.ew)*self.mv**self.beta2
        return e1 + 1.j*e2


class Fresnel0(object):
    def __init__(self, e):
        """
        calculate the Nadir Fresnel reflectivity
        e.g. Ulaby (2014), eq. 10.36
        Parameters
        ----------
        e : complex
            complex relative dielectric permitivity
        """
        self.x = self._calc(e)

    def _calc(self, e):
        return np.abs( (1.-np.sqrt(e))/(1.+np.sqrt(e))   )**2.



class Reflectivity(object):
    """
    calculate the reflectivity for H and V polarization
    """
    def __init__(self, eps, theta):
        """
        table 2.5 Ulaby (2014)
        assumes specular surface
        Parameters
        ----------
        eps : complex
            relative dielectric permitivity
        theta : float, ndarray
            incidence angle [rad]
            can be specified
        """
        self.eps = eps
        self.theta = theta

        self._calc_reflection_coefficients()

        self.v = np.abs(self.rho_v)**2.
        self.h = np.abs(self.rho_h)**2.


    def _calc_reflection_coefficients(self):
        """
        calculate reflection coefficients
        Woodhouse, 2006; Eq. 5.54, 5.55
        """
        # OLD
        co = np.cos(self.theta)
        si2 = np.sin(self.theta)**2.
        self.rho_v = (self.eps*co-np.sqrt(self.eps-si2))/(self.eps*co+np.sqrt(self.eps-si2))
        self.rho_h = (co-np.sqrt(self.eps-si2))/(co+np.sqrt(self.eps-si2))

        srv = self.rho_v
        srh = self.rho_h

        # # FROM PRISM1_FORWARDMODEL-1.m
        # n1 = np.sqrt(1.)
        # n2 = np.sqrt(self.eps)
        # costh2 = np.sqrt(1-(n1*np.sin(self.theta)/2.)**2)

        # self.rho_v = -(n2*np.cos(self.theta) - n1*costh2)/(n2*np.cos(self.theta) + n1*costh2)
        # self.rho_h = (n1*np.cos(self.theta) - n2*costh2)/(n1*np.cos(self.theta) + n2*costh2)

        # plt.plot(np.rad2deg(self.theta), self.rho_v-srv, label = 'v_diff')
        # plt.plot(np.rad2deg(self.theta), self.rho_h-srh, label = 'h_diff')
        # plt.legend()
        # # doesn't make much difference in results!

    def plot(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(np.rad2deg(self.theta), self.v, color='red', linestyle='-', label='V')
        ax.plot(np.rad2deg(self.theta), self.h, color='blue', linestyle='--', label='H')
        ax.grid()
        ax.legend()







"""
Major surface scatter class
"""
class SurfaceScatter(object):
    def __init__(self, eps=None, ks=None, theta=None, kl=None, mv=None, C_hh=None, C_vv=None, C_hv=None, D_hh=None, D_vv=None, D_hv=None, **kwargs):
        self.eps = eps
        self.ks = ks
        self.theta = theta
        self.kl = kl

        self.mv = mv
        self.C_hh = C_hh
        self.C_vv = C_vv
        self.D_hh = D_hh
        self.D_vv = D_vv
        self.C_hv = C_hv
        self.D_hv = D_hv

        self._check()

    def _check(self):
        pass
        # assert isinstance(self.eps, complex)
        
        

class Oh92(SurfaceScatter):
    def __init__(self, eps, ks, theta):
        """
        Parameters
        ----------
        eps : complex
            relative dielectric permitivity
        ks : float
            product of wavenumber and rms height
            be aware that both need to have the same units
        theta : float, ndarray
            incidence angle [rad]
        """
        super(Oh92, self).__init__(eps, ks, theta)

        # calculate p and q
        self.G0 = Fresnel0(self.eps)  # nadir fresnel reflectivity
        self.G = Reflectivity(self.eps, self.theta)
        self._calc_p()
        self._calc_q()

        # calculate backscatter
        # (added 10*np.log10() like in PRISM1_FORWARDMODEL-1.m)
        self._vv0 = self._calc_vv()
        self.vv = self._vv0
        self.hh = self.p * self._vv0
        self.hv = self.q * self._vv0

    def _calc_p(self):
        a = 1./(3.*self.G0.x)
        self.p = (1. - (2.*self.theta/np.pi)**a * np.exp(-self.ks))**2.

    def _calc_q(self):
        self.q = 0.23*(self.G0.x)**0.5 * (1.-np.exp(-self.ks))

    def _calc_vv(self):

        a = 0.7*(1.-np.exp(-0.65*self.ks**1.8))
        b = np.cos(self.theta)**3. * (self.G.v+self.G.h) / np.sqrt(self.p)
        return a*b

    def plot(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        t = np.rad2deg(self.theta)
        ax.plot(t, 10*np.log10(self.hh), color='blue', label='hh')
        ax.plot(t, 10*np.log10(self.vv), color='red', label='vv')
        ax.plot(t, 10*np.log10(self.hv), color='green', label='hv')
        ax.grid()
        ax.set_ylim(-25.,0.)
        ax.set_xlim(0.,70.)
        ax.legend()
        ax.set_xlabel('incidence angle [deg]')
        # ax.set_ylabel('backscatter [dB]')
        ax.set_ylabel('backscatters coefficient [dB m2/m2]')



class Oh04(SurfaceScatter):
    def __init__(self, mv, ks, theta):
        """
        Parameters
        ----------
        mv : float, ndarray
            volumetric soil moisture m3/m3
        ks : float
            product of wavenumber and rms height
            be aware that both need to have the same units
        theta : float, ndarray
            incidence angle [rad]
        """
        super(Oh04, self).__init__(mv=mv, ks=ks, theta=theta)

        # calculate p and q
        self._calc_p()
        self._calc_q()

        # calculate backascatter
        self.hv = self._calc_vh()
        # difference between hv and vh?
        self.vv = self.hv / self.q
        self.hh = self.hv / self.q * self.p

    def _calc_p(self):
        self.p = 1 - (2.*self.theta/np.pi)**(0.35*self.mv**(-0.65)) * np.exp(-0.4 * self.ks**1.4)

    def _calc_q(self):
        self.q = 0.095 * (0.13 + np.sin(1.5*self.theta))**1.4 * (1-np.exp(-1.3 * self.ks**0.9))

    def _calc_vh(self):
        a = 0.11 * self.mv**0.7 * np.cos(self.theta)**2.2
        b = 1 - np.exp(-0.32 * self.ks**1.8)
        return a*b

    def plot(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        t = np.rad2deg(self.theta)
        ax.plot(t, 10.*np.log10(self.hh), color='blue', label='hh')
        ax.plot(t, 10.*np.log10(self.vv), color='red', label='vv')
        ax.plot(t, 10.*np.log10(self.hv), color='green', label='hv')
        ax.grid()
        #ax.set_ylim(-25.,0.)
        #ax.set_xlim(0.,70.)
        ax.legend()
        ax.set_xlabel('incidence angle [deg]')
        ax.set_ylabel('backscatter [dB]')



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
    p = 1 - (2.*theta/np.pi)**(0.35*mv**(-0.65)) * np.exp(-0.4 * ks**1.4)
    q = 0.095 * (0.13 + np.sin(1.5*theta))**1.4 * (1-np.exp(-1.3 * ks**0.9))
    a = 0.11 * mv**0.7 * np.cos(theta)**2.2
    b = 1 - np.exp(-0.32 * ks**1.8)
    hv = a*b
    vv = a*b/q
    hh = hv/q*p
    return hh, vv, hv
    


def soil_backscatter(mv, ks, theta):
    oh = Oh04(mv, ks, theta)
    return(oh.hh, oh.vv, oh.hv)

def wcm(A, V1, B, V2, C, mv, ks, theta=23, pol="VV"):
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
    tau = np.exp(-2*B*V2/mu)
    veg = A*V1*(1-tau)
    sigma_soil = soil_backscatter(mv, ks, np.deg2rad(theta))
    isel = ["HH", "VV", "HV"].index(pol.upper())
    soil = tau*sigma_soil[isel] + C
    return veg + soil


def WCM_Oh2004_emulator(n_train=50, n_validate=400):

    parameters = ["Avv", "V1", "Bvv", "V2", "Cvv", "mv", "ks", "theta"]
    min_vals = [-10, 0, 0.05, 0, -40, 0.0, 0, 15]
    max_vals = [10,  7, 0.5,  7, -15, 0.6, 5, 40]


    def simulator(x):
        return wcm(*(x[0]))
    
    retval = gp_emulator.create_emulator_validation (simulator, parameters, min_vals, max_vals,
                                n_train, n_validate, do_gradient=True,
                                n_tries=15 )
    return retval
