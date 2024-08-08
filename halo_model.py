import numpy as np
from scipy.integrate import quad

from astropy import units as u
from astropy import constants as c

def delta_vir(z, cosmo):
    '''Calculate the overdensity of halo with respect to critical density 
    following Bryan & Norman 1998.
    Parameters:
       z: redshift
       cosmo: astropy cosmology object
    Return:
       overdensity (unitless). Multiply with cosmo.critical_density(z) to get physical density.
    '''
    x = cosmo.Om(z) - 1
    return 18 * np.pi**2 + 82.0 * x - 39.0 * x**2

def virial_values(M, z, cosmo, mdef = 'vir', quantity='pressure', mu = 0.59):
    '''Calculate the virial/normalized values for different quantities of halos.
    Parameters:
       M: halo mass in Msun unit, h=1
       z: redshift
       cosmo: astropy cosmology object
          Example: from astropy.cosmology import Planck15 as cosmo
       mdef: mass definition
          vir (virial definition from Bryan & Norman 1998)
          xxxc or xxxm (xxx is any integer), e.g. 200c
          scaled_vir (virial scaled to 200 at early times, used in CompaSO)
       quantity: virial quantity to calculate, 
          Choices are radius, density, pressure, temperature, velocity, time.
          radius: physical radius in kpc.
          density: average total density in g/cm^3. Multiply with cosmo.Ob0/cosmo.Om0 for baryonic density
          pressure: rho_gas * G*M/(2R) in keV/cm^3. See Lau+2015 and others.
          temperature: mu*mp*( G*M/(2R) ) in keV. Divide by Boltzmann constant for temperature.
          velocity: sqrt(G*M/R) in km/s.
          time: radius/velocity in Myr. Dynamical/virial crossing time independent of halo mass, and only on mdef and redshift.
          Multiply with 2 for time to return to the virial radius, 2*pi for orbital time
       mu: mean particle weigh. Only used in tempreature.
    Return:
       virial quantity. Can use .to('xx') to convert to desired units. See astropy documentations.
       Default units are radius(kpc), P(keV/cm3), T(keV), v(km/s), t(Myr)
    '''
    try:
        if not M.unit.is_equivalent(u.Msun):
            raise ValueError('wrong Mv unit')
    except AttributeError:
        M = M*u.Msun
        
    rho_crit = cosmo.critical_density(z)
    if mdef == 'vir':
        delta = delta_vir(z, cosmo)
        rho = delta*rho_crit
    elif mdef == 'scaled_vir':
        delta = delta_vir(z, cosmo)*200/(18 * np.pi**2)
        rho = delta*rho_crit
    elif mdef[-1] == 'c':
        delta = int(mdef[:-1])
        rho = delta*rho_crit
    elif mdef[-1] == 'm':
        delta = int(mdef[:-1])
        rho = delta*rho_crit*cosmo.Om(z)
    else:
        raise ValueError("Unsupported mdef")
    fb = cosmo.Ob0/cosmo.Om0
    R = ((3*M/(np.pi*4*rho))**(1./3)).to('kpc')
    if quantity=='radius':
        return R
    elif quantity=='density':
        return rho    
    elif quantity=='pressure':
        return (fb*rho*c.G*M/(2*R)).to('keV/cm3')
    elif quantity=='temperature':
        return (c.G*M*mu*c.m_p/(2*R)).to('keV')
    elif quantity=='velocity':
        return np.sqrt(c.G*M/R).to('km/s')
    elif quantity=='time':
        return (R/np.sqrt(c.G*M/R)).to('Myr')
    else:
        raise ValueError("Unsupported quantity")

def NFWf(x):
    '''KS01: Eq 8'''
    return np.log(1. + x) - x/(1. + x)

def NFWM(r, M, z, c_nfw, R):
    '''KS01: Eq 8'''
    return M * NFWf(c_nfw*r/R) / NFWf(c_nfw)

def Gamma(c_nfw):
    '''KS01: Eq 25'''
    return 1.15 + 0.01*(c_nfw - 6.5)

def eta0(c_nfw):
    '''KS01: Eq 26'''
    return 0.00676*(c_nfw - 6.5)**2 + 0.206*(c_nfw - 6.5) + 2.48

def NFWPhi(r, R, c_nfw):
    return (np.log(1. + c_nfw*r/R) / (c_nfw*r/R))

def theta(r, M, R, z, c_nfw):
    '''KS01: Eq 19'''
    rho0_by_P0 = 3*eta0(c_nfw)**-1 * R/(c.G*M)
    return 1. + ((Gamma(c_nfw) - 1.) / Gamma(c_nfw))*rho0_by_P0.si*(-1. * (c.G * M / R).si * (c_nfw / NFWf(c_nfw)))*(1-NFWPhi(r, R, c_nfw))

def rho0(M, R, z, c_nfw, cbf):
    '''Mg/Mv = fb'''
    nume = cbf * M
    denom = 4. * np.pi * quad(lambda r: theta(r, M, R, z, c_nfw)**(1.0 / (Gamma(c_nfw) - 1.0)) * r**2, 0, R)[0] 
    return nume/denom  ### need units

def rho_gas(r, M, R, z, c_nfw, cbf):
    '''KS01: Eq 15'''
    return (rho0(M.si.value, R.si.value, z, c_nfw, cbf) * theta(r, M, R, z, c_nfw)**(1.0 / (Gamma(c_nfw) - 1.0)) )*u.kg/u.m**3

def sig2_tot(r, M, R, z, c_nfw):
    '''KS01: Eq 16'''
    rho0_by_P0 = 3*eta0(c_nfw)**-1 * R/(c.G*M)
    return (1.0 / rho0_by_P0) * theta(r, M, R, z, c_nfw)

def halo_profile(Mv, z, cosmo, mdef = 'vir', c_nfw=None, mu = 0.59, r=None, conc_model='diemer19', **kwargs):
    '''Calculate the profiles for different quantities of halos according to NFW + Komatsuc & Seljak 2001.
    Parameters:
       Mv: halo mass in Msun unit (h=1)
       z: redshift
       cosmo: astropy cosmology object
          Example: from astropy.cosmology import Planck15 as cosmo
       mdef: overdensity mass definition
          vir (virial definition from Bryan & Norman 1998)
          xxxc or xxxm (xxx is any integer), e.g. 200c
          scaled_vir (virial scaled to 200 at early times, used in CompaSO)
       c_nfw: concentration of NFW profile
          If None, it uses "conc_model" argument to call corrsponding concentration model in Colossus.
       conc_model: Concentration model to be used. 
          Ignored if c_nfw is given.
          Default is 'diemer19'. Will try to set colossus cosmology to be the same as cosmo when used.
       mu: mean particle weigh. Only used in tempreature.
       r: radius at which profile is calculated. Must be given as normalized unit in Rvir, i.e. r/Rvir.
          If None, profile is calculated at 100 logarithmic points from 0.01-2Rvir.
    Return:
       Rv, c_nfw, rhogas, T, P profiles as astropy quantity array. 
       Can use .to('xx') to convert to desired units. See astropy documentations.
       Default units are Rv/r(kpc), rho(g/cm3), T(K), P(keV/cm3), M(Msun/kpc)
    '''
    try:
        if not Mv.unit.is_equivalent(u.Msun):
            raise ValueError('wrong Mv unit')
    except AttributeError:
        Mv = Mv*u.Msun
    
    R = virial_values(Mv, z, cosmo, mdef = mdef, quantity='radius')
    cbf = cosmo.Ob0 / cosmo.Om0
    if r is None:
        r = np.logspace(-2,np.log10(2), num=100)
    rads = r * R 
    
    if c_nfw is None:
        from colossus.cosmology import cosmology
        from colossus.halo import concentration
        try:
            if cosmo.name[:4]=='WMAP':
                cosmo_colossus = cosmology.setCosmology(cosmo.name)
            else:
                cosmo_colossus = cosmology.setCosmology(cosmo.name.lower())
        except:
            cosmo_colossus = cosmology.setCosmology('planck18')
            print("    Colossus does not have cosmology of similar name. Set to Planck18 by default.\n"\
            +"    Use colossus.cosmology.cosmology.setCosmology(your_cosmology) in run time to overwrite.")
        c_nfw = concentration.concentration(Mv.to('Msun').value*cosmo.h, mdef, z, model=conc_model)

    rhogas = rho_gas(rads, Mv, R, z, c_nfw, cbf).to('g/cm3')
    norm_temp = sig2_tot(rads, Mv, R, z, c_nfw)
    P = (norm_temp * rhogas).to('keV/cm3')
    T = (norm_temp*mu*c.m_p).to('K', equivalencies=u.temperature_energy())
    
    return R, c_nfw, rads, rhogas, T, P
    
    
def filament_halo(Mv, z, cosmo, mdef = 'vir', fs = 1./3, Mach = 1, Mvdot = None):
    '''Calculate the filament line-mass feeding given halos.
    Parameters:
       Mv: halo mass in Msun unit, h=1
       z: redshift
       cosmo: astropy cosmology object
          Example: from astropy.cosmology import Planck15 as cosmo
       mdef: mass definition
          vir (virial definition from Bryan & Norman 1998)
          xxxc or xxxm (xxx is any integer), e.g. 200c
          scaled_vir (virial scaled to 200 at early times, used in CompaSO)
       fs: the amount one filament contributes to the total accretion of halo. 
          Default is 1/3.
       Mach: speed at which filament flows in with respect to halo virial velocity. 
          Default is 1.
       Mvdot: dM/dt of halo in equivalent unit of Msun/yr. 
          If None, use Fakhouri et al. 2010. Inaccurate/overestimate at z<2.
    Return:
       Filament line-mass in Msun/kpc unit. Can use .to('xx') to convert to desired units.
    '''
    vvir = virial_values(M, z, cosmo, mdef = mdef, quantity='velocity')
    if Mvdot is None:
        Mvdot = 1572 * (Mv/1e12)**1.1 * ((1+z)/5)**2.5 * u.Msun/u.yr
    return (Mvdot*fs/(Mach*vvir)).to(u.Msun/u.kpc)
    
    
