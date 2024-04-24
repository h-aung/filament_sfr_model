import numpy as np

from scipy import interpolate as interp
from astropy import units as u
from astropy import constants as c
from scipy import optimize as opt
import customlib as cl

from astropy.cosmology import Planck15 as cosmo

def safelog(f):
    if hasattr(f, '__array__'):
        f = np.where(f<1e-33, 1e-33, f)
    elif f<1e-33:
        f = 1e-33
    return np.log10(f)

class two_d_table:
    '''
    read and interpolate 2D tables, with format:
    #comment
         x1   x2   x3 x4 ...
    y1   z11 z12 z13 z14
    y2   z21 z22 z23 z24
    .
    .
    .
    '''
    def __init__(self,file,x_in_log=False,y_in_log=False,z_in_log=False,x_interp_log=False,y_interp_log=False,z_interp_log=False,bound_val=np.NaN):
        self.x_in_log=x_in_log
        self.y_in_log=y_in_log
        self.z_in_log=z_in_log       
        self.x_interp_log=x_interp_log
        self.y_interp_log=y_interp_log
        self.z_interp_log=z_interp_log
        self.bound_val=bound_val
        f=open(file,'r')
        header_read=False
        tab_alloc=False
        for l in f:
            li=l.strip()
            if not li.startswith("#"):
                if not header_read:
                    x = np.array([float(x) for x in l.split()])
                    x=self.val_to_store(x,x_in_log,x_interp_log)
                    header_read=True
                else:
                    ll=[float(x) for x in l.split()]
                    yy=self.val_to_store(ll[0],y_in_log,y_interp_log)
                    zvec=np.array(ll[1:])
                    zvec=self.val_to_store(zvec,z_in_log,z_interp_log)
                    if tab_alloc:
                        y=np.append(y,[yy])
                        ztab=np.vstack((ztab,zvec))
                    else:
                        y=np.array([yy])
                        ztab=np.array(zvec)
                        tab_alloc=True
        self.table=interp.RectBivariateSpline(y,x,ztab,kx=1,ky=1)
        print('Read table:'+file)
        self.xbound=np.array([min(x),max(x)])

        self.ybound=np.array([min(y),max(y)])

    def __call__(self,x,y):
        if self.x_interp_log:
            xx=safelog(x)
        else:
            xx=x
        if self.y_interp_log:
            yy=safelog(y)
        else:
            yy=y
        zz=self.table(yy,xx)
        if self.z_interp_log:
            z=10**zz
        else:
            z=zz
        if not np.isnan(self.bound_val):
            for i,q in enumerate(np.array([yy]).flatten()):
                if q < self.ybound[0] or q>self.ybound[1]:
                    z[i]=self.bound_val
            for i,q in enumerate(np.array([xx]).flatten()):
                if q < self.xbound[0] or q>self.xbound[1]:
                    z[i]=self.bound_val
        return z if len(z)>1 else float(z[0])

    def val_to_store(self,f,in_log,interp_log):
        if in_log and not interp_log:
            return 10**f
        elif not in_log and interp_log:
            return safelog(f)
        else:
            return f

from scipy import interpolate as interp
from astropy import units as u
from astropy import constants as c
from scipy import optimize as opt

# constants
eV_to_erg = 1.60217733e-12
eV_to_gr = eV_to_erg/c.c**2
N_A=c.N_A.to(u.mol**(-1)).value
k_B=c.k_B.cgs.value
a_radiation_constant=7.5657e-15/eV_to_erg #eV.cm^-3.K^-4
Msolar=c.M_sun.to(u.g).value
megayear=u.megayear.to(u.s)
year=u.year.to(u.s)
pc=u.pc.to(u.cm)
kpc=1000*pc

class Cooling(two_d_table):
    'Return Sutherland and Dopita cooling by interpolating CIE table'
    def __init__(self,file='lambda',cooling_nH2=False,x_in_log=False,y_in_log=True,z_in_log=True,x_interp_log=True,y_interp_log=True,z_interp_log=True,bound_val=np.NaN):
        super().__init__(file,x_in_log,y_in_log,z_in_log,x_interp_log,y_interp_log,z_interp_log,bound_val)
        self.cooling_nH2=cooling_nH2

    def Lambda(self,zmet,temp):
        return super().__call__(zmet,temp)

    def Lambda_units(self,zmet,temp):
        res=self.Lambda(zmet,temp.to(u.K).value)
        res=res*u.erg/u.s*u.cm**3
        return res

    def macroscopic_cooling(self,mcg,comp):
        if comp.baryonic_cooling:
            #returns erg/sec/gram
            if self.cooling_nH2:
                prefactor=(N_A*mcg.nHnp)**2
            else:
                prefactor=(N_A/mcg.mu*mcg.chi)**2
            lam=(self.Lambda(mcg.zmetal,comp.Temperature(mcg)))*prefactor*mcg.dens
            #print('Cooling:Z,T,dens,lam=',mcg.zmetal,comp.Temperature(mcg),mcg.dens,lam)
            return lam
        else:
            return 0.
            
C=Cooling(file='CIE_cool_M5.txt',cooling_nH2=True)

def tcooltff(radius, halo_mass, z, cosmo, s = 1.2, metallicity = 0.):
    if s=="derive":
        s = ((3*cosmo.age(z)/2)*(2.8*((1+z)/8)**1.5)/u.Gyr).to('')
    delta_from_s = interp.interp1d([1.5,5],[5,50],fill_value='extrapolate')
    cosmic_fraction = cosmo.Ob0/cosmo.Om0
    #radius = np.logspace(-2,0) ### unit less
    density = delta_from_s(s)*cosmo.critical_density(z)*cosmo.Om(z)/radius**(1.62)
    filament_mass = 0.03*((halo_mass/1e12)**0.14)*((1+z)**2.5)*halo_mass*u.Msun/(cl.vcirc(halo_mass, z, 'vir',cosmo) * u.Gyr)   /3

    gamma = 5./3
    mu = 0.61
    virial_T = ((c.G * filament_mass*(mu*c.m_p))/c.k_B).to('K')
    virial_radii = np.sqrt(filament_mass*0.34/(2*np.pi*density[-1]))
    mass = 2*np.pi*density[-1]*(radius**0.34)*virial_radii**2 *cosmic_fraction/ 0.34
    tcool = c.k_B*virial_T / \
    ((gamma-1)*(density*cosmic_fraction/(mu*c.m_p))* C.Lambda_units(metallicity,virial_T)) 

    tff = radius*virial_radii/np.sqrt(c.G * mass)
    return (tcool/tff).to(''), (mass/filament_mass/cosmic_fraction).to('')
    
    
def tcooltff_isothermal(radius, halo_mass, z, s = 1.2, metallicity = 0.):
    if s=="derive":
        s = ((3*cosmo.age(z)/2)*(2.8*((1+z)/8)**1.5)/u.Gyr).to('')
    delta_from_s = interp.interp1d([1.5,5],[5,50],fill_value='extrapolate')
    cosmic_fraction = cosmo.Ob0/cosmo.Om0
    #radius = np.logspace(-2,0) ### unit less
    filament_mass = 0.03*((halo_mass/1e12)**0.14)*((1+z)**2.5)*halo_mass*u.Msun/(cl.vcirc(halo_mass, z, 'vir',cosmo) * u.Gyr)   /3
    gamma = 5./3
    mu = 0.61
    virial_T = ((c.G * filament_mass*(mu*c.m_p))/c.k_B).to('K')
    density = delta_from_s(s)*cosmo.critical_density(z)*cosmo.Om(z)/radius**(1.62)
    virial_radii = np.sqrt(filament_mass*0.34/(2*np.pi*density[-1]))
    rhoc = filament_mass / ( (32./57) * np.pi * virial_radii**2 )
    cs = np.sqrt(gamma*c.k_B*virial_T/(mu*c.m_p))
    H = 1*virial_radii
    
    density = rhoc*(1+ (190./128)*radius**2 )**(-2)
    
    mass = np.pi * rhoc * cosmic_fraction *virial_radii**2 * (radius**2/ (1+ 25.*radius**2/ 32.))
    tcool = c.k_B*virial_T / \
    ((gamma-1)*(density*cosmic_fraction/(mu*c.m_p))* C.Lambda_units(metallicity,virial_T)) 
    tff = radius*virial_radii/np.sqrt(c.G * mass)
    return (2*tcool/tff).to(''), (mass/filament_mass/cosmic_fraction).to('')

from colossus.halo import mass_defs
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_so, profile_nfw
cosmo_colossus = cosmology.setCosmology('planck15')
from colossus.lss import peaks
import colossus
from scipy.integrate import quad

def Gamma(c_nfw):
    return 1.15 + 0.01*(c_nfw - 6.5)

def eta0(c_nfw):
    return 0.00676*(c_nfw - 6.5)**2 + 0.206*(c_nfw - 6.5) + 2.48

def NFWPhi(r, M, z, c, mass_def='vir'):
    R = mass_so.M_to_R(M, z, mass_def)
    if(type(r) != np.ndarray and r == 0):
        return -1. * (G * M / R) * (c / NFWf(c))
    else:
        return -1. * (G * M / R) * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))

# this now agrees with Komatsu and Seljak eqn 19 for theta
# the confusion was that rho0 / P0 is 3*eta0^-1 and then units removed by scaling by R/GM
def theta(r, M, z, c, mass_def='vir'):
    R = mass_so.M_to_R(M, z, mass_def)
    # the rho0/P0 is actually 3eta^-1(0) * R/(GM) from Komatsu and Seljak
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    return 1. + ((Gamma(c) - 1.) / Gamma(c))*rho0_by_P0*(NFWPhi(0, M, z, c, mass_def='vir')-NFWPhi(r, M, z, c, mass_def='vir'))

def rho0(M, z, c = None, conc_model='diemer19', mass_def='vir'):
    if c is None:
        c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    nume = cbf * M
    denom = 4. * np.pi * quad(lambda r: theta(r, M, z, c, mass_def=mass_def)**(1.0 / (Gamma(c) - 1.0)) * r**2, 0, R)[0]
    return nume/denom

def rho_gas(r, M, z, c = None, conc_model='diemer19', mass_def='vir'):
    if c is None:
        c = concentration.concentration(M, mass_def, z, model=conc_model)
        return rho0(M, z, c=c, mass_def=mass_def) * theta(r, M, z, c, mass_def=mass_def)**(1.0 / (Gamma(c) - 1.0)), c
    else:
        return rho0(M, z, c=c, mass_def=mass_def) * theta(r, M, z, c, mass_def=mass_def)**(1.0 / (Gamma(c) - 1.0))

def sig2_tot(r, M, z, conc_model='diemer19', mass_def='vir', c=None):
    if c is None:
        c = concentration.concentration(M, mass_def, z, model=conc_model) # tabulated probably
    R = mass_so.M_to_R(M, z, mass_def) # tabulated probably
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gamma(c) - 1.) / Gamma(c)) * 3. *eta0(c)**-1 * (phi0 - phir)
    return (1.0 / rho0_by_P0) * theta

G = colossus.utils.constants.G
Msunkpc3_2_gcm3 = 6.77e-32

def NFWf(x):
    return np.log(1. + x) - x/(1. + x)

cbf = cosmo.Ob0 / cosmo.Om0
cosmo_to_evcm3 = 4.224e-10
eV_to_erg = 1.602e-12

def NFWM(r, M, z, c, R):
    return M * NFWf(c*r/R) / NFWf(c)
    
def tcooltff_halo(radius, halo_mass, z, metallicity = 0.):
    mass = cl.add_perh(halo_mass , cosmo.h) #Msun/h
    Rvir = mass_so.M_to_R(mass, z, 'vir') # kpc / h
    rads = radius * Rvir #np.logspace(np.log10(0.01*Rvir),np.log10(1.1*Rvir), 1000)
    norm_dens, conc = rho_gas(rads, mass, z, mass_def='vir')
    norm_temp = sig2_tot(rads, mass, z, mass_def='vir')
    pressure = norm_temp * norm_dens
    mu = 0.61
    gamma = 5./3
    cosmic_fraction = cosmo.Ob0/cosmo.Om0
    density = norm_dens*Msunkpc3_2_gcm3 * u.g/u.cm**3
    temperature = ((pressure*cosmo_to_evcm3 *eV_to_erg * u.erg/u.cm**3)* \
                   (mu*c.m_p)/ (density * c.k_B)).to('K')
    tcool = c.k_B*temperature / \
    ((gamma-1)*(density*cosmic_fraction/(mu*c.m_p))* C.Lambda_units(metallicity,temperature).flatten())
    enclosed_mass = cl.remove_perh(NFWM(rads, mass, z, conc, Rvir), cosmo.h)
    tff = cl.remove_perh(rads, cosmo.h) *u.kpc/np.sqrt(c.G * enclosed_mass *u.Msun / (rads *u.kpc) )
    return (tcool/tff).to('')/1.5, enclosed_mass/halo_mass
    
    
def halo_mar(m, z):
    return (0.03/cl.u.Gyr)*m*cl.u.Msun*((m/1e12)**0.14)*(1+z)**(5./2)

def filament_T(m, z, facc=1):
    mfil = (halo_mar(m, z)/3) /cl.virial_values(m, z, cosmo, mdef='200c',quantity='velocity')
    return cl.c.G*mfil*(0.59*cl.c.m_p)