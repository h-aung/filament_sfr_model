import numpy as np

from scipy import interpolate as interp
from astropy import units as u
from astropy import constants as c
from scipy import optimize as opt
#import customlib as cl
from scipy.integrate import solve_ivp

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

def virial_values_filament(M, z, cosmo, mdef = '40m', quantity='pressure', mu = 0.59):
    '''Calculate the virial/normalized values for different quantities of halos.
    Parameters:
       M: filament mass per unit length in Msun/kpc unit
       z: redshift
       cosmo: astropy cosmology object
          Example: from astropy.cosmology import Planck15 as cosmo
       mdef: overdensity mass definition
          xxxc or xxxm (xxx is any integer), e.g. 200c
          virial temperature and velocity does not depend on mdef unlike halo
       quantity: virial quantity to calculate, 
          Choices are radius, density, pressure, temperature, velocity, time.
          radius: physical radius in kpc.
          density: average total density in g/cm^3. Multiply with cosmo.Ob0/cosmo.Om0 for baryonic density
          temperature: mu*mp*( G*M*(2/3) ) in keV. Divide by Boltzmann constant for temperature. See Lu+2024 and filament_model.pdf.
          pressure: rho_gas * G*M*(2/3) in keV/cm^3.
          velocity: sqrt(G*M) in km/s.
          time: radius/velocity in Myr. Dynamical/virial crossing time independent of filament mass, and only on mdef and redshift.
          Multiply with 2 for time to return to the virial radius, 2*pi for orbital time
       mu: mean particle weigh. Only used in tempreature.
    Return:
       virial quantity. Can use .to('xx') to convert to desired units. See astropy documentations.
       Default units are radius(kpc), P(keV/cm3), T(keV), v(km/s), t(Myr)
    '''
    try:
        if not M.unit.is_equivalent(u.Msun/u.kpc):
            raise ValueError('wrong Mv unit')
    except AttributeError:
        M = M*u.Msun/u.kpc
    rho_crit = cosmo.critical_density(z)
    if mdef[-1] == 'c':
        delta = int(mdef[:-1])
        rho = delta*rho_crit
    elif mdef[-1] == 'm':
        delta = int(mdef[:-1])
        rho = delta*rho_crit*cosmo.Om(z)
    else:
        raise ValueError("Unsupported mdef")
    fb = cosmo.Ob0/cosmo.Om0
    R = np.sqrt(M/(np.pi * rho)).to('kpc')
    if quantity=='radius':
        return R
    elif quantity=='density':
        return rho    
    elif quantity=='pressure':
        return ( fb * rho * c.G * M /3 ).to('keV/cm3')
    elif quantity=='temperature':
        return ( c.G * M * mu * c.m_p/3).to('keV')
    elif quantity=='velocity':
        return np.sqrt(c.G*M).to('km/s')
    elif quantity=='time':
        return (R/np.sqrt(c.G*M)).to('Myr')
    else:
        raise ValueError("Unsupported quantity")

def filament_profile(Mv, z, cosmo, mdef = '40m', mu = 0.59, model='isothermal', r=None, **kwargs):
    '''Calculate the virial/normalized values for different quantities of halos.
    Parameters:
       M: filament mass per unit length in Msun/kpc unit
       z: redshift
       cosmo: astropy cosmology object
          Example: from astropy.cosmology import Planck15 as cosmo
       mdef: overdensity mass definition
          xxxc or xxxm (xxx is any integer), e.g. 200c. Ignored for model='collapse'. 
       mu: mean particle weigh. Only used in tempreature.
       model: isothermal - assume isothermal profile, see Lu+2024.
       collapse - assume self-similar cylindrical collapse, see Aung+2024 and filament_model.pdf.
       r: radius at which profile is calculated. Must be given as normalized unit in Rvir, i.e. r/Rvir.
       If None, profile is calculated at 100 logarithmic points from 0.01-1Rvir.
       Not used in collapse model right now.
       **kwargs: for collapse solver
    Return:
       Rv, r, rhogas, T, P, Mgas profiles as astropy quantity array. 
       Can use .to('xx') to convert to desired units. See astropy documentations.
       Default units are Rv/r(kpc), rho(g/cm3), T(keV), P(keV/cm3), M(Msun/kpc)
    '''
    if model=='isothermal':
        return filament_profile_isothermal(Mv, z, cosmo, mdef = mdef, mu = mu, r=r, dm=True)
    elif model=='isothermal_nodm':
        return filament_profile_isothermal(Mv, z, cosmo, mdef = mdef, mu = mu, r=r, dm=False)
    elif model=='collapse':
        return filament_profile_collapse(Mv, z, cosmo, mdef=mdef, mu = mu, **kwargs)
    else:
        raise ValueError("Unsupported model")
        
        
def filament_profile_isothermal(Mv, z, cosmo, mdef = '40m', mu = 0.59, r=None, dm=True):
    '''Called by filament_profile. 
    '''
    try:
        if not Mv.unit.is_equivalent(u.Msun/u.kpc):
            raise ValueError('wrong Mv unit')
    except AttributeError:
        Mv = Mv*u.Msun/u.kpc
    Rv = virial_values_filament(Mv, z, cosmo, mdef=mdef, quantity='radius')
    Tv = virial_values_filament(Mv, z, cosmo, mdef=mdef, quantity='temperature', mu=mu)
    Mcrit = 2*Tv/(c.G*mu*c.m_p) #critical line-mass
    if r is None:
        r = np.logspace(-2,0)
    r = r*Rv
    r0 = 0.25*Rv #from Lu+2024
    rho0 = Mcrit/(np.pi*r0**2)
    rho = rho0/(1 + (r/r0)**2)**2
    T = Tv*np.ones(len(r))
    if dm:
        fb = cosmo.Ob0/cosmo.Om0
    else:
        fb = 1
    P = (fb*rho*T/(mu*c.m_p)).to('keV/cm3')
    M = Mcrit/(1+ (r0/r)**2)
    normm = (Mv/(Mcrit/(1+ (r0/Rv)**2))).to('')
    return Rv.to('kpc'), r.to('kpc'), normm*fb*rho.to('g/cm3'), T.to('keV'), P.to('keV/cm3'), fb*normm*M.to('Msun/kpc')

def continuity_eq_p(r,init,delta,gamma): 
    '''Eq B6 with P'''
    D,P,V,M=init[0],init[1],init[2],init[3]
    Vs = V - delta*r
    Mat = [[Vs*r,0.0,D*r,0.0],
       [0.0,r,Vs*D*r,0.0], 
       [-gamma*Vs*P, Vs*D,0.0,0.0],
       [0.0,0.0,0.0,1.0]]
    b = [[2.0*D*r-D*V],
       [D*(r**2)/9+V*D*r*(1-delta)-M*D/3.0],
       [D*P*(4.0-2.0*delta-2.0*gamma)],
       [2.0*r*D]]
    return np.linalg.solve(Mat,b).T[0]

def continuity_eq_nop(r,init,delta,gamma): 
    '''Eq B6 with no P'''
    D,V,M=init[0],init[1],init[2]
    Vs = V - delta*r
    Mat = [[Vs*r,D*r,0.0],
       [0.0,Vs*r,0.0], 
       [0.0,0.0,1.0]]
    b = [[2.0*D*r-D*V],
       [(r**2)/9+V*r*(1-delta)-M/3.0],
       [2.0*r*D]]
    return np.linalg.solve(Mat,b).T[0]

def shock_jump(D1, V1, M1, delta, lambdash, gamma):
    '''Apply normal shock jump condition with infinite Mach, Eq B7.
    return D2, P2, V2, M2'''
    Vs = V1 - delta*lambdash
    return [(gamma+1)*D1/(gamma-1),(2*D1)*(Vs)**2/(gamma+1),
            delta*lambdash+(gamma-1)*Vs/(gamma+1),M1]

def filament_profile_collapse_nonnorm(init,delta,gamma,test_incr = 0.01, inner_tol = 1e-10, r_eval=None):
    '''Solve eq B6 and B7 in appendix and output lambda_sh, M_sh, lambda, D, P, V, M
    '''
    innerM, V, test_lambdash = 1, -1, 0.01
    if r_eval is not None:
        inner_tol_compute = np.amin(r_eval)
        incr_compute = np.amin(np.abs(r_eval[1] - r_eval[0]))
        if inner_tol > inner_tol_compute:
            inner_tol = inner_tol_compute
        if test_incr > incr_compute:
            test_incr = incr_compute
    nop_solution = solve_ivp(lambda r,var: continuity_eq_nop(r,var,delta,gamma), (1.00,inner_tol), init,method='Radau', t_eval = r_eval,dense_output=True)
    while V < 0 and innerM > 0:
        mask = nop_solution.t>=test_lambdash
        preshock_sol = nop_solution.y[:,mask]
        new_init = shock_jump(preshock_sol[0,-1], preshock_sol[1,-1], preshock_sol[2,-1], delta, test_lambdash, gamma)
        if r_eval is not None:
            t_eval = r_eval[r_eval<test_lambdash]
        else:
            t_eval = None
        postshock_sol = solve_ivp(lambda r,var: continuity_eq_p(r,var,delta,gamma), (test_lambdash,inner_tol), new_init ,method='Radau',dense_output=True, t_eval = t_eval)
        innerv = postshock_sol.y[2,-1]
        innerM = postshock_sol.y[3,-1]
        if innerv > 0 and innerM > 0:
            test_lambdash = test_lambdash-test_incr*innerv/(innerv - V)
        if innerM < 0 : 
            test_lambdash = np.nan
        test_lambdash += test_incr
        if test_lambdash<0:
            raise Error("Unknown Error")
        V = innerv
    t = np.r_[nop_solution.t[mask] , postshock_sol.t]
    D = np.r_[preshock_sol[0] , postshock_sol.y[0]]
    P = np.r_[np.zeros_like(nop_solution.t[mask]), postshock_sol.y[1]]
    V = np.r_[preshock_sol[1] , postshock_sol.y[2]]
    M = np.r_[preshock_sol[2] , postshock_sol.y[3]]
    return test_lambdash, preshock_sol[2,-1], t, D, P, V, M

def filament_profile_collapse(Mv, z, cosmo, mdef = '40m',mu = 0.59, dm=True, s=1.5,**kwargs):
    '''Called by filament_profile. 
    '''
    try:
        if not Mv.unit.is_equivalent(u.Msun/u.kpc):
            raise ValueError('wrong Mv unit')
    except AttributeError:
        Mv = Mv*u.Msun/u.kpc
    delta = (s+2)/3
    gamma = 5./3
    lambda_sh, M_sh, r, D, P, V, M = filament_profile_collapse_nonnorm([0.16, 0.  , 3.53],delta,gamma, r_eval = np.logspace(0,-3,num=100), **kwargs)
    T_sh = (P/D)[P>0][0]
    M = M*Mv/M_sh
    Mta = M[0]/3.53
    rhob = cosmo.Om(z) * cosmo.critical_density(z) 
    rta = np.sqrt(Mta / (np.pi * rhob) )
    Tv = virial_values_filament(Mv, z, cosmo, mdef=mdef, quantity='temperature', mu=mu)
    r = r*rta
    rho = D*rhob
    T = (P/D)*Tv/T_sh
    P = rho*T/(mu*c.m_p)
    return lambda_sh*rta.to('kpc'), r.to('kpc'), rho.to('g/cm3'), T.to('keV'), P.to('keV/cm3'), M.to('Msun/kpc')

def tcool(T, rho, metallicity = 0., gamma=5./3, mu=0.59):
    '''Return cooling time'''
    return T.to('keV',  equivalencies=u.temperature_energy()) / \
    ((gamma-1)*(rho/(mu*c.m_p))* C.Lambda_units(metallicity,T.to('K', equivalencies=u.temperature_energy()))) 

def tff(rhobar):
    '''Return freefall time of cylindar'''
    return np.sqrt(3*np.pi/(32*c.G*rhobar))