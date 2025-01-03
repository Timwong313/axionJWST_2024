##########Packages##########
from astropy import stats
from scipy import constants
from scipy import interpolate
from scipy import optimize
from scipy import integrate
import numpy as np
import math
import time
import sys
import random
import multiprocessing as mp
import os

##########Parameters##########
J_to_GeV = 6.242  #1e9 
kpc_to_cm = 3.086e12  #1e9
rho_s = 0.29 #0.18  #GeV/cm^3
rs = 19.1 #24  #kpc
R_sun = 8.27  #kpc
sigma_v = 160000/constants.c #m/s/c
solarV = 250000 #m/s
testRange = 150  # Radius of the modelling range (~75FWHM of the DM line)
anchorNo = 5  #No. of anchor points in cubic spline
step = 300  #3  #scanning the mass range in the given step (~FWHM of DM line)
maskRange = 1   #radius of the mask (in index)
dataNo = 1  #No. of simulated data sets

##########General##########
def quadgrowth(start,end,N,w=0.9):
    #0<w<1
    stepArr = np.array(range(0,N))
    d0 = (end-start)/(N-1)
    dk = d0*(1-w*(N-stepArr-1)/(N-1))
    return start+stepArr*dk

def locate(arr,x): 
    #arr[i] just larger than x
    i = 0
    while arr[i] < x:
        i += 1
    return i

def massToWvl(m):
    #eV to um
    return 2*constants.Planck*constants.c/m/constants.e/1e-6

def wvlToMass(wvl):
    #um to eV
    return 2*constants.Planck*constants.c/wvl/constants.e/1e-6

def couplingToGamma(g,ma):  
    #GeV-1 => s-1
    return g**2*ma**3/64/np.pi/658.2119569

def gammaToCoupling(gamma,ma):  
    #s-1 => GeV-1
    return np.sqrt(64*math.pi*658.2119569*gamma/ma**3)

def FWHMestim(S_spec,M_spec,L_spec):
    #FWHM estimation:
    Swvl = np.mean(S_spec[:2])
    Mwvl = np.mean(M_spec[:2])
    Lwvl = np.mean(L_spec[:2])
    Sres = np.mean(S_spec[2:])
    Mres = np.mean(M_spec[2:])
    Lres = np.mean(L_spec[2:])
    return (Swvl/Sres+Mwvl/Mres+Lwvl/Lres)/3

def minimization(x,y,BOUND=(-1e9,1e9)):
    f = interpolate.interp1d(x,y)
    F = lambda x : float(f(x))
    min = optimize.minimize_scalar(F,bounds=BOUND)
    return min.x, min.fun

##########ConservConstraint##########
class line_model:

    def __init__(self,longArr,latArr):
        self.longArr = longArr
        self.latArr = latArr
        self.fileNo = len(longArr)
        self.D = integrate.quad_vec(self.NFW_profile,0,np.inf)[0]*(kpc_to_cm/J_to_GeV)*1e18  #TJy/sr
        self.avgD = np.mean(self.D*J_to_GeV/kpc_to_cm*1e-18)
        self.solarLosV = solarV*np.sin(self.longArr)*np.cos(self.latArr) 
        self.dopplerFac = np.sqrt((1-self.solarLosV/constants.c)/(1+self.solarLosV/constants.c))
    
    #NFW profile in terms of observing distance:
    def NFW_profile(self,s):  #s in kpc
        r_rs = np.sqrt(R_sun*R_sun+s*s-s*R_sun*np.cos(self.longArr)*np.cos(self.latArr))/rs
        return rho_s/(r_rs+r_rs*r_rs)

    def effSpectrum(self,wvl,exptArr,d_lmd):
        w = wvl*sigma_v  #um
        sigma_lmd = d_lmd/2/np.sqrt(2*np.log(2))  #um
        newSigma2 = sigma_lmd*sigma_lmd+w*w  #um^2
        normalize = wvl*wvl/np.sqrt(2*math.pi*newSigma2)/constants.c
        print('###########################')
        print('doppler width = ', np.mean(w))
        print('sigma_lmd', sigma_lmd)
        print('sigma = ', np.mean(np.sqrt(newSigma2)))
        # print('normalize = ', normalize)
        print('###########################')
        
        spectrum_eff = np.zeros((len(wvl),len(wvl)))
        weight = exptArr/np.sum(exptArr)
        for i in range(len(wvl)):
            lmd0 = wvl[i]
            for j in range(self.fileNo):
                shftWvl = wvl/self.dopplerFac[j]
                spectrum_eff[i,:] += self.D[j]*normalize[i]*np.exp(-(shftWvl-lmd0)**2/2/newSigma2[i])*weight[j] #us
        return spectrum_eff

def theoFlux(gamma,spectrum_eff):
    return gamma*spectrum_eff/4/math.pi
    
def model_range(i,wvl,flux,error,spectrum):
    spec = spectrum[i,:]
    if (i>=testRange) & (i<=len(wvl)-testRange):
        wvlCut = wvl[i-testRange:i+testRange]
        fluxCut = flux[i-testRange:i+testRange]
        errorCut = error[i-testRange:i+testRange]
        specCut = spec[i-testRange:i+testRange]
    elif (i<testRange):
        wvlCut = wvl[:testRange*2]
        fluxCut = flux[:testRange*2]
        errorCut = error[:testRange*2]
        specCut = spec[:testRange*2]
    elif (i>len(wvl)-testRange):
        wvlCut = wvl[-testRange*2:]
        fluxCut = flux[-testRange*2:]
        errorCut = error[-testRange*2:]
        specCut = spec[-testRange*2:]

    return wvlCut,fluxCut,errorCut,specCut

def extractBound(gammaArr,chi2,Null,c=2.71):  
    #Null=True: checking chi2-0
    #Null=False: checking chi2-min(chi2)
    if (chi2.ndim==1):
        chi2 = np.reshape(chi2,(1,len(chi2)))
    testLen = np.ma.size(chi2,axis=0)
    gammaBound = np.zeros(testLen)
    for i in range(testLen):
        if (Null==False):
            min_i = np.argmin(chi2[i,:])
            dchi2 = chi2[i,min_i:] - chi2[i,min_i]
            gamma = gammaArr[min_i:]
        else:
            dchi2 = chi2[i,:]
            gamma = gammaArr
        gammaBound[i] = np.interp(c,dchi2,gamma)
    return gammaBound

def chi2_conservative(obsWvl,obsFlux,obsError,gammaArr,spectrum):  
    #output chi2 in the searching space of gamma and lamda0
    start = time.time()
    wvl_bd = obsWvl[::step]
    chi2 = np.zeros((len(wvl_bd),len(gammaArr)))

    #####Computation#####
    for i in range(len(wvl_bd)):
        lmd0 = wvl_bd[i]
        wvlCut, fluxCut, errorCut, specCut = model_range(step*i,obsWvl,obsFlux,obsError,spectrum)
        for j in range(len(gammaArr)):
            gamma = gammaArr[j]
            flux_model = theoFlux(gamma,specCut)
            chi2[i,j] = np.sum(((flux_model-fluxCut)/errorCut)**2,where=(flux_model-fluxCut)>0)             
    gammaBd = extractBound(gammaArr,chi2,Null=True)
    #####################

    end = time.time()
    print('Duration: ', (end - start)/60, ' [min]')
    return chi2, gammaBd, wvl_bd

##########ContinConstraint##########
def fluxModel(wvl,gamma,spectrum_eff,beta):
        wvlSpline = np.linspace(wvl[0],wvl[-1],len(beta))
        fluxModel =  theoFlux(gamma,spectrum_eff) + interpolate.CubicSpline(wvlSpline,beta)(wvl)
        return fluxModel

def lnlike_continuum(gamma,spectrum_eff,beta,wvl,flux,error,mask=None):
        flux_model = fluxModel(wvl,gamma,spectrum_eff,beta)
        return np.sum(((flux_model-flux)/error)**2,where=(mask==False)) 

def mask(wvl, flux, error, spec):
    #Mask the interval=maskRange which has a >3sigma fluctuation
    mask_condition = np.zeros(len(flux),dtype=bool)
    initial = np.linspace(flux[0],flux[-1],anchorNo)
    likelihood = lambda beta : lnlike_continuum(0,spec,beta,wvl,flux,error)
    minLike = optimize.minimize(likelihood,initial)
    nullModel = fluxModel(wvl,0,spec,minLike.x)
    df = flux - nullModel
    for i in range(maskRange,len(flux)-maskRange):
        if np.all(abs(df)[i-maskRange:i+maskRange+1] > 3*error[i-maskRange:i+maskRange+1]):
            mask_condition[i-maskRange:i+maskRange] = True
    return mask_condition

def contFitting(gammaArr, spectrum_eff, wvl, flux, error):
    chi2 = np.zeros(len(gammaArr))
    minBeta = np.zeros((len(gammaArr),anchorNo))
    initial = np.linspace(flux[0],flux[-1],anchorNo)
    maskCondition = mask(wvl,flux,error,spectrum_eff)

    for i in range(len(gammaArr)):
        gamma = gammaArr[i]
        likelihood = lambda beta : lnlike_continuum(gamma, spectrum_eff, beta, wvl, flux, error, mask=maskCondition)
        minLike = optimize.minimize(likelihood,initial)
        minBeta[i,:] = minLike.x
        chi2[i] = minLike.fun
    min_i = np.argmin(chi2)
    gammaBd = extractBound(gammaArr,chi2,Null=False)
    likelihood = lambda beta : lnlike_continuum(gammaBd, spectrum_eff, beta, wvl, flux, error, mask=maskCondition)
    minLike = optimize.minimize(likelihood,initial)
    modelFit = np.transpose(np.array([fluxModel(wvl,gammaArr[min_i],spectrum_eff,minBeta[min_i,:]),fluxModel(wvl,gammaBd,spectrum_eff,minLike.x),fluxModel(wvl,0,spectrum_eff,minBeta[0,:])]))
    return chi2, gammaBd, modelFit, minBeta, maskCondition

def simulateData(flux_model,error):
    simData = np.random.normal(loc=flux_model,scale=error,size=(dataNo,2*testRange)) #assume gaussian
    return simData

def sensitivityBand(wvl,error,flux_model,gammaArr,spectrum_eff):  
    simFlux = simulateData(flux_model,error)
    simChi2 = np.zeros((dataNo,len(gammaArr)))
    simFit = np.zeros((dataNo,testRange*2,3))
    simBeta = np.zeros((dataNo,len(gammaArr),anchorNo))
    gammaBound = np.zeros(dataNo)
    arg1 = [gammaArr]*dataNo
    arg2 = [spectrum_eff]*dataNo
    arg3 = [wvl]*dataNo
    arg4 = [simFlux[i,:] for i in range(dataNo)]
    arg5 = [error]*dataNo 
    result = list(map(contFitting,arg1,arg2,arg3,arg4,arg5))
    for i in range(dataNo):
        simChi2[i,:], gammaBound[i], simFit[i,:,:],_,_ = result[i]  #contFitting(gammaArr,spectrum_eff,wvl,simFlux[i,:],error) 
    #gammaBand = np.array([np.mean(gammaBound),np.std(gammaBound)])
    band68 = np.array([np.percentile(gammaBound,84),np.percentile(gammaBound,16)])
    band95 = np.array([np.percentile(gammaBound,97.5),np.percentile(gammaBound,2.5)])
    gammaBand = np.array([band68,band95])
    return gammaBand, gammaBound, simFlux, simFit, simChi2

def fit_sim(gammaArr, wvl, flux, error, spec) :
    chi2, gammaBd, modelFit, beta, mask = contFitting(gammaArr, spec, wvl, flux, error)
    result = (chi2,gammaBd,modelFit,beta,mask) + sensitivityBand(wvl,error,modelFit[:,0],gammaArr,spec)
    return result

def detection_signif(chi2):
    N = np.sqrt(chi2[:,0]-np.min(chi2,axis=1))
    return N

class continuumFitting(mp.Process):  
    #gammaArr has to be monotonically increasing
    def __init__(self,queue,gammaArr,wvl,flux,error,spectrum):
        mp.Process.__init__(self)
        self.queue = queue
        self.gamma = gammaArr
        self.wvl = wvl
        self.flux = flux
        self.error = error
        self.spec = spectrum  
        self.gamma_len = len(self.gamma)
        self.lenWvl = len(wvl)  #length of the input wavelength array
        self.wvl_bd = wvl[::step]  #wavelength array that corresponds to the testing masses
        self.test_len = len(self.wvl_bd)  #length of the testing wavelength array
        self.jobnum = random.randint(100,999)
        self.workno = int(os.cpu_count()/4)
        print('('+str(self.jobnum)+'): No. of test masses: ', self.test_len)
            
    def run(self):
        print('Job ('+str(self.jobnum)+') starts...')

        #####First trail#####
        stime = time.time()
        wvlCut, fluxCut, errorCut, specCut = model_range(0,self.wvl,self.flux,self.error,self.spec)
        self.result = [fit_sim(self.gamma, wvlCut, fluxCut, errorCut, specCut)]
        etime =  time.time()
        print('('+str(self.jobnum)+')\'s estimated computational time:', (etime-stime)*self.test_len/7200, ' [hrs]')
        sys.stdout.flush()

        #####Main calculation#####
        fitArgs = [(self.gamma,)+model_range(step*i,self.wvl,self.flux,self.error,self.spec) for i in range(1,self.test_len)]
        pool = mp.Pool(self.workno)
        self.result += pool.starmap(fit_sim,fitArgs,chunksize=math.ceil(self.test_len/self.workno))
        pool.close()
        pool.join()
        self.queue.put(self.result,block=True)

    # def bestFit(self,lmd0,gammaArr,chi2,wvl,flux,error):
    #     gamma_bf,like_bf = minimization(gammaArr,chi2,BOUND=(0,gammaArr[np.argmin(chi2)+5]))
    #     initial = np.linspace(flux[0],flux[-1],5)
    #     likelihood = lambda beta : self.lnlike_continuum(lmd0, gamma_bf, beta, wvl, flux, error)
    #     minLike = optimize.minimize(likelihood,initial)
    #     minBeta = optimize.minimize(likelihood,initial).x
    #     return np.append(gamma_bf,minBeta), minLike.fun

class fittingResult:

    def __init__(self,result):
        self.test_len = len(result)
        self.gamma_len = np.size(result[0][0])
        self.chi2 = np.zeros((self.test_len,self.gamma_len))
        self.gammaBd = np.zeros(self.test_len)
        self.gammaBand = np.zeros((self.test_len,2,2))
        self.modelFit = np.zeros((self.test_len,testRange*2,3))
        self.beta = np.zeros((self.test_len,self.gamma_len,5))
        self.mask = np.zeros((self.test_len,testRange*2))
        self.simData = np.zeros((self.test_len,dataNo,2*testRange))
        self.simFit = np.zeros((self.test_len,dataNo,2*testRange,3))
        self.simChi2 = np.zeros((self.test_len,dataNo,self.gamma_len))
        self.simGammaBd = np.zeros((self.test_len,dataNo))
        self.getResult(result)

    def getResult(self, result):
        for i in range(0,self.test_len):
            self.chi2[i,:], self.gammaBd[i], self.modelFit[i,:,:], self.beta[i,:,:], self.mask[i,:], self.gammaBand[i,:], self.simGammaBd[i,:], self.simData[i,:,:], self.simFit[i,:,:,:], self.simChi2[i,:,:] = result[i]
        self.N = detection_signif(self.chi2)


##########Results##########
def writeParams(fileName):
    out = open(fileName, "w")
    out.write('#Parameters:\n')
    out.write('#rho_s, rs, R_sun, sigma_v, solarV, testRange, anchorNo, step, maskRange, dataNo\n')
    out.write(str(rho_s)+'\n')
    out.write(str(rs)+'\n')
    out.write(str(R_sun)+'\n')
    out.write(str(sigma_v)+'\n')
    out.write(str(solarV)+'\n')
    out.write(str(testRange)+'\n')
    out.write(str(anchorNo)+'\n')
    out.write(str(step)+'\n')
    out.write(str(maskRange)+'\n')
    out.write(str(dataNo)+'\n')
    out.close()
