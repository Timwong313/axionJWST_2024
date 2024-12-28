#####Package#####
import numpy as np
import os
import multiprocessing as mp 
import time
import MRS_func as f

if __name__ == '__main__':
    #####Input#####
    gammaTest1_consv = np.logspace(np.log10(5e-25),-23,500)
    gammaTest2_consv = np.logspace(np.log10(5e-25),-23,500)
    gammaTest3_consv = np.logspace(np.log10(5e-25),-23,500)
    gammaTest4_consv = np.logspace(np.log10(3e-24),-23,1000)
    #gammaTest1_cont = np.append(0,np.logspace(np.log10(1e-27),np.log10(1e-23),200))
    #gammaTest2_cont = np.append(0,np.logspace(np.log10(1e-27),np.log10(1e-23),200))
    #gammaTest3_cont = np.append(0,np.logspace(np.log10(8e-28),np.log10(8e-23),200))
    #gammaTest4_cont = np.append(0,np.logspace(-27,np.log10(8e-23),200))
    gammaTest1_cont = f.quadgrowth(0,1e-24,200)
    gammaTest2_cont = f.quadgrowth(0,1e-24,200)
    gammaTest3_cont = f.quadgrowth(0,5e-23,200)
    gammaTest4_cont = f.quadgrowth(0,5e-23,200)

    #####Data#####
    dataDir = '../analysis_data/testSysErr(4)/sys025'
    stkData = np.load(dataDir+'/stkData.npz')
    metaData = np.load(dataDir+'/metaData.npz')
    wvl_ch1, flux_ch1, err_ch1 = stkData['ch1']
    wvl_ch2, flux_ch2, err_ch2 = stkData['ch2']
    wvl_ch3, flux_ch3, err_ch3 = stkData['ch3']
    wvl_ch4, flux_ch4, err_ch4 = stkData['ch4']
    exptArr = metaData['expt']
    longArr = metaData['l']
    latArr = metaData['b']
    exptArr_ch1 = exptArr[:,0]
    exptArr_ch2 = exptArr[:,1]
    exptArr_ch3 = exptArr[:,2]
    exptArr_ch4 = exptArr[:,3]
    
    #####Initialization#####
    S_spec_ch1 = np.array([4.9,5.74,3320,3710])
    M_spec_ch1 = np.array([5.66,6.63,3190,3750])
    L_spec_ch1 = np.array([6.53,7.65,3100,3610])
    S_spec_ch2 = np.array([7.51,8.77,2990,3110])
    M_spec_ch2 = np.array([8.67,10.13,2750,3170])
    L_spec_ch2 = np.array([10.02,11.70,2860,3300])
    S_spec_ch3 = np.array([11.55,13.47,2530,2880])
    M_spec_ch3 = np.array([13.34,15.57,1790,2640])
    L_spec_ch3 = np.array([15.41,17.98,1980,2790])
    S_spec_ch4 = np.array([17.70,20.95,1460,1930])
    M_spec_ch4 = np.array([20.69,24.48,1680,1770])
    L_spec_ch4 = np.array([24.19,27.90,1630,1330])
    d_lmd_ch1 = f.FWHMestim(S_spec_ch1,M_spec_ch1,L_spec_ch1)  #um
    d_lmd_ch2 = f.FWHMestim(S_spec_ch2,M_spec_ch2,L_spec_ch2)  #um
    d_lmd_ch3 = f.FWHMestim(S_spec_ch3,M_spec_ch3,L_spec_ch3)  #um
    d_lmd_ch4 = f.FWHMestim(S_spec_ch4,M_spec_ch4,L_spec_ch4)  #um
    LM = f.line_model(latArr,longArr)  #Initialize the line_model class
    spectrum1_eff = LM.effSpectrum(wvl_ch1,exptArr_ch1,d_lmd_ch1)
    spectrum2_eff = LM.effSpectrum(wvl_ch2,exptArr_ch2,d_lmd_ch2)
    spectrum3_eff = LM.effSpectrum(wvl_ch3,exptArr_ch3,d_lmd_ch3)
    spectrum4_eff = LM.effSpectrum(wvl_ch4,exptArr_ch4,d_lmd_ch4)
    
    #####ConservConstraint#####
    consvChi2_ch1, consvGammaBd_ch1, wvl_bd_ch1 = f.chi2_conservative(wvl_ch1,flux_ch1,err_ch1,gammaTest1_consv,spectrum1_eff)
    consvChi2_ch2, consvGammaBd_ch2, wvl_bd_ch2 = f.chi2_conservative(wvl_ch2,flux_ch2,err_ch2,gammaTest2_consv,spectrum2_eff)
    consvChi2_ch3, consvGammaBd_ch3, wvl_bd_ch3 = f.chi2_conservative(wvl_ch3,flux_ch3,err_ch3,gammaTest3_consv,spectrum3_eff)
    consvChi2_ch4, consvGammaBd_ch4, wvl_bd_ch4 = f.chi2_conservative(wvl_ch4,flux_ch4,err_ch4,gammaTest4_consv,spectrum4_eff)
    massArr_bd_ch1 = f.wvlToMass(wvl_bd_ch1)
    massArr_bd_ch2 = f.wvlToMass(wvl_bd_ch2)
    massArr_bd_ch3 = f.wvlToMass(wvl_bd_ch3)
    massArr_bd_ch4 = f.wvlToMass(wvl_bd_ch4)
    consvCouplingBd_ch1 = f.gammaToCoupling(consvGammaBd_ch1,massArr_bd_ch1)
    consvCouplingBd_ch2 = f.gammaToCoupling(consvGammaBd_ch2,massArr_bd_ch2)
    consvCouplingBd_ch3 = f.gammaToCoupling(consvGammaBd_ch3,massArr_bd_ch3)
    consvCouplingBd_ch4 = f.gammaToCoupling(consvGammaBd_ch4,massArr_bd_ch4)
    consvResult_ch1 = np.array([wvl_bd_ch1,consvGammaBd_ch1,consvCouplingBd_ch1])
    consvResult_ch2 = np.array([wvl_bd_ch2,consvGammaBd_ch2,consvCouplingBd_ch2])
    consvResult_ch3 = np.array([wvl_bd_ch3,consvGammaBd_ch3,consvCouplingBd_ch3])
    consvResult_ch4 = np.array([wvl_bd_ch4,consvGammaBd_ch4,consvCouplingBd_ch4])
    
    #####ContinConstraint#####
    queue1 = mp.Manager().Queue()
    queue2 = mp.Manager().Queue()
    queue3 = mp.Manager().Queue()
    queue4 = mp.Manager().Queue()
    fc1 = f.continuumFitting(queue1,gammaTest1_cont,wvl_ch1,flux_ch1,err_ch1,spectrum1_eff)
    fc2 = f.continuumFitting(queue2,gammaTest2_cont,wvl_ch2,flux_ch2,err_ch2,spectrum2_eff)
    fc3 = f.continuumFitting(queue3,gammaTest3_cont,wvl_ch3,flux_ch3,err_ch3,spectrum3_eff)
    fc4 = f.continuumFitting(queue4,gammaTest4_cont,wvl_ch4,flux_ch4,err_ch4,spectrum4_eff)
    ts = time.time()
    fc1.start()
    fc2.start()
    fc3.start()
    fc4.start()
    fc1.join()
    fc2.join()
    fc3.join()
    fc4.join()
    print("Fitting done!")
    r1 = f.fittingResult(queue1.get())
    r2 = f.fittingResult(queue2.get())
    r3 = f.fittingResult(queue3.get())
    r4 = f.fittingResult(queue4.get())
    te = time.time()
    print('Duration:', (te-ts)/3600, '[hrs]')
    contCouplingBd_ch1 = f.gammaToCoupling(r1.gammaBd,massArr_bd_ch1)
    contCouplingBd_ch2 = f.gammaToCoupling(r2.gammaBd,massArr_bd_ch2)
    contCouplingBd_ch3 = f.gammaToCoupling(r3.gammaBd,massArr_bd_ch3)
    contCouplingBd_ch4 = f.gammaToCoupling(r4.gammaBd,massArr_bd_ch4)
    contResult_ch1 = np.array([fc1.wvl_bd,r1.gammaBd,contCouplingBd_ch1,r1.N])
    contResult_ch2 = np.array([fc2.wvl_bd,r2.gammaBd,contCouplingBd_ch2,r2.N])
    contResult_ch3 = np.array([fc3.wvl_bd,r3.gammaBd,contCouplingBd_ch3,r3.N])
    contResult_ch4 = np.array([fc4.wvl_bd,r4.gammaBd,contCouplingBd_ch4,r4.N])
    
    #####SaveResults#####
    nameDir = dataDir+'/result'
    i = 1
    while os.path.isdir(nameDir+str(i)):
        i += 1
    resultDir = nameDir+str(i)
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)
    f.writeParams(resultDir+'/params.txt')
    np.savez(resultDir+'/consvResult.npz',ch1=consvResult_ch1,ch2=consvResult_ch2,ch3=consvResult_ch3,ch4=consvResult_ch4)
    np.savez(resultDir+'/contResult.npz',ch1=contResult_ch1,ch2=contResult_ch2,ch3=contResult_ch3,ch4=contResult_ch4)
    np.savez(resultDir+'/consvChi2.npz',ch1=consvChi2_ch1,ch2=consvChi2_ch2,ch3=consvChi2_ch3,ch4=consvChi2_ch4)
    np.savez(resultDir+'/gammaTest_consv.npz',ch1=gammaTest1_consv,ch2=gammaTest2_consv,ch3=gammaTest3_consv,ch4=gammaTest4_consv)
    np.savez(resultDir+'/contChi2.npz',ch1=r1.chi2,ch2=r2.chi2,ch3=r3.chi2,ch4=r4.chi2)
    np.savez(resultDir+'/gammaTest_cont.npz',ch1=gammaTest1_cont,ch2=gammaTest2_cont,ch3=gammaTest3_cont,ch4=gammaTest4_cont)
    np.savez(resultDir+'/modelFit.npz',ch1=r1.modelFit,ch2=r2.modelFit,ch3=r3.modelFit,ch4=r4.modelFit)
    np.savez(resultDir+'/gammaBand.npz',ch1=r1.gammaBand,ch2=r2.gammaBand,ch3=r3.gammaBand,ch4=r4.gammaBand)
    np.savez(resultDir+'/simData.npz',ch1=r1.simData,ch2=r2.simData,ch3=r3.simData,ch4=r4.simData)
    np.savez(resultDir+'/simFit.npz',ch1=r1.simFit,ch2=r2.simFit,ch3=r3.simFit,ch4=r4.simFit)
    np.savez(resultDir+'/simChi2.npz',ch1=r1.simChi2,ch2=r2.simChi2,ch3=r3.simChi2,ch4=r4.simChi2)
    np.savez(resultDir+'/simGammaBd.npz',ch1=r1.simGammaBd,ch2=r2.simGammaBd,ch3=r3.simGammaBd,ch4=r4.simGammaBd)
    np.savez(resultDir+'/mask.npz',ch1=r1.mask,ch2=r2.mask,ch3=r3.mask,ch4=r4.mask)
