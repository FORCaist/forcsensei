import ipywidgets as widgets
from ipywidgets import VBox, HBox
import codecs as cd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import forcsensei.utils as ut
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#### PREPROCESSING OPTIONS ####
def options(X):
    style = {'description_width': 'initial'} #general style settings

    ### Define sample properties ###
    fn = X['fn']
    prop_title = widgets.HTML(value='<h3>Sample preprocessing options</h3>')
    mass_title = widgets.HTML(value='To disable mass normalization use a value of -1')

    sample, unit, mass = ut.sample_details(X)

    sample_widge = widgets.Text(value=sample,description='Sample name:',style=style)
    
    if mass == "N/A":
        mass_widge = widgets.FloatText(value=-1, description = 'Sample mass (g):',style=style)
    else:
        mass_widge = widgets.FloatText(value=mass, description = 'Sample mass (g):',style=style)

    mass_widge1 = HBox([mass_widge,mass_title])
    
    ### Define measurement corrections ###
    correct_title = widgets.HTML(value='<h3>Select preprocessing options:</h3>')
    
    slope_widge = widgets.FloatSlider(
        value=70,
        min=1,
        max=100.0,
        step=1,
        description='Slope correction [%]:',
        style=style,
        readout_format='.0f',
    )
    
    slope_title = widgets.HTML(value='To disable high-field slope correction use a value of 100%')
    slope_widge1 = HBox([slope_widge,slope_title])
    
    drift_widge = widgets.Checkbox(value=False, description='Measurement drift correction')
    fpa_widge = widgets.Checkbox(value=False, description='Remove first point artifact')
    lpa_widge = widgets.Checkbox(value=False, description='Remove last point artifact')
    correct_widge = VBox([correct_title,sample_widge,mass_widge1,slope_widge1,drift_widge,fpa_widge,lpa_widge])

    preprocess_nest = widgets.Tab()
    preprocess_nest.children = [correct_widge]
    preprocess_nest.set_title(0, 'PREPROCESSING')
    display(preprocess_nest)

    X["sample"] = sample_widge
    X["mass"] = mass_widge
    X["unit"] = unit
    X["drift"] = drift_widge
    X["slope"] = slope_widge
    X["fpa"] = fpa_widge
    X["lpa"] = lpa_widge
    
    return X

#### PREPROCESSING COMMAND ####
def execute(X):
  
    #parse measurements
    H, Hr, M, Fk, Fj, Ft, dH = ut.parse_measurements(X)
    Hcal, Mcal, tcal = ut.parse_calibration(X)
    Hc1, Hc2, Hb1, Hb2 = ut.measurement_limts(X)
    
    # make a data dictionary for passing large numbers of arguments
    # should unpack in functions for consistency
    X["H"] = H
    X["Hr"] = Hr
    X["M"] = M
    X["dH"] = dH
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["Ft"] = Ft
    X["Hcal"] = Hcal
    X["Mcal"] = Mcal
    X["tcal"] = tcal
    X["Hc1"] = Hc1
    X["Hc2"] = Hc2
    X["Hb1"] = Hb1
    X["Hb2"] = Hb2

    if X['unit']=='Cgs':
        X = ut.CGS2SI(X)
    
    if X["drift"].value == True:
        X = drift_correction(X)   
  
    if X["slope"].value < 100:
        X = slope_correction(X)
  
    if X["fpa"].value == True:
        X = remove_fpa(X)
    
    if X["lpa"].value == True:
        X = remove_lpa(X)
    

    #find when loop is open
    X = find_Hopen(X)

    #extend FORCs
    #X = FORC_extend2(X)

    #perform lower branch subtraction
    X = lowerbranch_subtract(X)    

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(121)
    X = plot_hysteresis(X,ax1)
    ax2 = fig.add_subplot(122)
    X = plot_delta_hysteresis(X,ax2)
    
    #pl.dump(fig,open('hysteresis.pickle','wb'))
    #pl.dump(ax1,open('hysteresis.pickle','wb'))

    outputfile = X["sample"].value+'_HYS.pdf'    
    plt.savefig(outputfile, bbox_inches="tight")
    plt.show()
    
    X['hys_fig'] = fig

    return X

#### PREPROCESSING ROUTINES ####

def find_Hopen(X,dMlim=0.05):
    
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    Hmin = np.min(H)
    Hmax = np.max(H)

    Nbar = 10
    nH = int((Hmax - Hmin)/dH)
    Hi = np.linspace(Hmin,Hmax,nH*50+1)
    Mi = np.empty(Hi.size)
    
    #perform basic loess
    for i in range(Hi.size):
        idx = (H>=Hi[i]-2.5*dH) & (H<=Hi[i]+2.5*dH)
        Mbar = M[idx]
        Hbar = H[idx]
        Fbar = Fk[idx]
        F0 = np.sort(np.unique(Fbar))
        if F0.size>Nbar:
            F0=F0[-Nbar]
        else:
            F0=np.min(F0)
        idx = Fbar>=F0
        
        p = np.polyfit(Hbar[idx],Mbar[idx],2)
        Mi[i] = np.polyval(p,Hi[i])
    
    Hlower = Hi
    Mlower = Mi

    idx0 = X['Fj']==1.0
    idx1 = X['Fk']==np.max(X['Fk'])
    
    lower = np.interp(X['H'][idx0],Hlower,Mlower,left=np.nan,right=np.nan)

    dM = X['M'][idx0]-lower
    dM0 = np.interp(0,np.flip(X['H'][idx0]),np.flip(dM),left=np.nan,right=np.nan)

    idx = (dM<=dMlim*dM0) & (X['H'][idx0]<0)
    Hopen = -np.max(X['H'][idx0][idx])


    X['Hopen'] = Hopen
    
    return X

def remove_lpa(X):
    
    #unpack
    Fj = X["Fj"]
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Ft = X["Ft"]
    
    #remove last point artifact
    Nforc = int(np.max(Fk))
    W = np.ones(Fk.size)
    
    for i in range(Nforc):      
        Fj_max=np.sum((Fk==i))
        idx = ((Fk==i) & (Fj==Fj_max))
        W[idx]=0.0
    
    idx = (W > 0.5)
    H=H[idx]
    Hr=Hr[idx]
    M=M[idx]
    Fk=Fk[idx]
    Fj=Fj[idx]
    Ft=Ft[idx]
    Fk=Fk-np.min(Fk)+1. #reset FORC number if required
    
    #repack
    X["Fj"] = Fj
    X["H"] = H   
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft        
    
    return X

def remove_fpa(X):
    
    #unpack
    Fj = X["Fj"]
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    Ft = X["Ft"]
    
    #remove first point artifact
    idx=((Fj<=1.0))
    H=H[~idx]
    Hr=Hr[~idx]
    M=M[~idx]
    Fk=Fk[~idx]
    Fj=Fj[~idx]
    Ft=Ft[~idx]
    Fk=Fk-np.min(Fk)+1. #reset FORC number if required
    Fj=Fj-1.
    
    #repack
    X["Fj"] = Fj
    X["H"] = H   
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft        
    
    return X

def drift_correction(X):
  
    #unpack
    M = X["M"]
    Mcal = np.squeeze(X["Mcal"])    
    Ft = np.squeeze(X["Ft"])
    tcal = np.squeeze(X["tcal"])
  
    #perform drift correction
    M=M*Mcal[0]/np.interp(Ft,tcal,Mcal,left=np.nan) #drift correction
  
    #repack
    X["M"] = M
  
    return X

def FORC_extend2(X):
    
    Ne = 5 #extend up to 20 measurement points backwards
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    for i in range(int(X['Fk'][-1])):
        M0 = M[Fk==i+1]
        H0 = H[Fk==i+1]
        Hr0 = Hr[Fk==i+1][0]

        H1 = np.flip(2*H0[0]-H0[1:])
        M1 = np.flip(2*M0[0]-M0[1:])
        #M1 = M0[0] - (np.flip(M0)[1:]-M0[0])
        #H1 = H0[0] - (np.flip(H0)[1:]-H0[0])


        if M1.size>Ne:
            H1 = H1[-Ne-1:-1]
            M1 = M1[-Ne-1:-1]
        
        if i==0:    
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H1,H0))
            M_new = np.concatenate((M1,M0))
            Hr_new = np.ones(N_new)*Hr0
            Fk_new = np.ones(N_new)
            Fj_new = np.arange(N_new)+1-M1.size
        else:
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H_new,H1,H0))
            M_new = np.concatenate((M_new,M1,M0))
            Hr_new = np.concatenate((Hr_new,np.ones(N_new)*Hr0))
            Fk_new = np.concatenate((Fk_new,np.ones(N_new)+i))
            Fj_new = np.concatenate((Fj_new,np.arange(N_new)+1-M1.size))
            
    #pack up variables
    X['H'] = H_new
    X['Hr'] = Hr_new
    X['M'] = M_new
    X['Fk'] = Fk_new
    X['Fj'] = Fj_new
    
    return X

def FORC_extend(X):
    
    Ne = 20 #extend up to 20 measurement points backwards
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    for i in range(int(X['Fk'][-1])):
        M0 = M[Fk==i+1]
        H0 = H[Fk==i+1]
        Hr0 = Hr[Fk==i+1][0]

        M1 = M0[0] - (np.flip(M0)[1:]-M0[0])
        H1 = H0[0] - (np.flip(H0)[1:]-H0[0])


        if M1.size>Ne:
            H1 = H1[-Ne-1:-1]
            M1 = M1[-Ne-1:-1]
        
        if i==0:    
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H1,H0))
            M_new = np.concatenate((M1,M0))
            Hr_new = np.ones(N_new)*Hr0
            Fk_new = np.ones(N_new)
            Fj_new = np.arange(N_new)+1-M1.size
        else:
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H_new,H1,H0))
            M_new = np.concatenate((M_new,M1,M0))
            Hr_new = np.concatenate((Hr_new,np.ones(N_new)*Hr0))
            Fk_new = np.concatenate((Fk_new,np.ones(N_new)+i))
            Fj_new = np.concatenate((Fj_new,np.arange(N_new)+1-M1.size))
            
    #pack up variables
    X['H'] = H_new
    X['Hr'] = Hr_new
    X['M'] = M_new
    X['Fk'] = Fk_new
    X['Fj'] = Fj_new
    
    return X

def lowerbranch_subtract(X):
    """Function to subtract lower hysteresis branch from FORC magnetizations
    
    Inputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    
    Outputs:
    M: lower branch subtracted magnetization [float, SI units]
   
    
    """
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    Hmin = np.min(H)
    Hmax = np.max(H)


    Nbar = 10
    nH = int((Hmax - Hmin)/dH)
    Hi = np.linspace(Hmin,Hmax,nH*50+1)
    Mi = np.empty(Hi.size)
    
    #perform basic loess
    for i in range(Hi.size):
        idx = (H>=Hi[i]-2.5*dH) & (H<=Hi[i]+2.5*dH)
        Mbar = M[idx]
        Hbar = H[idx]
        Fbar = Fk[idx]
        F0 = np.sort(np.unique(Fbar))
        if F0.size>Nbar:
            F0=F0[-Nbar]
        else:
            F0=np.min(F0)
        idx = Fbar>=F0
        
        p = np.polyfit(Hbar[idx],Mbar[idx],2)
        Mi[i] = np.polyval(p,Hi[i])
    
    Hlower = Hi
    Mlower = Mi
    Mcorr=M-np.interp(H,Hlower,Mlower,left=np.nan,right=np.nan) #subtracted lower branch from FORCs via interpolation

    Fk=Fk[~np.isnan(Mcorr)] #remove any nan
    Fj=Fj[~np.isnan(Mcorr)] #remove any nan
    H=H[~np.isnan(Mcorr)] #remove any nan
    Hr=Hr[~np.isnan(Mcorr)] #remove any nan
    M=M[~np.isnan(Mcorr)] #remove any nan
    Mcorr = Mcorr[~np.isnan(Mcorr)] #remove any nan
    
    #repack
    X["H"] = H    
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["DM"] = Mcorr
    #X["Hlower"] = Hi
    #X["Mlower"] = Mi
    
    return X

    ###### HELPER FUNCTIONS TO READ FROM FILE

def slope_correction(X):
  
    #unpack
    H = X["H"]
    M = X["M"]
  
    # high field slope correction
    Hidx = H > (X["slope"].value/100) * np.max(H)
    p = np.polyfit(H[Hidx],M[Hidx],1)
    M = M - H*p[0]
  
    #repack
    X["M"]=M
  
    return X

#### PLOTTING ROUTINES #####

def plot_hysteresis(X,ax):

  #unpack 
    M = X["M"]
    H = X["H"]
    Fk = X["Fk"]
    Fj = X["Fj"]

    #mpl.style.use('seaborn-whitegrid')
    hfont = {'fontname':'STIXGeneral'}

    for i in range(5,int(np.max(Fk)),5):
    
        if X["mass"].value > 0.0: #SI and mass normalized (T and Am2/kg)
            if X['unit']=='SI':    
                ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value/1000.0),'-k')
            elif X['unit']=='cgs':           
                ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value),'-k')        
        else: #SI not mass normalized (T and Am2)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)],'-k')
            ax.plot(H[(Fk==i) & (Fj<=0)],M[(Fk==i) & (Fj<=0)],'-r')


    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=14,color='k')
    ax.tick_params(axis='both',which='minor',direction='out',length=5,width=1,color='k')

    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('k')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ylim=np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-ylim,ylim])
  
    #ax.set_ylim([-1,1])
    yticks0 = ax.get_yticks()
    yticks = yticks0[yticks0 != 0]
    ax.set_yticks(yticks)
  
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('k')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    xmax = np.max(np.abs(ax.get_xlim()))
    ax.set_xlim([-xmax,xmax])

    #label x-axis
    if X['unit']=='SI':
        ax.set_xlabel(r'B [T]',horizontalalignment='right', position=(1,25), fontsize=14)
    elif X['unit']=='cgs':
        ax.set_xlabel(r'H [Oe]',horizontalalignment='right', position=(1,25), fontsize=14)

    #label y-axis according to unit system
    if X["mass"].value > 0.0:
        if X['unit']=='SI':    
            ax.set_ylabel(r'M [Am$^2$/kg]',verticalalignment='top', position=(25,0.9), labelpad=20, fontsize=14,**hfont)
        elif X['unit']=='cgs':
            ax.set_ylabel(r'M [emu/g]',verticalalignment='top', position=(25,0.9), labelpad=20, fontsize=14,**hfont)
    else: 
        if X['unit']=='SI':    
            ax.set_ylabel(r'M [Am$^2$]',verticalalignment='top', position=(25,0.9), labelpad=20, fontsize=14,**hfont)
        elif X['unit']=='cgs':
            ax.set_ylabel(r'M [emu]',verticalalignment='top', position=(25,0.9), labelpad=20, fontsize=14,**hfont)

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    X["xmax"]=xmax
    X["hys_ax"] = ax
    return X

def plot_delta_hysteresis(X,ax):

    #unpack 
    M = X["DM"]
    H = X["H"]
    Fk = X["Fk"]
    Fj = X["Fj"]

    hfont = {'fontname':'STIXGeneral'}

    for i in range(5,int(np.max(Fk)),5):
    
        if X["mass"].value > 0.0: #SI and mass normalized (T and Am2/kg)
            if X['unit'] == 'SI':
                ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value/1000.0),'-k')
            elif X['unit'] == 'cgs':
                ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value),'-k')        
        else: #SI not mass normalized (T and Am2)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)],'-k')
            ax.plot(H[(Fk==i) & (Fj<=0)],M[(Fk==i) & (Fj<=0)],'-r')
 
      
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=14,color='k')
    ax.tick_params(axis='both',which='minor',direction='out',length=5,width=1,color='k')

    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('k')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
  
    ylim=np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-ylim*0.1,ylim])
    yticks0 = ax.get_yticks()
    yticks = yticks0[yticks0 != 0]
    ax.set_yticks(yticks)
  
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('k')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    #Xticks = ax.get_xticks()
    #Xidx = np.argwhere(np.abs(Xticks)>0.01)
    #ax.set_xticks(Xticks[Xidx])
    ax.set_xticks(X['hys_ax'].get_xticks())

    #xmax = X["xmax"]
    #ax.set_xlim([-xmax,xmax])
    ax.set_xlim(X['hys_ax'].get_xlim())

    #label x-axis
    if X['unit']=='SI':
        ax.set_xlabel(r'B [T]',horizontalalignment='right', position=(1,25), fontsize=14)
    elif X['unit']=='cgs':
        ax.set_xlabel(r'H [Oe]',horizontalalignment='right', position=(1,25), fontsize=14)

    #label y-axis according to unit system
    if X["mass"].value > 0.0:
        if X['unit']=='SI':    
            ax.set_ylabel(r'M [Am$^2$/kg]',verticalalignment='top', position=(25,0.9), labelpad=20, fontsize=14,**hfont)
        elif X['unit']=='cgs':
            ax.set_ylabel(r'M [emu/g]',verticalalignment='top', position=(25,0.9), labelpad=20, fontsize=14,**hfont)
    else: 
        if X['unit']=='SI':    
            ax.set_ylabel(r'M [Am$^2$]',verticalalignment='top', position=(25,0.9), labelpad=20, fontsize=14,**hfont)
        elif X['unit']=='cgs':
            ax.set_ylabel(r'M [emu]',verticalalignment='top', position=(25,0.9), labelpad=20, fontsize=14,**hfont)

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    
    #X["delta_ax"] = ax

    return X

