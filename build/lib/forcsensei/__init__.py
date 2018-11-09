# FORCsensei module 
# give command OPENBLAS_NUM_THREADS=1 in terminal if using multithreading

import numpy as np
import codecs as cd
import scipy as sp
from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
from google.colab import files
mpl.rcParams['pdf.fonttype'] = 42
from termcolor import cprint

### NEW CODE
def sample_details(fn):

  sample = "."
  if len(fn.split(sample))>1:
    sample = sample.join(fn.split(sample)[:-1])
  else:
    sample = fn.split(sample)
  
  if type(sample) is list:
    sample=sample[0]
  
  units=parse_units(fn)
  mass=parse_mass(fn)
  
  return sample, units, mass

def load_file():
  
  uploaded = files.upload()
  for fn in uploaded.keys():
    print('User uploaded FORC file "{name}"'.format(name=fn))
  
  sample, units, mass = sample_details(fn)
  
  return sample, units, mass, fn

def check_pp(fn, pp):
  
  sample0, units0, mass0 = sample_details(fn)
  
  print('Preprocessing setting check')
  print('---------------------------')
  print(' ')
  
  status = 1
  # test: sample name
  if type(pp["sample name"]) is not str:
    status = -1
    cprint('Error: Sample name is not a character string','red')
  else:
    cprint('Sample name: '+pp["sample name"],'green')
  
  # test: sample mass
  if (pp["sample mass (g)"] == 'N/A'):
    cprint('Sample mass (g): '+ pp["sample mass (g)"],'green')
  elif (type(pp["sample mass (g)"]) is not float) or (pp["sample mass (g)"]<=0.0):
    status = -1
    cprint('Error: Sample mass is not a valid number','red')
  else:
    cprint('Sample mass (g): '+str(pp["sample mass (g)"]),'green')

  # test: units
  if (pp["units"] != 'SI') and (pp["units"] != 'Cgs'):
    status = -1
    cprint('Error: Units should be "SI" or "Cgs"','red')
  elif (pp["units"] == 'SI') and (units0 == 'Cgs'):
    status = 0
    cprint('Inconsistency: Units do not match those in the data file','blue')
  elif (pp["units"] == 'Cgs') and (units0 == 'SI'):
    status = 0
    cprint('Inconsistency: Units do not match those in the data file','blue')
  else:
    cprint('Units: '+str(pp["units"]),'green')

  # test: mass normalization
  if (pp["mass normalize"] != True) and (pp["mass normalize"] != False):
    status = -1
    cprint('Error: Mass normalization should be True or False','red')
  elif (pp["mass normalize"] is True) and (pp["sample mass (g)"] == 'N/A'):
    status = -1
    cprint('Error: Mass normalization requested, but no mass provided','red')
  elif (pp["mass normalize"] is True) and (pp["sample mass (g)"] != mass0) and (mass0 != 'N/A'):
    status = 0
    cprint('Inconsistency: Provided mass and data file mass are different','blue')
  else:
    cprint('Mass normalize: '+str(pp["mass normalize"]),'green')
  
  # test: drift correction
  if (pp["drift correction"] != True) and (pp["drift correction"] != False):
    status = -1
    cprint('Error: Drift correction should be True or False','red')
  else:
    cprint('Drift correction: '+str(pp["drift correction"]),'green')
  
  # test: high field slope correction
  if (pp["high field slope correction"] != True) and (pp["high field slope correction"] != False):
    status = -1
    cprint('Error: High field slope correction should be True or False','red')
  else:
    cprint('High field slope correction: '+str(pp["high field slope correction"]),'green')
  
  # test: first point artifact
  if (pp["first point artifact"] != True) and (pp["first point artifact"] != False):
    status = -1
    cprint('Error: First point artifact should be True or False','red')
  else:
    cprint('First point artifact: '+str(pp["first point artifact"]),'green')

  # test: replace outliers
  if (pp["replace outliers"] != True) and (pp["replace outliers"] != False):
    status = -1
    cprint('Error: Replace outliers should be True or False','red')
  else:
    cprint('Replace outliers: '+str(pp["replace outliers"]),'green')
    
  # test: subtract lower branch
  if (pp["subtract lower branch"] != True) and (pp["subtract lower branch"] != False):
    status = -1
    cprint('Error: Subtract lower branch should be True or False','red')
  else:
    cprint('Subtract lower branch: '+str(pp["subtract lower branch"]),'green')

   # test: plots
  if (pp["plots"] != True) and (pp["plots"] != False):
    status = -1
    cprint('Error: Plots option should be True or False','red')
  else:
    cprint('Plots: '+str(pp["plots"]),'green')
  
  # test: save plots
  if (pp["save plots"] != True) and (pp["save plots"] != False):
    status = -1
    cprint('Error: Save plots option should be True or False','red')
  else:
    cprint('Save plots: '+str(pp["save plots"]),'green')  
  
  cprint(' ')
  
  if status == -1:  
    cprint('--------------------------------------------------------------','red')
    cprint('There are errors in your settings, your analysis will not run!','red')
    cprint('A video tutorial on these settings is provided above.','red')
    cprint('--------------------------------------------------------------','red')
  
  if status == 0:  
    cprint('-----------------------------------------------------------------------','blue')
    cprint('There are inconsistencies in your settings, but your analysis will run!','blue')
    cprint('A video tutorial on these settings is provided above.','blue')
    cprint('-----------------------------------------------------------------------','blue')
  
  if status == 1:  
    cprint('-------------------------------------------------------','green')
    cprint('There are no errors or inconsistencies in your settings','green')
    cprint('Your analysis is ready to run.','green')
    cprint('-------------------------------------------------------','green')


### DATA PREPROCESSING
def preprocessing(pp,fn):
  
  #parse measurements
  H, Hr, M, Fk, Fj, Ft, dH = parse_measurements(fn)
  Hcal, Mcal, tcal = parse_calibration(fn)
  
  # make a data dictionary for passing large numbers of arguments
  # should unpack in functions for consistency
  data = {
    "H":H,
    "Hr": Hr,
    "M": M,
    "dH": dH,
    "Fk": Fk,
    "Fj": Fj,
    "Ft": Ft,
    "Hcal": Hcal,
    "Mcal": Mcal,
    "tcal": tcal  
  }
  
  if pp["drift correction"] == True:
    data = drift_correction(data)   
  
  data = convert_units(pp,data,fn)
  
  if pp["mass normalize"] == True:
    data = mass_normalize(pp,data)
  
  if pp["high field slope correction"] == True:
    data = slope_correction(data)
  
  if pp["first point artifact"] == True:
    data = remove_fpa(data)
    
  if pp["last point artifact"] == True:
    data = remove_lpa(data)
    
  if pp["replace outliers"] == True:
    data = remove_outliers(data)
  
  if pp["subtract lower branch"] == True:
    data = lowerbranch_subtract(data)
  
  if pp["plots"] == True:
    plot_hysteresis(pp,data)
    if pp["subtract lower branch"] == True:
      plot_delta_hysteresis(pp,data)
    
  return data


# drift correction
def drift_correction(data):
  
  #unpack
  M = data["M"]
  Mcal = data["Mcal"]    
  Ft = data["Ft"]
  tcal = data["tcal"]
  
  #perform drift correction
  M=M*Mcal[0]/np.interp(Ft,tcal,Mcal,left=np.nan) #drift correction
  
  #repack
  data["M"] = M
  
  return data


def convert_units(pp,data,fn):
  
  _, units, _ = sample_details(fn)

  if (pp["units"] == 'SI') and (units == 'Cgs'): #convert to CGS
    H = data["H"]
    M = data["M"]
    
    H = H/1E4 #convert T into Oe
    M = M/1E3
      
    data["H"] = H
    data["M"] = M
    
  elif (pp["units"] == 'Cgs') and (units == 'SI'): #convert to CGS
    H = data["H"]
    M = data["M"]
    
    H = H*1E4 #convert Oe into T
    M = M*1E3
      
    data["H"] = H
    data["M"] = M
      
  return data

def mass_normalize(pp,data):
  
  M = data["M"]
    
  if (pp["units"] == 'SI'): 
    M = M / (pp["sample mass (g)"]/1000.) #convert to AM^2/kg
      
  if (pp["units"] == 'Cgs'): 
    M = M / pp["sample mass (g)"] #convert to emu/g
      
  data["M"] = M
      
  return data


# slope correction
def slope_correction(data):
  
  #unpack
  H = data["H"]
  M = data["M"]
  
  # high field slope correction
  Hidx = H > 0.8 * np.max(H)
  p = np.polyfit(H[Hidx],M[Hidx],1)
  M = M - H*p[0]
  
  #repack
  data["M"]=M
  
  return data

# remove FPA
def remove_fpa(data):
    
    #unpack
    Fj = data["Fj"]
    H = data["H"]    
    Hr = data["Hr"]
    M = data["M"]
    Fk = data["Fk"]
    Fj = data["Fj"]
    Ft = data["Ft"]
    
    #remove first point artifact
    idx=((Fj==1.0))
    H=H[~idx]
    Hr=Hr[~idx]
    M=M[~idx]
    Fk=Fk[~idx]
    Fj=Fj[~idx]
    Ft=Ft[~idx]
    Fk=Fk-np.min(Fk)+1. #reset FORC number if required
    Fj=Fj-1.
    
    #repack
    data["Fj"] = Fj
    data["H"] = H   
    data["Hr"] = Hr
    data["M"] = M
    data["Fk"] = Fk
    data["Fj"] = Fj
    data["Ft"] = Ft        
    
    return data
  
# remove lPA
def remove_lpa(data):
    
    #unpack
    Fj = data["Fj"]
    H = data["H"]    
    Hr = data["Hr"]
    M = data["M"]
    Fk = data["Fk"]
    Fj = data["Fj"]
    Ft = data["Ft"]
    
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
    data["Fj"] = Fj
    data["H"] = H   
    data["Hr"] = Hr
    data["M"] = M
    data["Fk"] = Fk
    data["Fj"] = Fj
    data["Ft"] = Ft        
    
    return data
  
def remove_outliers(data):
    
    """Function to replace "bad" measurements to zero.
    
    Inputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    
    Outputs:
    Fmask: mask, accepted points = 1, rejected points = 0
    R: residuals from fitting process
    Rcrit: critical residual threshold 
    
    """
    #unpack variables
    H = data["H"]    
    Hr = data["Hr"]
    M = data["M"]
    Fk = data["Fk"]
    Fj = data["Fj"]
    
    SF=2 #half width of the smooth (full width = 2SF+1)
    Mst=np.zeros(M.size)*np.nan #initialize output of smoothed magnetizations
    for i in range(M.size): #loop through each measurement
        idx=((Fk==Fk[i]) & (Fj<=Fj[i]+SF) & (Fj>=Fj[i]-SF)) #finding smoothing window in terms of H
        Npts=np.sum(idx) #check enough points are available (may not be the case as edges)
        if Npts>3:
            #create centered quadratic design matrix WRT H
            A = np.concatenate((np.ones(Npts)[:,np.newaxis],\
                                (H[idx]-H[i])[:,np.newaxis],\
                                ((H[idx]-H[i])**2)[:,np.newaxis]),axis=1)
            Mst[i] = np.linalg.lstsq(A,M[idx],rcond=None)[0][0] #regression estimate of M
        else:
            Mst[i] = M[i] #not enough points, so used M

    Mstst=np.zeros(M.size)*np.nan
    for i in range(M.size):
        idx=((Fk<=Fk[i]+SF) & (Fk>=Fk[i]-SF)  & (Fk[i]-Fk+(Fj-Fj[i])==0))
        Npts=np.sum(idx)
        if Npts>3:
            #create centered quadratic design matrix WRT Hr
            A = np.concatenate((np.ones(Npts)[:,np.newaxis],\
                                (Hr[idx]-Hr[i])[:,np.newaxis],\
                                ((Hr[idx]-Hr[i])**2)[:,np.newaxis]),axis=1)
            Mstst[i] = np.linalg.lstsq(A,Mst[idx],rcond=None)[0][0] #regression estimate of Mst
        else: 
            Mstst[i] = Mst[i] #not enough points, so used Mst
            
    
    R = Mstst-Mst #estimated residuals
    Rcrit = np.std(R)*2.5 #set cut-off at 2.5 sigma
    Fmask=np.ones(M.size) #initialize mask
    Fmask[np.abs(R)>Rcrit]=0.0
    
    idx = (np.abs(R)<Rcrit) #points flagged as outliers
    
    #remove points deemed to be outliers
    H = H[idx]
    Hr = Hr[idx]
    M = M[idx]
    Fk = Fk[idx]
    Fj = Fj[idx]
      
    #reset indicies as required
    Fk = Fk - np.min(Fk)+1
  
    Nforc = int(np.max(Fk))
    for i in range(Nforc):
      idx = (Fk == i)
      idx0 = np.argsort(Fj[idx])
      for i in range(idx.size):
        Fj[idx[idx0[i]]] = i+1
    
    #repack variables
    H = data["H"]    
    Hr = data["Hr"]
    M = data["M"]
    Fk = data["Fk"]
    Fj = data["Fj"]
    
    
    
    return data
  
def lowerbranch_subtract(data):
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
    H = data["H"]    
    Hr = data["Hr"]
    M = data["M"]
    Fk = data["Fk"]
    Fj = data["Fj"]
    
    idx=(Fj==1) #define the upper branch based on the 1st measurement point in each FORC
    Hupper=-H[idx] #upper branch applied field (minus is to convert to lower branch)
    Mupper=-M[idx] #upper branch magnetization (minus is to convert to lower branch)

    idx=(Fk==np.max(Fk)) #use the last FORC to represent the lower branch
    Hlower=H[idx] #lower branch applied field
    Mlower=M[idx] #lower branch magnetization

    #adjust offset between upper and lower branch if required
    Mtest=Mupper-np.interp(Hlower[-1],Hupper,Mupper,left=np.nan,right=np.nan)+Mlower[-1]
    idx=Hupper>Hlower[-1]
    Hupper=Hupper[idx]
    Mupper=Mtest[idx]

    Hlower=np.concatenate((Hlower,Hupper)) #combine fields
    Mlower=np.concatenate((Mlower,Mupper)) #correct upper offset and combine magnetizations
    
    Mcorr=M-np.interp(H,Hlower,Mlower,left=np.nan,right=np.nan) #subtracted lower branch from FORCs via interpolation

    Fk=Fk[~np.isnan(Mcorr)] #remove any nan
    Fj=Fj[~np.isnan(Mcorr)] #remove any nan
    H=H[~np.isnan(Mcorr)] #remove any nan
    Hr=Hr[~np.isnan(Mcorr)] #remove any nan
    M=M[~np.isnan(Mcorr)] #remove any nan
    Mcorr = Mcorr[~np.isnan(Mcorr)] #remove any nan
    
    #repack
    data["H"] = H    
    data["Hr"] = Hr
    data["M"] = M
    data["Fk"] = Fk
    data["Fj"] = Fj
    data["DM"] = Mcorr
    
    
    return data
  
def plot_hysteresis(pp,data):

  #unpack 
  sample = pp["sample name"]
  M = data["M"]
  H = data["H"]
  Fk = data["Fk"]

  mpl.style.use('seaborn-whitegrid')
  hfont = {'fontname':'STIXGeneral'}

  fig, ax = plt.subplots(figsize=(8,8))

  for i in range(5,int(np.max(Fk)),7):
    
    if pp["units"] == "Cgs":
      ax.plot(H[Fk==i],M[Fk==i],'-k')
    else:
      ax.plot(H[Fk==i]*1000,M[Fk==i],'-k')

  ax.grid(False)
  ax.minorticks_on()
  ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=12,color='k')
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
  #Xticks = ax.get_xticks()
  #Xidx = np.argwhere(np.abs(Xticks)>0.01)
  #ax.set_xticks(Xticks[Xidx])

  #label x-axis according to unit system
  if pp["units"]=="Cgs":
    ax.set_xlabel('Oe [mT]',horizontalalignment='right', position=(1,25), fontsize=12)
  else:
    ax.set_xlabel('B [mT]',horizontalalignment='right', position=(1,25), fontsize=12)

  #label y-axis according to unit system
  if ((pp["units"]=="SI") and (pp["mass normalize"] == True)):
    ax.set_ylabel('M [Am2/kg]',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
  elif ((pp["units"]=="SI") and (pp["mass normalize"] == False)): 
    ax.set_ylabel('M [Am2]',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
  elif ((pp["units"]=="Cgs") and (pp["mass normalize"] == True)): 
    ax.set_ylabel('M [emu/g]',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
  elif ((pp["units"]=="Cgs") and (pp["mass normalize"] == False)): 
    ax.set_ylabel('M [emu]',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)

  if pp["save plots"] == True:
      plt.savefig(sample+'_hys.pdf',bbox_inches="tight")
      files.download(sample+'_hys.pdf')

  plt.show()
  
  
def plot_delta_hysteresis(pp,data):

  #unpack 
  sample = pp["sample name"]
  M = data["DM"]
  H = data["H"]
  Fk = data["Fk"]

  mpl.style.use('seaborn-whitegrid')
  hfont = {'fontname':'STIXGeneral'}

  fig, ax = plt.subplots(figsize=(8,8))

  for i in range(5,int(np.max(Fk)),7):
    
    if pp["units"] == "Cgs":
      ax.plot(H[Fk==i],M[Fk==i],'-k')
    else:
      ax.plot(H[Fk==i]*1000,M[Fk==i],'-k')
      
  ax.grid(False)
  ax.minorticks_on()
  ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=12,color='k')
  ax.tick_params(axis='both',which='minor',direction='out',length=5,width=1,color='k')

  ax.spines['left'].set_position('zero')
  ax.spines['left'].set_color('k')

  # turn off the right spine/ticks
  ax.spines['right'].set_color('none')
  ax.yaxis.tick_left()
  #ax.set_ylabel('M / M$_0$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
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
  #ax.set_xlabel('B [mT]',horizontalalignment='right', position=(1,25), fontsize=12)
  xmax = np.max(np.abs(ax.get_xlim()))
  ax.set_xlim([-xmax,xmax])
  Xticks = ax.get_xticks()
  Xidx = np.argwhere(np.abs(Xticks)>0.01)
  ax.set_xticks(Xticks[Xidx])

   #label x-axis according to unit system
  if pp["units"]=="Cgs":
    ax.set_xlabel('Oe [mT]',horizontalalignment='right', position=(1,25), fontsize=12)
  else:
    ax.set_xlabel('B [mT]',horizontalalignment='right', position=(1,25), fontsize=12)

  #label y-axis according to unit system
  if ((pp["units"]=="SI") and (pp["mass normalize"] == True)):
    ax.set_ylabel('M - Mhys [Am2/kg]',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
  elif ((pp["units"]=="SI") and (pp["mass normalize"] == False)): 
    ax.set_ylabel('M - Mhys [Am2]',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
  elif ((pp["units"]=="Cgs") and (pp["mass normalize"] == True)): 
    ax.set_ylabel('M - Mhys [emu/g]',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
  elif ((pp["units"]=="Cgs") and (pp["mass normalize"] == False)): 
    ax.set_ylabel('M - Mhys [emu]',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
  
  if pp["save plots"] == True:
    plt.savefig(sample+'_delta.pdf',bbox_inches="tight")
    files.download(sample+'_delta.pdf')

  plt.show()

###

# define function which will look for lines in the header that start with certain strings
def find_data_lines(fp):
    """Helper function to identify measurement lines in a FORC data file.
    
    Given the various FORC file formats, measurements lines are considered to be those which:
    Start with a '+' or,
    Start with a '-' or,
    Are blank (i.e. lines between FORCs and calibration points) or,
    Contain a ','

    Inputs:
    fp: file identifier

    Outputs:
    line: string corresponding to data line that meets the above conditions
    """
    return [line for line in fp if ((line.startswith('+')) or (line.startswith('-')) or (line.strip()=='') or line.find(',')>-1.)]

#function to parse calibration points and provide time stamps
def lines_that_start_with(string, fp):
    """Helper function to lines in a FORC data file that start with a given string
    
    Inputs:
    string: string to compare lines to 
    fp: file identifier

    Outputs:
    line: string corresponding to data line that meets the above conditions
    """
    return [line for line in fp if line.startswith(string)]

def parse_header(file,string):
    """Function to extract instrument settings from FORC data file header
    
    Inputs:
    file: name of data file (string)    
    string: instrument setting to be extracted (string)

    Outputs:
    output: value of instrument setting [-1 if setting doesn't exist] (float)
    """
    output=-1 #default output (-1 corresponds to no result, i.e. setting doesn't exist)
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idx = line.find('=') #Some file formats may contain an '='
            if idx>-1.: #if '=' found
                output=float(line[idx+1:]) #value taken as everything to right of '='
            else: # '=' not found
                idx = len(string) #length of the setting string 
                output=float(line[idx+1:])  #value taken as everything to right of the setting name 

    return output

def parse_units(file):
    """Function to extract instrument unit settings ('') from FORC data file header
    
    Inputs:
    file: name of data file (string)    

    Outputs:
    CGS [Cgs setting] or SI [Hybrid SI] (string)
    """
    string = 'Units of measure' #header definition of units
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idxSI = line.find('Hybrid SI') #will return location if string is found, otherwise returns -1
            idxCGS = line.find('Cgs') #will return location if string is found, otherwise returns -1
    
    if idxSI>idxCGS: #determine which unit string was found in the headerline and output
        return 'SI'
    else:
        return 'CGS'

def parse_mass(file):
    """Function to extract sample from FORC data file header
    
    Inputs:
    file: name of data file (string)    

    Outputs:
    Mass in g or N/A
    """
    output = 'N/A'
    string = 'Mass' #header definition of units
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idx = line.find('=') #Some file formats may contain an '='
            if idx>-1.: #if '=' found
                output=(line[idx+1:]) #value taken as everything to right of '='
            else: # '=' not found
                idx = len(string) #length of the setting string 
                output=(line[idx+1:])  #value taken as everything to right of the setting name
        
            if output.find('N/A') > -1:
                output = 'N/A'
            else:
                output = float(output)

    return output

def calibration_times(file, Npts):
    """Function to estimate the time at which calibration points were measured in a FORC sequence
    
    Follows the procedure given in:
    R. Egli (2013) VARIFORC: An optimized protocol for calculating non-regular first-order reversal curve (FORC) diagrams. Global and Planetary Change, 110, 302-320, doi:10.1016/j.gloplacha.2013.08.003.

    Inputs:
    file: name of data file (string)    
    Npts: number of calibration points (int)

    Outputs:
    tcal_k: Estimated times at which the calibration points were measured (float)
    """    
    Units=parse_units(file) #determine measurement system (CGS or SI)

    string='PauseRvrsl' #Pause at reversal field (new file format, -1 if not available)
    tr0=parse_header(file,string)
    
    string='PauseNtl' #Pause at reversal field (old file format, -1 if not available)
    tr1=parse_header(file,string)

    tr=np.max((tr0,tr1)) #select Pause value depending on file format
    
    string='Averaging time' #Measurement averaging time 
    tau=parse_header(file,string)

    string='PauseCal' #Pause at calibration point
    tcal=parse_header(file,string)

    string='PauseSat' #Pause at saturation field
    ts=parse_header(file,string)

    string='SlewRate' #Field slewrate
    alpha=parse_header(file,string)

    string='HSat' #Satuation field
    Hs=parse_header(file,string)

    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(file,string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(file,string)

    string='Hc2' #upper Hc value for the FORC box (n.b. Hc1 is assumed to be 0)
    Hc2=parse_header(file,string)

    string='NForc' # Numer of measured FORCs (new file format, -1 if not available)
    N0=parse_header(file,string)

    string='NCrv'  # Numer of measured FORCs (old file format, -1 if not available)
    N1=parse_header(file,string)

    N=np.max((N0,N1)) #select Number of FORCs depending on file format

    if Units=='CGS':
        alpha=alpha/1E4 #convert from Oe to T
        Hs=Hs/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T
    
    dH = (Hc2-Hb1+Hb2)/N #estimated field spacing
    
    #now following Elgi's estimate of the measurement time
    nc2 = Hc2/dH
    Dt1 = tr + tau + tcal + ts + 2.*(Hs-Hb2-dH)/alpha
    Dt2 = tr + tau + (Hc2-Hb2-dH)/alpha

    Npts=int(Npts)
    tcal_k=np.zeros(Npts)
    
    for k in range(1,Npts+1):
        if k<=1+nc2:
            tcal_k[k-1]=k*Dt1-Dt2+dH/alpha*k**2+(tau-dH/alpha)*(k-1)**2
        else:
            tcal_k[k-1]=k*Dt1-Dt2+dH/alpha*k**2+(tau-dH/alpha)*((k-1)*(1+nc2)-nc2)

    return tcal_k

def measurement_times(file,Fk,Fj):
    """Function to estimate the time at which magnetization points were measured in a FORC sequence
    
    Follows the procedure given in:
    R. Egli (2013) VARIFORC: An optimized protocol for calculating non-regular first-order reversal curve (FORC) diagrams. Global and Planetary Change, 110, 302-320, doi:10.1016/j.gloplacha.2013.08.003.

    Inputs:
    file: name of data file (string)    
    Fk: FORC indicies (int)
    Fj: Measurement indicies within given FORC

    Outputs:
    Ft: Estimated times at which the magnetization points were measured (float)
    """    
    Units=parse_units(file) #determine measurement system (CGS or SI)

    string='PauseRvrsl' #Pause at reversal field (new file format, -1 if not available)
    tr0=parse_header(file,string)
    
    string='PauseNtl' #Pause at reversal field (old file format, -1 if not available)
    tr1=parse_header(file,string)

    tr=np.max((tr0,tr1)) #select Pause value depending on file format
    
    string='Averaging time' #Measurement averaging time 
    tau=parse_header(file,string)

    string='PauseCal' #Pause at calibration point
    tcal=parse_header(file,string)

    string='PauseSat' #Pause at saturation field
    ts=parse_header(file,string)

    string='SlewRate' #Field slewrate
    alpha=parse_header(file,string)

    string='HSat' #Satuation field
    Hs=parse_header(file,string)

    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(file,string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(file,string)

    string='Hc2' #upper Hc value for the FORC box (n.b. Hc1 is assumed to be 0)
    Hc2=parse_header(file,string)

    string='NForc' # Numer of measured FORCs (new file format, -1 if not available)
    N0=parse_header(file,string)

    string='NCrv'  # Numer of measured FORCs (old file format, -1 if not available)
    N1=parse_header(file,string)

    N=np.max((N0,N1)) #select Number of FORCs depending on file format

    if Units=='CGS':
        alpha=alpha/1E4 #convert from Oe to T
        Hs=Hs/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T

    dH = (Hc2-Hb1+Hb2)/N #estimated field spacing
    
    #now following Elgi's estimate of the measurement time
    nc2 = Hc2/dH

    Dt1 = tr + tau + tcal + ts + 2.*(Hs-Hb2-dH)/alpha
    Dt3 = Hb2/alpha

    Npts=int(Fk.size)
    Ft=np.zeros(Npts)
    
    for i in range(Npts):
        if Fk[i]<=1+nc2:
            Ft[i]=Fk[i]*Dt1+Dt3+Fj[i]*tau+dH/alpha*(Fk[i]*(Fk[i]-1))+(tau-dH/alpha)*(Fk[i]-1)**2
        else:
            Ft[i]=Fk[i]*Dt1+Dt3+Fj[i]*tau+dH/alpha*(Fk[i]*(Fk[i]-1))+(tau-dH/alpha)*((Fk[i]-1)*(1+nc2)-nc2)

    return Ft

def parse_calibration(file):
    """Function to extract measured calibration points from a FORC sequence
    
    Inputs:
    file: name of data file (string)    

    Outputs:
    Hcal: sequence of calibration fields [float, SI units]
    Mcal: sequence of calibration magnetizations [float, SI units]
    tcal: Estimated times at which the calibration points were measured (float, seconds)
    """ 

    dum=-9999.99 #dum value to indicate break in measurement seqence between FORCs and calibration points
    N0=int(1E6) #assume that any file will have less than 1E6 measurements
    H0=np.zeros(N0)*np.nan #initialize NaN array to contain field values
    M0=np.zeros(N0)*np.nan #initialize NaN array to contain magnetization values
    H0[0]=dum #first field entry is dummy value
    M0[0]=dum #first magnetization entry is dummy value 

    count=0 #counter to place values in arrays
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in find_data_lines(fp): #does the current line contain measurement data
            count=count+1 #increase counter
            idx = line.find(',') #no comma indicates a blank linw
            if idx>-1: #line contains a comma
                H0[count]=float(line[0:idx]) #assign field value (1st column)
                line=line[idx+1:] #remove the leading part of the line (only characters after the first comma remain)
                idx = line.find(',') #find next comman
                if idx>-1: #comma found in line
                    M0[count]=float(line[0:idx]) #read values up to next comma (assumes 2nd column is magnetizations)
                else: #comma wasn't found   
                    M0[count]=float(line) # magnetization value is just the remainder of the line 
            else:
                H0[count]=dum #line is blank, so fill with dummy value
                M0[count]=dum #line is blank, so fill with dummy value

    idx_start=np.argmax(H0!=dum) #find the first line that contains data            
    M0=M0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector           
    M0=M0[~np.isnan(M0)] #remove any NaNs at the end of the array
    H0=H0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector
    H0=H0[~np.isnan(H0)] #remove any NaNs at the end of the array

    ## now need to pull out the calibration points, will be after alternate -9999.99 entries
    idxSAT = np.array(np.where(np.isin(H0, dum))) #location of dummy values
    idxSAT = np.ndarray.squeeze(idxSAT) #squeeze into 1D
    idxSAT = idxSAT[0::2]+1 #every second index+1 should be calibration points

    Hcal=H0[idxSAT[0:-1]] #calibration fields
    Mcal=M0[idxSAT[0:-1]] #calibration magnetizations
    tcal=calibration_times(file,Hcal.size) #estimate the time of each calibratio measurement

    Units=parse_units(file)
    if Units=='CGS': #ensure SI units
        Hcal=Hcal/1E4 #convert from Oe to T
        Mcal=Mcal/1E3 #convert from emu to Am^2

    return Hcal, Mcal, tcal

def parse_measurements(file):
    """Function to extract measurement points from a FORC sequence
    
    Inputs:
    file: name of data file (string)    

    Outputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    Ft: Estimated times at which the points were measured (float, seconds)
    dH: Measurement field spacing [float SI units]
    """ 

    dum=-9999.99 #dum value to indicate break in measurement seqence between FORCs and calibration points
    N0=int(1E6) #assume that any file will have less than 1E6 measurements
    H0=np.zeros(N0)*np.nan #initialize NaN array to contain field values
    M0=np.zeros(N0)*np.nan #initialize NaN array to contain magnetization values
    H0[0]=dum #first field entry is dummy value
    M0[0]=dum #first magnetization entry is dummy value 

    count=0 #counter to place values in arrays
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in find_data_lines(fp): #does the current line contain measurement data
            count=count+1 #increase counter
            idx = line.find(',') #no comma indicates a blank linw
            if idx>-1: #line contains a comma
                H0[count]=float(line[0:idx]) #assign field value (1st column)
                line=line[idx+1:] #remove the leading part of the line (only characters after the first comma remain)
                idx = line.find(',') #find next comman
                if idx>-1: #comma found in line
                    M0[count]=float(line[0:idx]) #read values up to next comma (assumes 2nd column is magnetizations)
                else: #comma wasn't found   
                    M0[count]=float(line) # magnetization value is just the remainder of the line 
            else:
                H0[count]=dum #line is blank, so fill with dummy value
                M0[count]=dum #line is blank, so fill with dummy value

    idx_start=np.argmax(H0!=dum) #find the first line that contains data            
    M0=M0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector           
    M0=M0[~np.isnan(M0)] #remove any NaNs at the end of the array
    H0=H0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector
    H0=H0[~np.isnan(H0)] #remove any NaNs at the end of the array

    ## determine indicies of each FORC
    idxSAT = np.array(np.where(np.isin(H0, dum))) #find start address of each blank line
    idxSAT = np.ndarray.squeeze(idxSAT) #squeeze into 1D
    idxSTART = idxSAT[1::2]+1 #find start address of each FORC
    idxEND = idxSAT[2::2]-1 ##find end address of each FORC

    
    #Extract first FORC to initialize arrays 
    M=M0[idxSTART[0]:idxEND[0]+1] #Magnetization values
    H=H0[idxSTART[0]:idxEND[0]+1] #Field values
    Hr=np.ones(idxEND[0]+1-idxSTART[0])*H0[idxSTART[0]] #Reversal field values
    Fk=np.ones(idxEND[0]+1-idxSTART[0]) #index number of FORC
    Fj=np.arange(1,1+idxEND[0]+1-idxSTART[0])# measurement index within given FORC

    #Extract remaining FORCs one by one into into a long-vector
    for i in range(1,idxSTART.size):
        M=np.concatenate((M,M0[idxSTART[i]:idxEND[i]+1]))
        H=np.concatenate((H,H0[idxSTART[i]:idxEND[i]+1]))
        Hr=np.concatenate((Hr,np.ones(idxEND[i]+1-idxSTART[i])*H0[idxSTART[i]]))
        Fk=np.concatenate((Fk,np.ones(idxEND[i]+1-idxSTART[i])+i))
        Fj=np.concatenate((Fj,np.arange(1,1+idxEND[i]+1-idxSTART[i])))
    
    Units=parse_units(file) #Ensure use of SI units
    if Units=='CGS':
        H=H/1E4 #Convert Oe into T
        Hr=Hr/1E4 #Convert Oe into T
        M=M/1E3 #Convert emu to Am^2

    dH = np.mean(np.diff(H[Fk==np.max(Fk)])) #mean field spacing

    Ft=measurement_times(file,Fk,Fj) #estimated time of each measurement point

    return H, Hr, M, Fk, Fj, Ft, dH

def play_tutorial(index):
    
    #define list of tutorial videos
    tutorial = ['ilyS6K4ry3U'] #tutorial 1
    tutorial.append('6kCS_nJC72g') #tutorial 2
    tutorial.append('fZ0oB2wEBOI') #tutorial 3
        
    vid = YouTubeVideo(id = tutorial[index-1],autoplay=True)
    display(vid)