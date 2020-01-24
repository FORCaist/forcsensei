import pandas as pd
import codecs as cd
import numpy as np

#### NEW ROUTINES JAN2020 ####
def header_keywords(X):

    keywords = {}
    defaults = {}

    keywords['units'] = ('Units of measure ','Units of measure:')
    defaults['units'] = 'SI'
    
    keywords['mass'] = ('Mass:','Mass ','Mass           =')
    defaults['mass'] = 'N/A'
    
    keywords['pause at reversal'] = ('Pause at reversal fields:','PauseRvrsl','PauseNtl       =','PauseNtl ')
    defaults['pause at reversal'] = 'N/A'
    
    keywords['averaging time'] = ('Averaging time:','Averaging time ','Averaging time =')
    defaults['averaging time'] = 'N/A'
    
    keywords['pause at calibration'] = ('Pause at calibration field:','PauseCal','PauseCal       =')
    defaults['pause at calibration'] = 'N/A'
    
    keywords['pause at saturation'] = ('Pause at saturation field:','PauseSat','PauseSat       =')
    defaults['pause at saturation'] = 'N/A'
    
    keywords['field slewrate'] = ('SlewRate ','SlewRate       =')
    defaults['field slewrate'] = 1.0
    
    keywords['saturation field'] = ('Saturation field:','HSat           =','HSat')
    defaults['saturation field'] = 'N/A'
    
    keywords['Hb2'] = ('Max Hu field:','Hb2','Hb2            =')
    defaults['Hb2'] = 'N/A'
    
    keywords['Hb1'] = ('Min Hu field:','Hb1','Hb1            =')
    defaults['Hb1'] = 'N/A'
    
    keywords['Hc2'] = ('Max Hc field:','Hc2','Hc2            =')
    defaults['Hc2'] = 'N/A'
    
    keywords['Hc1'] = ('Min Hc field:','Hc1','Hc1            =')
    defaults['Hc1'] = 0.0
    
    keywords['number of FORCs'] = ('Number of FORCs:','NForc','NCrv')
    defaults['number of FORCs'] = 'N/A'
    
    keywords['number of rows'] = ('Number of data','NData')
    defaults['number of rows'] = 'N/A'

    keywords['calibration field'] = ('HCal','HCal           =')
    defaults['calibration field'] = 'N/A'
    
    X['keywords'] = keywords
    X['defaults'] = defaults
    
    return X


def line_num_for_phrase_in_file(phrase, filename):
    with cd.open(filename,'r',encoding='latin9') as f:
        for (i, line) in enumerate(f):
            if phrase in line:
                return i
    return False

def line_num_for_starting_data_line(filename):
    with cd.open(filename,'r',encoding='latin9') as f:
        for (i, line) in enumerate(f):
            if (line.startswith('+')) or (line.startswith('-')):
                return i
    return False

def line_for_phrase_in_file(phrase, filename):
    with cd.open(filename,'r',encoding='latin9') as f:
        for (i, line) in enumerate(f):
            if phrase in line:
                return line
    return False

def end_of_line_for_phrase_in_file(phrase, filename):
    with cd.open(filename,'r',encoding='latin9') as f:
        for (i, line) in enumerate(f):
            if phrase in line:
                line = line[len(phrase):]
                return line.strip()
    return -1

def parse_digits(string,keywords0):
    
    string = string[len(keywords0):]
    
    if any(char.isdigit() for char in string):
    
        idxS = next(i for i,j in list(enumerate(string,1))[::1] if j.isdigit())
        idxF = next(i for i,j in list(enumerate(string,1))[::-1] if j.isdigit())
    
        if idxS>1:
            if string[idxS-2]=='-':
                idxS -= 1
        
        return float(string[idxS-1:idxF])
    else:
        return False
    
def result_by_condition(listOfElements, condition):
    ''' Returns the indexes of items in the list that returns True when passed
    to condition() '''

    for i in range(len(listOfElements)): 
        if condition(listOfElements[i]) == True:
            output = listOfElements[i]
            return output, i

def parse_header(keywords,defaults,key,fn):
    
    keywords0 = keywords[key]
    defaults0 = defaults[key]
    
    if not isinstance(keywords0, tuple):
        keywords0 = (keywords0,'dum')
    
    output = []
    for string in keywords0:
        output.append(line_for_phrase_in_file(string, fn))
        
    if not any(output):
        return defaults0
        
    line, idx = result_by_condition(output, lambda x: x != False)            
    result = parse_digits(line,keywords0)
    
    if result is False:
        result = line[len(keywords0[idx]):].strip()
    elif 'mT' in line:
        result /= 1000.0
    
    return result

def parse_measurements(X):

    Hcalib = parse_header(X['keywords'],X['defaults'],'calibration field',X['fn'])
    
    skiprows = line_num_for_phrase_in_file('Time Stamp',X['fn'])
    if skiprows is not False: #timestamp exists - new file format
        df = pd.read_csv(X['fn'],skiprows=skiprows,encoding='latin9')
        Htemp = np.array(df['Field (µ0H) [T]'])
        Mtemp = np.array(df['Moment (m) [A·m²]'])
        ttemp = np.array(df['Time Stamp [s]'])
        if Hcalib == 'N/A':
            Hcalib = Htemp[0]
        idx = np.argwhere((Htemp[1:]<Htemp[:-1]) & (np.abs(Htemp[1:]-Hcalib)>0.0025)) #index of all calibration points

        #create last FORC first
        M = Mtemp[int(idx[-1])+1:]
        H = Htemp[int(idx[-1])+1:]
        Ft = ttemp[int(idx[-1])+1:]
        Hr = np.ones(len(M))*Htemp[int(idx[-1])+1]
        Fk = np.ones(len(M))*150
        Fj = np.arange(len(M))+1

        for i in range(len(idx)-1):
            Mappend = Mtemp[int(idx[i])+1:int(idx[i+1])]
            M = np.append(M,Mappend)
            H = np.append(H,Htemp[int(idx[i])+1:int(idx[i+1])])
            Hr = np.append(Hr,np.ones(len(Mappend))*Htemp[int(idx[i])+1])
            Ft = np.append(Ft,ttemp[int(idx[i])+1:int(idx[i+1])])
            Fk = np.append(Fk,np.ones(len(Mappend))+i)
            Fj = np.append(Fj,np.arange(len(Mappend))+1)

    else:
        skiprows = line_num_for_starting_data_line(X['fn'])
        nrows = parse_header(X['keywords'],X['defaults'],'number of rows',X['fn'])
        df = pd.read_csv(X['fn'],skiprows=skiprows,encoding='latin9',header=None,nrows=nrows)
        temp = np.array(df)
        Htemp = temp[:,0]
        Mtemp = temp[:,1]
        if Hcalib == 'N/A':
            Hcalib = Htemp[0]
        idx = np.argwhere((Htemp[1:]<Htemp[:-1]) & (np.abs(Htemp[1:]-Hcalib)>0.005)) #index of all calibration points

        #create last FORC first
        M = Mtemp[int(idx[-1])+1:]
        H = Htemp[int(idx[-1])+1:]
        Hr = np.ones(len(M))*Htemp[int(idx[-1])+1]
        Fk = np.ones(len(M))*len(idx)
        Fj = np.arange(len(M))+1

        for i in range(len(idx)-1):
            Mappend = Mtemp[int(idx[i])+1:int(idx[i+1])]
            M = np.append(M,Mappend)
            H = np.append(H,Htemp[int(idx[i])+1:int(idx[i+1])])
            Hr = np.append(Hr,np.ones(len(Mappend))*Htemp[int(idx[i])+1])
            Fk = np.append(Fk,np.ones(len(Mappend))+i)
            Fj = np.append(Fj,np.arange(len(Mappend))+1)

        Ft=measurement_times(X,Fk,Fj)
    
    #sort results into increasing time order
    idx = np.argsort(Ft)
    M, H, Hr, Ft, Fk, Fj = M[idx], H[idx], Hr[idx], Ft[idx], Fk[idx], Fj[idx]
    
    unit = parse_header(X['keywords'],X['defaults'],'units',X['fn']) #Ensure use of SI units
    
    if unit=='Cgs':
        H=H/1E4 #Convert Oe into T
        Hr=Hr/1E4 #Convert Oe into T
        M=M/1E3 #Convert emu to Am^2

    dH = np.mean(np.diff(H[Fk==np.max(Fk)])) #mean field spacing
    
    return H, Hr, M, Fk, Fj, Ft, dH

def parse_units(X):
    """Function to extract instrument unit settings ('') from FORC data file header
    
    Inputs:
    file: name of data file (string)    

    Outputs:
    CGS [Cgs setting] or SI [Hybrid SI] (string)
    """
    unit = parse_header(X['keywords'],X['defaults'],'units',X['fn']) #Ensure use of SI units

    if 'Hybrid' in unit:
        unit = 'SI'
    
    return unit


def parse_mass(X):

    mass = parse_header(X['keywords'],X['defaults'],'mass',X['fn']) #Ensure use of SI units
    
    return mass

def measurement_times(X,Fk,Fj):
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
    
    unit = parse_header(X['keywords'],X['defaults'],'units',X['fn'])
    tr=parse_header(X['keywords'],X['defaults'],'pause at reversal',X['fn'])
    tau=parse_header(X['keywords'],X['defaults'],'averaging time',X['fn'])
    tcal=parse_header(X['keywords'],X['defaults'],'pause at calibration',X['fn'])
    ts=parse_header(X['keywords'],X['defaults'],'pause at saturation',X['fn'])
    alpha=parse_header(X['keywords'],X['defaults'],'field slewrate',X['fn'])
    Hs=parse_header(X['keywords'],X['defaults'],'saturation field',X['fn'])
    Hb2=parse_header(X['keywords'],X['defaults'],'Hb2',X['fn'])
    Hb1=parse_header(X['keywords'],X['defaults'],'Hb1',X['fn'])
    Hc2=parse_header(X['keywords'],X['defaults'],'Hc2',X['fn'])
    N=parse_header(X['keywords'],X['defaults'],'number of FORCs',X['fn'])

    if unit=='Cgs':
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

def parse_calibration(X):
    
    Hcalib = parse_header(X['keywords'],X['defaults'],'calibration field',X['fn'])
    
    skiprows = line_num_for_phrase_in_file('Time Stamp',X['fn'])
    if skiprows is not False: #timestamp exists - new file format
        df = pd.read_csv(X['fn'],skiprows=skiprows,encoding='latin9')
        Htemp = np.array(df['Field (µ0H) [T]'])
        Mtemp = np.array(df['Moment (m) [A·m²]'])
        ttemp = np.array(df['Time Stamp [s]'])
        if Hcalib == 'N/A':
            Hcalib = Htemp[0]
        idx = np.argwhere(np.abs(Htemp-Hcalib)<0.001)
        Hcal, Mcal, tcal = Htemp[idx], Mtemp[idx], ttemp[idx]
    else: #no timestamp - old file format, find line starting with "+"
        skiprows = line_num_for_starting_data_line(X['fn'])
        nrows = parse_header(X['keywords'],X['defaults'],'number of rows',X['fn'])
        df = pd.read_csv(X['fn'],skiprows=skiprows,encoding='latin9',header=None,nrows=nrows)
        temp = np.array(df)
        Htemp = temp[:,0]
        Mtemp = temp[:,1]
        if Hcalib == 'N/A':
            Hcalib = Htemp[0]
        idx = np.argwhere(np.abs(Htemp-Hcalib)<0.001)
        Hcal, Mcal = Htemp[idx], Mtemp[idx]
        tcal = calibration_times(X, len(Hcal))
    
    return Hcal, Mcal, tcal

def calibration_times(X, Npts):
    """Function to estimate the time at which calibration points were measured in a FORC sequence
    
    Follows the procedure given in:
    R. Egli (2013) VARIFORC: An optimized protocol for calculating non-regular first-order reversal curve (FORC) diagrams. Global and Planetary Change, 110, 302-320, doi:10.1016/j.gloplacha.2013.08.003.

    Inputs:
    file: name of data file (string)    
    Npts: number of calibration points (int)

    Outputs:
    tcal_k: Estimated times at which the calibration points were measured (float)
    """    
    #unit=parse_units(file) #determine measurement system (CGS or SI)
    unit = parse_header(X['keywords'],X['defaults'],'units',X['fn'])
    tr=parse_header(X['keywords'],X['defaults'],'pause at reversal',X['fn'])
    tau=parse_header(X['keywords'],X['defaults'],'averaging time',X['fn'])
    tcal=parse_header(X['keywords'],X['defaults'],'pause at calibration',X['fn'])
    ts=parse_header(X['keywords'],X['defaults'],'pause at saturation',X['fn'])
    alpha=parse_header(X['keywords'],X['defaults'],'field slewrate',X['fn'])
    Hs=parse_header(X['keywords'],X['defaults'],'saturation field',X['fn'])
    Hb2=parse_header(X['keywords'],X['defaults'],'Hb2',X['fn'])
    Hb1=parse_header(X['keywords'],X['defaults'],'Hb1',X['fn'])
    Hc2=parse_header(X['keywords'],X['defaults'],'Hc2',X['fn'])
    N=parse_header(X['keywords'],X['defaults'],'number of FORCs',X['fn'])

    if unit=='Cgs':
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

def sample_details(X):

    sample = X['fn'].split('/')[-1]
    sample = sample.split('.')
    
    if type(sample) is list:
        sample=sample[0]

    units=parse_units(X)
    mass=parse_mass(X)
  
    return sample, units, mass

def measurement_limts(X):
    """Function to find measurement limits and conver units if required

    Inputs:
    file: name of data file (string)    


    Outputs:
    Hc1: minimum Hc
    Hc2: maximum Hc
    Hb1: minimum Hb
    Hb2: maximum Hb
    """    
    
    unit = parse_header(X['keywords'],X['defaults'],'units',X['fn'])
    Hb2=parse_header(X['keywords'],X['defaults'],'Hb2',X['fn'])
    Hb1=parse_header(X['keywords'],X['defaults'],'Hb1',X['fn'])
    Hc2=parse_header(X['keywords'],X['defaults'],'Hc2',X['fn'])
    Hc1=parse_header(X['keywords'],X['defaults'],'Hc1',X['fn'])

    if unit=='Cgs': #convert CGS to SI
        Hc2=Hc2/1E4 #convert from Oe to T
        Hc1=Hc1/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T  

    return Hc1, Hc2, Hb1, Hb2

#### Unit conversion ####
def CGS2SI(X):
    
    X["H"] = X["H"]/1E4 #convert Oe into T
    X["M"] = X["M"]/1E3 #convert emu to Am2
      
    return X


    