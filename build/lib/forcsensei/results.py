import numpy as np
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, Layout, VBox, HBox
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as sps
import matplotlib.ticker as mtick
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


#### FORC plotting ####
def forc(X):

    #unpack data
    Xi = X['Xi']
    Yi = X['Yi']
    Zi = X['Zi']
    SEi = X['SEi']
    Pi = X['Pi']
    Hc1 = X['Hc1']
    Hc2 = X['Hc2']
    Hb1 = X['Hb1']
    Hb2 = X['Hb2']

    #Set up widgets for interactive plot
    style = {'description_width': 'initial'} #general style settings
    
    #DEFINE INTERACTIVE WIDGETS
    
    #should a colorbar be included
    colorbar_widge = widgets.Checkbox(value=False, description = 'Show final FORC plot',style=style) 
    pval_widge = widgets.Checkbox(value=False, description = 'Show 0.05 significance contour',style=style) 

    colormin_widge = widgets.FloatSlider(
        value=0.0,
        min=0.00,
        max=0.999,
        step=0.001,
        description='Rescale colormap minimum',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=False,
        readout_format='.2f',
        style=style
    )

    colormax_widge = widgets.FloatSlider(
        value=1.0,
        min=0.001,
        max=1,
        step=0.001,
        description='Rescale colormap maximum',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=False,
        readout_format='.2f',
        style=style
    )

    #Frequency for contour lines to be included in plot
    contour_widge = widgets.Select(
        options=[['Select contour frequency',-1],
                 ['Every level',1],
                 ['Every 2nd level',2],
                 ['Every 3rd level',3],
                 ['Every 4th level',4],
                 ['Every 5th level',5],
                 ['Every 10th level',10],
                 ['Every 20th level',20],
                 ['Every 50th level',50],
                ],
        value=-1,
        rows=1,
        description='Plot contours',style=style)
    
    contourpts_widge = widgets.FloatSlider(value=1.0,min=0.5,max=3.0,step=0.5, description = 'Contour line width [pts]',style=style)

    #check box for plot download
    download_widge = widgets.Checkbox(value=False, description = 'Download plot',style=style) 
    
    #How many contour levels should be included
    level_widge = widgets.Select(
        options=[['20',20],['30',30],['50',50],['75',75],['100',100],['200',200],['500',500]],
        value=100,
        rows=1,
        description='Number of color levels',style=style)

    #plot limit widgets
    if X['unit']=='SI': 
        xmin_widge = widgets.FloatText(value=0,description='Minimum B$_\mathrm{c}$ [Oe]',style=style,step=10)    
        xmax_widge = widgets.FloatText(value=np.round(Hc2*1000)/1000,description='Maximum B$_\mathrm{c}$ [Oe]',style=style,step=10)
        ymin_widge = widgets.FloatText(value=np.round((Hb1-Hc2)*1000)/1000,description='Minimum B$_\mathrm{u}$ [Oe]',style=style,step=10)
        ymax_widge = widgets.FloatText(value=np.round(Hb2*1000)/1000,description='Maximum B$_\mathrm{u}$ [Oe]',style=style,step=10)
    elif X['unit']=='cgs':
        xmin_widge = widgets.FloatText(value=0,description='Minimum H$_\mathrm{c}$ [Oe]',style=style,step=10)    
        xmax_widge = widgets.FloatText(value=np.round(Hc2*1000)/1000,description='Maximum H$_\mathrm{c}$ [Oe]',style=style,step=10)
        ymin_widge = widgets.FloatText(value=np.round((Hb1-Hc2)*1000)/1000,description='Minimum H$_\mathrm{u}$ [Oe]',style=style,step=10)
        ymax_widge = widgets.FloatText(value=np.round(Hb2*1000)/1000,description='Maximum H$_\mathrm{u}$ [Oe]',style=style,step=10) 

    #launch the interactive FORC plot
    x = interactive(forcplot,
             Xi=fixed(Xi), #X point grid
             Yi=fixed(Yi), #Y point grid
             Zi=fixed(Zi), #interpolated Z values
             SEi = fixed(SEi), #interpolated standard errors
             Pi = fixed(Pi), #P values
             fn=fixed(X['sample']), #File information
             mass=fixed(X['mass']), #Preprocessing information
             unit=fixed(X['unit']),
             colorbar=colorbar_widge, #Include colorbar             
             level=level_widge, #Number of levels to plot 
             contour=contour_widge, #Contour levels to plot
             contourpts=contourpts_widge, #Contour line width
             xmin=xmin_widge, #X-minimum
             xmax=xmax_widge, #X-maximum
             ymin=ymin_widge, #Y-minimum
             ymax=ymax_widge, #Y-maximum
             colormin = colormin_widge, #adjust colormap minimum
             colormax = colormax_widge, #adjust colormap minimum
             download = download_widge #download plot
            )
    
    #create tabs
    tab_nest = widgets.Tab()
    # tab_nest.children = [tab_visualise]
    tab_nest.set_title(0, 'FORC PLOTTING')

    #interact function in isolation
    tab_nest.children = [VBox(children = x.children)]
    display(tab_nest)
    
    #display(x) #display the interactive plot

def forcplot(Xi,Yi,Zi,SEi,Pi,fn,mass,unit,colorbar,level,contour,contourpts,xmin,xmax,ymin,ymax,colormin,colormax,download):
    

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    
    if mass.value<0.0:        
        Xi_new = Xi
        Yi_new = Yi
        Zi_new = Zi
        SEi_new = SEi
        Pi_new = Pi
        SEi_new[Zi_new==0.0]=0.0
        SEi_new[np.isnan(SEi_new)]=0.0
        if unit=='SI':
            xlabel_text = 'B$_\mathrm{c}$ [T]' #label Hc axis [SI units]
            xlabel_csv = 'Bc [T]'
            ylabel_text = 'B$_\mathrm{u}$ [T]' #label Hu axis [SI units]
            ylabel_csv = 'Bu [T]'
            cbar_text = 'Am$^2$ T$^{-2}$'
            se_csv = 'rho [Am**2 / T**2]'
        elif unit=='cgs':
            xlabel_text = 'H$_\mathrm{c}$ [Oe]' #label Hc axis [SI units]
            xlabel_csv = 'Hc [Oe]'
            ylabel_text = 'H$_\mathrm{u}$ [Oe]' #label Hu axis [SI units]
            ylabel_csv = 'Hu [Oe]'
            cbar_text = 'emu$ Oe$^{-2}$'
            se_csv = 'rho [emu / Oe**2]'            
    else:
        Xi_new = Xi
        Yi_new = Yi
        Zi_new = Zi / (mass.value/1000.0)
        SEi_new = SEi / (mass.value/1000.0)
        SEi_new[Zi_new==0.0]=0.0
        SEi_new[np.isnan(SEi_new)]=0.0
        Pi_new = Pi
        if unit=='SI':
            Zi_new = Zi / (mass.value/1000.0)
            SEi_new = SEi / (mass.value/1000.0)        
            xlabel_text = 'B$_\mathrm{c}$ [T]' #label Hc axis [SI units]
            xlabel_csv = 'Bc [T]'
            ylabel_text = 'B$_\mathrm{u}$ [T]' #label Hu axis [SI units]
            ylabel_csv = 'Bu [T]'
            cbar_text = 'Am$^2$ T$^{-2}$ kg$^{-1}$'
            se_csv = 'se [Am**2 / T**2 / kg]'
        elif unit=='cgs':
            Zi_new = Zi / (mass.value)
            SEi_new = SEi / (mass.value)        
            xlabel_text = 'H$_\mathrm{c}$ [Oe]' #label Hc axis [SI units]
            xlabel_csv = 'Hc [Oe]'
            ylabel_text = 'H$_\mathrm{u}$ [Oe]' #label Hu axis [SI units]
            ylabel_csv = 'Hu [Oe]'
            cbar_text = 'emu Oe$^{-2}$ g$^{-1}$'
            se_csv = 'se [emu/ Oe**2 / g]'        
        
        SEi_new[Zi_new==0.0]=0.0
        SEi_new[np.isnan(SEi_new)]=0.0        

    #define colormaps
    idx=(Xi_new>=xmin) & (Xi_new<=xmax) & (Yi_new>=ymin) & (Yi_new<=ymax) #find points currently in view
    cmap,vmin,vmax = FORCinel_colormap(Zi_new[idx])
    #cmap, norm = FORCinel_colormap(Zi_new[idx])

    Zi_trunc = np.copy(Zi_new)
    Zi_trunc[np.isnan(Zi_trunc)] = 0.0
    Zi_trunc[Zi_trunc<vmin]=vmin
    
    vmini = vmin*(1-colormin)
    vmaxi = vmax*colormax

    idx = (Zi_trunc>=vmini) & (Zi_trunc<=vmaxi)
    cmap,vmin,vmax = FORCinel_colormap(Zi_trunc[idx])

    CS = ax.contourf(Xi_new, Yi_new, Zi_trunc, level, cmap = cmap, vmin=vmin, vmax=vmax)
           
    if (contour>0) & (contour<level):
        CS2 = ax.contour(CS, levels=CS.levels[::contour], colors='k',linewidths=contourpts)

    #if pval==True:
    #    CS3 = ax.contour(Xi_new, Yi_new, Pi_new, levels=[0.05], colors=['r'])

    ax.set_xlabel(xlabel_text,fontsize=14) #label Hc axis [SI units]
    ax.set_ylabel(ylabel_text,fontsize=14) #label Hu axis [SI units]  

    # Set plot Xlimits
    xlimits = np.sort((xmin,xmax))
    ax.set_xlim(xlimits)
    
    #Set plot Ylimits
    ylimits = np.sort((ymin,ymax))
    ax.set_ylim(ylimits)
    
    #Set ticks and plot aspect ratio
    ax.tick_params(labelsize=14)
    ax.set_aspect('equal') #set 1:1 aspect ratio
    ax.minorticks_on() #add minor ticks
    
    #Add colorbar
    if colorbar == True:    
        cbar = fig.colorbar(CS,fraction=0.04, pad=0.08,format='%.2e')
        cbar.ax.tick_params(labelsize=14)
        #cbar.ax.set_title(cbar_text,fontsize=14)
        cbar.set_label(cbar_text,fontsize=14)
        #cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    
    #Activate download to same folder as data file
    if download==True:
        outputfile = fn.value+'_FORC.pdf'
        plt.savefig(outputfile, dpi=300, bbox_inches="tight")
        
        ar = np.column_stack((np.reshape(Xi_new,(-1,1)),np.reshape(Yi_new,(-1,1)),np.reshape(Zi_trunc,(-1,1)),np.reshape(SEi,(-1,1))))
        outputfile = fn.value+'_XYZ.csv'
        with open(outputfile, 'w') as fp:
            fp.write(xlabel_csv+','+ylabel_csv+','+se_csv+','+se_csv + '\n')
            np.savetxt(fp, ar, '%s', ',')
    
    #show the final plot
    plt.show()

def FORCinel_colormap(Z):

    #setup initial colormap assuming that negative range does not require extension
    cdict = {'red':     ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                         (0.3193,  102/255, 102/255),
                       (0.563,  204/255, 204/255),
                       (0.6975,  204/255, 204/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255)),

            'green':   ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  178/255, 178/255),
                        (0.563,  204/255, 204/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  102/255, 102/255),
                       (0.9748,  25/255, 25/255),
                       (1.0, 25/255, 25/255)),

             'blue':   ((0.0,  255/255, 255/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  102/255, 102/255),
                        (0.563,  76/255, 76/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255))}

    if np.abs(np.min(Z))<=np.max(Z)*0.19: #negative extension is not required
        #cmap = LinearSegmentedColormap('forc_cmap', cdict)
        vmin = -np.max(Z)*0.19
        vmax = np.max(Z)
    else: #negative extension is required
        vmin=np.min(Z)
        vmax=np.max(Z)        
    
    anchors = np.zeros(10)
    anchors[1]=(-0.025*vmax-vmin)/(vmax-vmin)
    anchors[2]=(-0.005*vmax-vmin)/(vmax-vmin)
    anchors[3]=(0.025*vmax-vmin)/(vmax-vmin)
    anchors[4]=(0.19*vmax-vmin)/(vmax-vmin)
    anchors[5]=(0.48*vmax-vmin)/(vmax-vmin)
    anchors[6]=(0.64*vmax-vmin)/(vmax-vmin)
    anchors[7]=(0.80*vmax-vmin)/(vmax-vmin)
    anchors[8]=(0.97*vmax-vmin)/(vmax-vmin)
    anchors[9]=1.0

    Rlst = list(cdict['red'])
    Glst = list(cdict['green'])
    Blst = list(cdict['blue'])

    for i in range(9):
        Rlst[i] = tuple((anchors[i],Rlst[i][1],Rlst[i][2]))
        Glst[i] = tuple((anchors[i],Glst[i][1],Glst[i][2]))
        Blst[i] = tuple((anchors[i],Blst[i][1],Blst[i][2]))
        
    cdict['red'] = tuple(Rlst)
    cdict['green'] = tuple(Glst)
    cdict['blue'] = tuple(Blst)

    cmap = LinearSegmentedColormap('forc_cmap', cdict)

    return cmap, vmin, vmax

    #### Profile Plotting ####

#### Profile plotting ####

def profile_options(X):
    Hb1 = X['Hb1']-X['Hc2']
    Hb2 = X['Hb2']
    Hc1 = np.maximum(X['Hc1'],0)
    Hc2 = X['Hc2']
    style = {'description_width': 'initial'} #general style settings
    
    HL = widgets.HTML(value='<hr style="height:3px;border:none;color:#333;background-color:#333;" />')
    
    P_title = widgets.HTML(value='<h3>Select profile type:</h3>')
    P_widge = widgets.RadioButtons(options=[('Horizontal profile',0), ('Vertical profile',1)],
                                       value=0,
                                       style=style)
    
    H_title = widgets.HTML(value='<h4>Horizontal profile specification:</h4>')

    if X['unit'] == 'SI':
        x_Hb_widge = widgets.FloatSlider(
            value=0.0,
            min=Hb1,
            max=Hb2,
            step=0.001,
            description='B$_u$ [T]',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout={'width': '350px'},
            style = style
        )
    elif X['unit'] == 'cgs':
        x_Hb_widge = widgets.FloatSlider(
            value=0.0,
            min=Hb1,
            max=Hb2,
            step=10,
            description='H$_u$ [Oe]',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout={'width': '350px'},
            style = style
        )
        
    if X['unit'] == 'SI':
        x_Hc_widge = widgets.FloatRangeSlider(
            value=[Hc1,Hc2],
            min=Hc1,
            max=Hc2,
            step=0.001,
            description='B$_c$ [T]',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout={'width': '350px'},
            style = style
        )
    elif X['unit'] == 'cgs':
        x_Hc_widge = widgets.FloatRangeSlider(
            value=[Hc1,Hc2],
            min=Hc1,
            max=Hc2,
            step=10,
            description='H$_c$ [Oe]',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout={'width': '350px'},
            style = style
        )
    
    V_title = widgets.HTML(value='<h4>Vertical profile specification:</h4>')
    
    if X['unit'] == 'SI':
        y_Hc_widge = widgets.FloatSlider(
            value=(Hc1+Hc2)/2.0,
            min=Hc1,
            max=Hc2,
            step=0.001,
            description='B$_c$ [T]',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout={'width': '350px'},
            style = style
        )
    elif X['unit'] == 'cgs':
        y_Hc_widge = widgets.FloatSlider(
            value=(Hc1+Hc2)/2.0,
            min=Hc1,
            max=Hc2,
            step=10,
            description='H$_c$ [T]',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout={'width': '350px'},
            style = style
        )

    if X['unit'] == 'SI':
        y_Hb_widge = widgets.FloatRangeSlider(
            value=[Hb1,Hb2],
            min=Hb1,
            max=Hb2,
            step=0.001,
            description='B$_u$ [T]',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout={'width': '350px'},
            style = style
        )

    elif X['unit'] == 'cgs':
        y_Hb_widge = widgets.FloatRangeSlider(
            value=[Hb1,Hb2],
            min=Hb1,
            max=Hb2,
            step=10,
            description='H$_u$ [T]',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout={'width': '350px'},
            style = style
        )
    
    profile_widge = VBox([P_title,P_widge,HL,H_title,x_Hb_widge,x_Hc_widge, \
                         HL,V_title,y_Hc_widge,y_Hb_widge])
    
    profile_nest = widgets.Tab()
    profile_nest.children = [profile_widge]
    profile_nest.set_title(0, 'PLOT PROFILES')
    display(profile_nest)   
    
    X['P_widge'] = P_widge
    X['x_Hb_widge'] = x_Hb_widge
    X['x_Hc_widge'] = x_Hc_widge
    X['y_Hc_widge'] = y_Hc_widge
    X['y_Hb_widge'] = y_Hb_widge

    return X

def profile_plot(X):

    if X['P_widge'].value==0:
        X = x_profile(X,X['x_Hc_widge'].value,X['x_Hb_widge'].value)
    else:
        X = y_profile(X,X['y_Hc_widge'].value,X['y_Hb_widge'].value)
    
    return X

def x_profile(X,Hc,Hb):

    Hc1, Hc2 = Hc[0], Hc[1]

    dH = X['dH']
    NH = int(np.sqrt((Hc2-Hc1)**2)/dH)
    Hc0 = np.linspace(Hc1,Hc2,NH)
    Hb0 = np.linspace(Hb,Hb,NH)
    
    rho_int = X['Zint'](Hc0,Hb0)
    coef = sps.norm.ppf(0.025/np.sum(rho_int.mask==False))
    CI_int = X['SEint'](Hc0,Hb0)*coef

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)
    
    if X['mass'].value>0.0:
        if X['unit'] == 'SI':
            ax1.plot(Hc0,rho_int/(X['mass'].value/1000.0),color='k')
            ax1.fill_between(Hc0, (rho_int-CI_int)/(X['mass'].value/1000.0), (rho_int+CI_int)/(X['mass'].value/1000.0),color='lightgrey')
            ax1.set_ylabel('Am$^2$ T$^{-2}$ kg$^{-1}$',fontsize=14)
        elif X['unit'] == 'cgs':
            ax1.plot(Hc0,rho_int/(X['mass'].value),color='k')
            ax1.fill_between(Hc0, (rho_int-CI_int)/(X['mass'].value), (rho_int+CI_int)/(X['mass'].value),color='lightgrey')
            ax1.set_ylabel('emu Oe$^{-2}$ g$^{-1}$',fontsize=14)
    else:
        ax1.plot(Hc0,rho_int,color='k')
        ax1.fill_between(Hc0, (rho_int-CI_int), (rho_int+CI_int),color='lightgrey')
        if X['unit'] == 'SI':
            ax1.set_ylabel('Am$^2$ T$^{-2}$',fontsize=14)
        elif X['unit'] == 'cgs':
            ax1.set_ylabel('emu Oe$^{-2}$',fontsize=14)


    ax1.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='14')
    ax1.tick_params(axis='both',which='minor',direction='out',length=3.5,width=1,color='k')
    
    if X['unit'] == 'SI':
        ax1.set_xlabel('B$_\mathrm{c}$ [T]',fontsize=14)
    elif X['unit'] == 'cgs':
        ax1.set_xlabel('H$_\mathrm{c}$ [Oe]',fontsize=14)        
    
    ax1.minorticks_on()
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))


    outputfile = X['sample'].value+'_Hc_PROFILE.pdf'
    plt.savefig(outputfile, dpi=300, bbox_inches="tight")
    plt.show
    
    return X

def y_profile(X,Hc,Hb):

    Hb1, Hb2 = Hb[0], Hb[1]

    dH = X['dH']
    NH = int(np.sqrt((Hb2-Hb1)**2)/dH)
    Hc0 = np.linspace(Hc,Hc,NH)
    Hb0 = np.linspace(Hb1,Hb2,NH)
    
    rho_int = X['Zint'](Hc0,Hb0)
    coef = sps.norm.ppf(0.025/np.sum(rho_int.mask==False))
    CI_int = X['SEint'](Hc0,Hb0)*coef

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)

    if X['mass'].value>0.0:
        if X['unit'] == 'SI':
            ax1.plot(Hb0,rho_int/(X['mass'].value/1000.0),color='k')
            ax1.fill_between(Hb0, (rho_int-CI_int)/(X['mass'].value/1000.0), (rho_int+CI_int)/(X['mass'].value/1000.0),color='lightgrey')
            ax1.set_ylabel('Am$^2$ T$^{-2}$ kg$^{-1}$',fontsize=14)
        elif X['unit'] == 'cgs':
            ax1.plot(Hb0,rho_int/(X['mass'].value),color='k')
            ax1.fill_between(Hb0, (rho_int-CI_int)/(X['mass'].value), (rho_int+CI_int)/(X['mass'].value),color='lightgrey')
            ax1.set_ylabel('emu Oe$^{-2}$ g$^{-1}$',fontsize=14)
    else:
        ax1.plot(Hb0,rho_int,color='k')
        ax1.fill_between(Hb0, (rho_int-CI_int), (rho_int+CI_int),color='lightgrey')
        if X['unit'] == 'SI':
            ax1.set_ylabel('Am$^2$ T$^{-2}$',fontsize=14)
        elif X['unit'] == 'cgs':
            ax1.set_ylabel('emu Oe$^{-2}$',fontsize=14)            
    
    ax1.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='14')
    ax1.tick_params(axis='both',which='minor',direction='out',length=3.5,width=1,color='k')

    if X['unit'] == 'SI':
        ax1.set_xlabel('B$_\mathrm{u}$ [T]',fontsize=14)
    elif X['unit'] == 'cgs':
        ax1.set_xlabel('H$_\mathrm{u}$ [Oe]',fontsize=14)        
    
    ax1.minorticks_on()
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))


    outputfile = X['sample'].value+'_Hu_PROFILE.pdf'
    plt.savefig(outputfile, dpi=300, bbox_inches="tight")
    plt.show
    
    return X