def simulate_profile_fit(ACCUMULATION_RATE,CORE_DEPTH,DECAY_CONST,ACTIVITY_INITIAL,SUPPORTED_LEVEL,NOISE_AMPLITUDE,NUM_POINTS_MODELED,SUPLVL_DEPTH_GUESS,**SUPLVL_VALUE):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    

    #simulate loglinear data with added noise from radioactive decay equation
    age = np.linspace(start=0, stop=CORE_DEPTH/ACCUMULATION_RATE, num=NUM_POINTS_MODELED)
    dep = np.linspace(start=0, stop=CORE_DEPTH,                   num=NUM_POINTS_MODELED)
    a_t = ACTIVITY_INITIAL * np.exp(-0.0311 * age) + SUPPORTED_LEVEL
    noise = np.random.normal(0, NOISE_AMPLITUDE, NUM_POINTS_MODELED)
    a_t_noise = a_t + noise
    index_suplvl = min(range(NUM_POINTS_MODELED), key=lambda i: abs(dep[i]-SUPLVL_DEPTH_GUESS))
    if len(SUPLVL_VALUE)==0:
        suplvl_estimate = np.mean(a_t_noise[index_suplvl:])
    elif len(SUPLVL_VALUE)>0:
        suplvl_estimate = SUPLVL_VALUE['SUPLVL_VALUE']
    a_t_noise_excess = a_t_noise - suplvl_estimate


    #define function to calculate loglinear trendline
    def func(x,a,b):
        return a*np.log(x)+ b

    #remove negative data for trendline computation
    a_t_noise_excess_pos = np.ma.masked_array(a_t_noise_excess[:index_suplvl], mask=a_t_noise_excess[:index_suplvl]<0).compressed()
    dep_pos = np.ma.masked_array(dep[:index_suplvl], mask=a_t_noise_excess[:index_suplvl]<0).compressed()
    #apply function to data
    popt, pcov = curve_fit(func, a_t_noise_excess_pos, dep_pos)

    #compute curve stats
    y_pred = func(a_t_noise_excess_pos, *popt)
    r_squared=r2_score(dep_pos, y_pred)

    # PLOT -----------------------------------------------------------------------------------------

    fig, [ax1,ax3,ax4] = plt.subplots(1,3, sharey=True)
    fig.set_size_inches(12,4)
    ax1.invert_yaxis()

    ax1.scatter(a_t_noise, dep, s=2,c='b')
    ax1.set_ylabel('depth (cmbsf)')
    ax1.set_xlabel('activity (dpm/g)')
    ax1.text(.1, .92, f'Simulated profile', transform=ax1.transAxes, fontsize='medium', fontweight='bold', color='k')
    ax1.legend(["Supported"],loc='lower right')


    ax3.scatter(a_t_noise[:index_suplvl], dep[:index_suplvl], s=2, c='b')
    ax3.scatter(a_t_noise_excess[:index_suplvl], dep[:index_suplvl], s=2,c='r')
    ax3.scatter(a_t_noise[index_suplvl:], dep[index_suplvl:], s=2,c='k')
    ax3.vlines(suplvl_estimate, dep[index_suplvl], dep[-1],edgecolors='k')
    ax3.plot(a_t_noise_excess_pos, y_pred, color='r',alpha=0.5)
    ax3.legend(["sup","exs","bkg","suplvl","fit"],loc='lower right')

    ax3.set_xscale("log")
    ax3.set_ylabel('depth (cmbsf)')
    ax3.set_xlabel('activity (dpm/g)')
    ax3.text(.1, .92, f'Interpreted profile', transform=ax3.transAxes, fontsize='medium', fontweight='bold', color='k')
    ax4.axis('off')
    ax4.text(.1, .92, f'Line Fit Performance', transform=ax4.transAxes, fontsize='medium', fontweight='bold', color='k')
    ax4.text(.4, .85, f'True', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='b')
    ax4.text(.6, .85, f'Estimated', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='r')
    ax4.text(.0, .78, f'Sup. level', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='k')
    ax4.text(.4,.78, f'{SUPPORTED_LEVEL}', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='b')
    ax4.text(.6,.78, f'{np.round(suplvl_estimate,3)}', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='r')
    ax4.text(.0, .71, f'Acc. Rate', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='k')
    ax4.text(.4,.71, f'{ACCUMULATION_RATE}', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='b')
    ax4.text(.6,.71, f'{np.round(popt[0]*DECAY_CONST,3)}', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='r')
    ax4.text(.0, .64, f'R-Squared', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='k')
    ax4.text(.4,.64, f'---', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='b')
    ax4.text(.6,.64, f'{np.round(r_squared,3)}', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='r')
    ax4.text(.0, .57, f'Fit Slope', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='k')
    ax4.text(.4,.57, f'---', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='b')
    ax4.text(.6,.57, f'{np.round(popt[0],3)}', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='r')
    ax4.text(.0, .5, f'SAR % diff from true', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='k')
    ax4.text(.6, .5, f'{np.round((((popt[0]*DECAY_CONST-ACCUMULATION_RATE)/((ACCUMULATION_RATE)))*100),2)} %', transform=ax4.transAxes, fontsize='medium', fontweight='normal', color='r')
