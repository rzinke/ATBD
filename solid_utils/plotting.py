# Author: Marin Govorcin
# June, 2024
# Transient validation display function added by Saoussen Belhadj-aissa. July, 2024
# Time-series display routines added by Robert Zinke. March 2025

from typing import Callable
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from numpy.typing import NDArray


## Time-series plotting routines
from mintpy.mask import mask_matrix
from mintpy.objects import timeseries
from mintpy.utils import ptime, readfile

def plot_ts_rmse(ax, date_list, ts, ref_dis=None):
    """
    """
    # Parameters
    n_dates = len(date_list)
    positions = range(1, n_dates+1)

    # Determine the RMSE of each layer
    rmse = np.nanstd(ts, axis=(1,2))

    # Plot results
    markerline, stemlines, baseline = ax.stem(positions, 1000*rmse)
    plt.setp(baseline, color="k")

    # Plot reference line
    if ref_dis is not None:
        ax.axhline(ref_dis, color='k', linestyle='--')

    # Format plot
    ax.set_xticks(positions)
    ax.set_xticklabels(date_list, rotation=80)

    ax.set_ylabel('RMSE (mm)')


def plot_ts_violins(ax, date_list, ts, ref_dis=None):
    """
    """
    # Parameters
    n_dates, M, N = ts.shape
    MN = M * N
    skips = MN // 1000

    # Loop through layers
    ts_violins = []
    for i in range(n_dates):
        ts_lyr = ts[i,...]
        ts_lyr = ts_lyr[~np.isnan(ts_lyr)]

        ts_violins.append(1000*ts_lyr[::skips])
    ts_violins = np.column_stack(ts_violins)

    # Remove mean for plotting purposes
    ts_violins -= np.mean(ts_violins, axis=0)

    # Statistics
    ts_pcts = np.percentile(ts_violins, [16, 84], axis=0)

    # Plot violins
    positions = list(range(1, n_dates+1))
    ax.violinplot(ts_violins, positions=positions, widths=0.9, showextrema=False)
    ax.vlines(positions, ts_pcts[0,:], ts_pcts[1,:], lw=5)

    # Plot reference lines
    if ref_dis is not None:
        ax.axhline(ref_dis, color='k', linestyle='--')
        ax.axhline(-ref_dis, color='k', linestyle='--')

    # Format plot
    ax.set_xticks(positions)
    ax.set_xticklabels(date_list, rotation=80)

    ax.set_ylabel('Displacement Range (mm)')


def plot_ts_stats(ts_file, msk_file, ref_dis=None, dates_per_fig=10):
    """
    """
    # Read data cube
    ts, _ = readfile.read(ts_file, datasetName='timeseries')
    date_list = timeseries(ts_file).get_date_list()

    # Mask data cube
    if msk_file is not None:
        msk, _ = readfile.read(msk_file, datasetName='mask')
        ts = mask_matrix(ts, msk)

    # Number of layers in dataset
    n_lyrs = ts.shape[0]
    print(f"{n_lyrs:d} layers found")

    # Determine number of figures
    n_figs = np.ceil(n_lyrs / dates_per_fig).astype(int)

    # Loop through figures
    for i in range(n_figs):
        # Instantiate figure and axis
        fig, axes = plt.subplots(figsize=(9, 5.5), nrows=2)

        # Start and end indexes of data subset
        start_ndx = i * dates_per_fig
        end_ndx = start_ndx + dates_per_fig if i < (n_figs - 1) \
            else start_ndx + n_lyrs % dates_per_fig

        # Subset data
        date_list_fig = date_list[start_ndx:end_ndx]
        ts_fig = ts[start_ndx:end_ndx,...]

        # Plot RMSE of each layer
        plot_ts_rmse(axes[0], date_list_fig, ts_fig, ref_dis)
        axes[0].set_xlim([0, dates_per_fig+1])
        axes[0].set_xticklabels([])
        if ref_dis is not None:
            axes[0].set_ylim([0, 4*ref_dis])

        # Violin plots of each layer
        plot_ts_violins(axes[1], date_list_fig, ts_fig, ref_dis)
        axes[1].set_xlim([0, dates_per_fig+1])
        if ref_dis is not None:
            axes[1].set_ylim([-4*ref_dis, 4*ref_dis])

        # Format figure
        fig.suptitle(f"Displacements ({date_list_fig[0]:s} "
                     f"- {date_list_fig[-1]:s})")
        fig.tight_layout()


from solid_utils.fitting import IterativeOutlierFit

def plot_ts_fits(ref_site:str, gnss_fits:dict, insar_fits:dict):
    """Plot GNSS and InSAR timeseries and fit results, superimposed for each
    GNSS site location.

    For the GNSS and InSAR fits, one should pass a dictionary of
    IterativeOutlierFit objects, with key:value pairs of
    <site_name>:<fit_object>.

    Parameters: ref_site - str, reference site name
                gnss_fits - dict of IterativeOutlierFit objects
                insar_fits - dict of IterativeOutlierFit objects
    """
    # Check validity of passed datasets
    if gnss_fits.keys() != insar_fits.keys():
        raise Execption('GNSS and InSAR fits must relate to the same stations')
    site_names = [*gnss_fits.keys()]  # list of site names from dict keys

    for value in [*gnss_fits.values()] + [*gnss_fits.values()]:
        if type(value) != IterativeOutlierFit:
            raise Exception('Fits must have type IterativeOutlierFit')

    # Reference velocities
    gnss_ref_vel = gnss_fits[ref_site].m_hat[1]
    gnss_ref_vel_err = gnss_fits[ref_site].mhat_se[1]

    insar_ref_vel = insar_fits[ref_site].m_hat[1]
    insar_ref_vel_err = insar_fits[ref_site].mhat_se[1]
    
    # Loop through GNSS and InSAR sites
    for site_name in site_names:
        # Remove reference site velocity from GNSS velocity
        gnss_offset = gnss_fits[site_name].m_hat[0]
        gnss_vel_detr = gnss_fits[site_name].m_hat[1] - gnss_ref_vel
        gnss_vel_detr_err = np.sqrt(gnss_fits[site_name].mhat_se[1]**2
                                    + gnss_ref_vel_err**2)

        # Remove reference site velocity from GNSS timeseries
        gnss_dates = gnss_fits[site_name].dates
        t_gnss = np.array([(date - gnss_dates[0]).days/365.25
                           for date in gnss_dates])
        gnss_dis_detr = gnss_fits[site_name].dis - gnss_ref_vel*t_gnss
        gnss_fit_detr = gnss_fits[site_name].dis_hat - gnss_ref_vel*t_gnss

        # Remove reference site velocity from InSAR velocity
        insar_offset = insar_fits[site_name].m_hat[0]
        insar_vel_detr = insar_fits[site_name].m_hat[1] - insar_ref_vel
        insar_vel_detr_err = np.sqrt(insar_fits[site_name].mhat_se[1]**2
                                     + insar_ref_vel_err**2)

        # Remove reference site velocity from GNSS timeseries
        insar_dates = insar_fits[site_name].dates
        t_insar = np.array([(date - insar_dates[0]).days/365.25
                           for date in insar_dates])
        insar_dis_detr = insar_fits[site_name].dis - insar_ref_vel*t_insar
        insar_fit_detr = insar_fits[site_name].dis_hat - insar_ref_vel*t_insar
        insar_env_lo_detr = insar_fits[site_name].err_envelope[0] - insar_ref_vel*t_insar
        insar_env_hi_detr = insar_fits[site_name].err_envelope[1] - insar_ref_vel*t_insar

        # Plotting parameters
        n_dates = len(gnss_fits[site_name].dates)
        label_skips = n_dates//6
    
        # Spawn figure and axis
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axhline(0, c='dimgrey', linestyle='--')
    
        # Plot filtered data and model fit
        ax.scatter(gnss_dates, 1000*(gnss_dis_detr - gnss_offset),
                   s=3**2, c='grey', alpha=0.5, label=f"GNSS")
        ax.plot(gnss_dates, 1000*(gnss_fit_detr - gnss_offset), c='k')
    
        ax.scatter(insar_dates, 1000*(insar_dis_detr - insar_offset),
                   s=5**2, c='orange', label='InSAR')
        ax.plot(insar_dates, 1000*(insar_fit_detr - insar_offset),
                c='orange', linewidth=2)
        ax.fill_between(insar_dates,
                        y1=1000*(insar_env_lo_detr - insar_offset),
                        y2=1000*(insar_env_hi_detr - insar_offset),
                        color='orange', alpha=0.3, label='95% conf')

        # Format plot
        ax.legend()
        ax.set_xticks(gnss_dates[::label_skips])
        ax.set_xticklabels([date.strftime('%Y-%m-%d') \
                            for date in gnss_dates[::label_skips]],
                           rotation=80)
        ax.set_ylabel('LOS dis - detr (mm)')
        title = site_name
        if site_name == ref_site:
            title += ' (ref)'
        title += f"\n(GNSS {1000*gnss_vel_detr:.1f} " \
                 + f"+- {1000*gnss_vel_detr_err:.2f} mm/yr)"
        title += f"\n(InSAR {1000*insar_vel_detr:.1f} " \
                 + f"+- {1000*insar_vel_detr_err:.2f} mm/yr)"
        ax.set_title(title)
        fig.tight_layout()


## Validation display
def display_validation(pair_distance: NDArray, pair_difference: NDArray, pair_difference_err: NDArray,
                       site_name: str, start_date: str, end_date: str,
                       requirement: float = 2, distance_rqmt: list = [0.1, 50],
                       n_bins: int = 10, threshold: float = 0.683, 
                       sensor:str ='Sentinel-1', validation_type:str='secular',
                       validation_data:str='GNSS'):

   '''
    Parameters:
      pair_distance : array      - 1d array of pair distances used in validation
      pair_difference : array    - 1d array 0f pair double differenced velocity residuals
      site_name : str            - name of the cal/val site
      start_date  : str          - data record start date, eg. 20190101
      end_date : str             - data record end date, eg. 20200101
      requirement : float        - value required for test to pass
                                    e.g, 2 mm/yr for 3 years of data over distance requiremeent
      distance_rqmt : list       - distance over requirement is tested, eg. length scales of 0.1-50 km
      n_bins : int               - number of bins
      threshold : float          - threshold represents percentile of Gaussian normal distribution
                                    within residuals are expected to be to pass the test
                                    e.g. 0.683 for 68.3% or 1-sigma limit 
      sensor : str               - sensor used in validation, e.g Sentinel-1 or NISAR
      validation_type : str      - type of validation: secular, coseismic, transient
      validation_data : str      - data used to validate against; GNSS or INSAR

   Return
      validation_table
      validation_figure
   '''
   # init dataframe
   df = pd.DataFrame(np.vstack([pair_distance,
                                pair_difference,
                                pair_difference_err]).T,
                                columns=['distance', 'double_diff', 'double_diff_err'])

   # remove nans
   df_nonan = df.dropna(subset=['double_diff'])
   bins = np.linspace(*distance_rqmt, num=n_bins+1)
   bin_centers = (bins[:-1] + bins[1:]) / 2
   binned_df = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                observed=False)[['double_diff']]

   # get binned validation table 
   validation = pd.DataFrame([])
   validation['total_count[#]'] = binned_df.apply(lambda x: np.ma.masked_invalid(x).count())
   validation['passed_req.[#]'] = binned_df.apply(lambda x: np.count_nonzero(x < requirement))
   
   # Add total at the end
   validation = pd.concat([validation, pd.DataFrame(validation.sum(axis=0)).T])
   validation['passed_pc'] = validation['passed_req.[#]'] / validation['total_count[#]']
   validation['success_fail'] = validation['passed_pc'] > threshold
   validation.index.name = 'distance[km]'
   # Rename last row
   validation.rename({validation.iloc[-1].name:'Total'}, inplace=True)

   # Figure
   fig, ax = plt.subplots(1, figsize=(9, 3), layout="none", dpi=200)
   
   # Plot residuals
   ms = 8 if pair_difference.shape[0] < 1e4 else 0.3
   alpha = 0.6 if pair_difference.shape[0] < 1e4 else 0.2
   ax.scatter(df_nonan.distance, df_nonan.double_diff,
              color='black', s=ms, zorder=1, alpha=alpha, edgecolor='None')
   ax.errorbar(df_nonan.distance, df_nonan.double_diff, yerr=df_nonan.double_diff_err,
               color='black', linestyle='none', alpha=alpha, linewidth=0.5)

   ax.fill_between(distance_rqmt, 0, requirement, color='#e6ffe6', zorder=0, alpha=0.6)
   ax.fill_between(distance_rqmt, requirement, 21, color='#ffe6e6', zorder=0, alpha=0.6)
   ax.vlines(bins, 0, 21, linewidth=0.3, color='gray', zorder=1)
   ax.axhline(requirement, color='k', linestyle='--', zorder=3)

   # Bar plot for each bin
   quantile_th = binned_df.quantile(q=threshold)['double_diff'].values
   for bin_center, quantile, flag in zip(bin_centers,
                                         quantile_th,
                                         validation['success_fail']):
      if flag:
         color = '#227522'
      else:
         color = '#7c1b1b'
      ax.bar(bin_center, quantile, align='center', width=np.diff(bins)[0],
            color='None', edgecolor=color, linewidth=2, zorder=3)
      
   # Add legend with data info
   legend_kwargs = dict(transform=ax.transAxes, verticalalignment='top')
   props = dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0.4)
   textstr = f'Sensor: {sensor} \n{validation_data}-InSAR point pairs\n'
   textstr += f'Record: {start_date}-{end_date}'

   # place a text box in upper left in axes coords
   ax.text(0.02, 0.95, textstr, fontsize=8, bbox=props, **legend_kwargs)
   
   # Add legend with validation info 
   textstr = f'{validation_type.capitalize()} requirement\n'
   textstr += f'Site: {site_name}\n'
   if validation.loc['Total']['success_fail']:
      validation_flag = 'PASSED'
      validation_color = '#239d23'
   else: 
      validation_flag ='FAILED'
      validation_color = '#bc2e2e'

   props = {**props, **{'facecolor':'none', 'edgecolor':'none'}}
   ax.text(0.818, 0.93, textstr, fontsize=8, bbox=props, **legend_kwargs)
   ax.text(0.852, 0.82,  f"{validation_flag}",
           fontsize=10, weight='bold',
           bbox=props, **legend_kwargs)

   rect = patches.Rectangle((0.8, 0.75), 0.19, 0.2,
                           linewidth=1, edgecolor='black',
                           facecolor=validation_color,
                           transform=ax.transAxes)
   ax.add_patch(rect)

   # Title & labels
   fig.suptitle(f"{validation_type.capitalize()} requirement: {site_name}", fontsize=10)
   ax.set_xlabel("Distance (km)", fontsize=8)
   if validation_data == 'GNSS':
       txt = "Double-Differenced \nVelocity Residual (mm/yr)"
   else:
       txt = "Relative Velocity measurement (mm/yr)"    
   ax.set_ylabel(txt, fontsize=8)
   ax.minorticks_on()
   ax.tick_params(axis='x', which='minor', length=4, direction='in', top=False, width=1.5)
   ax.tick_params(axis='both', labelsize=8)
   ax.set_xticks(bin_centers, minor=True)
   ax.set_xticks(np.arange(0,55,5))
   ax.set_ylim(0,20)
   ax.set_xlim(*distance_rqmt)

   validation = validation.rename(columns={'success_fail': f'passed_req [>{threshold*100:.1f}%]'})

   return validation, fig

def display_validation_table(validation_table):
    # Display Statistics
    def bold_last_row(row):
        is_total = row.name == 'Total'
        styles = ['font-weight: bold; font-size: 14px; border-top: 3px solid black' if is_total else '' for _ in row]
        return styles
    
    def style_success_fail(value):
        color = '#e6ffe6' if value else '#ffe6e6'
        return 'background-color: %s' % color

    # Overall pass/fail criterion
    if validation_table.loc['Total'][validation_table.columns[-1]]:
        print("This velocity dataset passes the requirement.")
    else:
        print("This velocity dataset does not pass the requirement.")

    return (validation_table.style
            .bar(subset=['passed_pc'], vmin=0, vmax=1, color='gray')
            .format(lambda x: f'{x*100:.0f}%', na_rep="none", precision=1, subset=['passed_pc'])
            .apply(bold_last_row, axis=1)
            .map(style_success_fail, subset=[validation_table.columns[-1]])
           )


def display_coseismic_validation(pair_distance: NDArray, pair_difference: NDArray,
                                 site_name: str, start_date: str, end_date: str,
                                 requirement: Callable, distance_rqmt: list = [0.1, 50],
                                 n_bins: int = 10, threshold: float = 0.683,
                                 sensor:str ='Sentinel-1', validation_type:str='secular',
                                 validation_data:str='GNSS'):
    '''
    Parameters:
       pair_distance : array      - 1d array of pair distances used in validation
       pair_difference : array    - 1d array 0f pair double differenced velocity residuals
       site_name : str            - name of the cal/val site
       start_date  : str          - data record start date, eg. 20190101
       end_date : str             - data record end date, eg. 20200101
       requirement : lambda       - formula for validation requirement
       distance_rqmt : list       - distance over requirement is tested, eg. length scales of 0.1-50 km
       n_bins : int               - number of bins
       threshold : float          - threshold represents percentile of Gaussian normal distribution
                                    within residuals are expected to be to pass the test
                                    e.g. 0.683 for 68.3% or 1-sigma limit 
       sensor : str               - sensor used in validation, e.g Sentinel-1 or NISAR
       validation_type : str      - type of validation: secular, coseismic, transient
       validation_data : str      - data used to validate against; GNSS or INSAR

    Return
       validation_table
       validation_figure
    '''
    # Init dataframe
    df = pd.DataFrame(np.vstack([pair_distance,
                                 pair_difference]).T,
                                 columns=['distance', 'double_diff'])

    # Apply requirement
    df['requirement'] = df['double_diff'] < requirement(df['distance'])
    df['requirement'] = df['requirement'].astype(int)

    # Remove nans
    df_nonan = df.dropna(subset=['double_diff'])

    # Bin data
    bins = np.linspace(*distance_rqmt, num=n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_series = pd.cut(df_nonan['distance'], bins)
    binned_df = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                 observed=False)[['double_diff', 'requirement']]
    
    # Get binned validation table
    validation = pd.DataFrame([])
    validation['total_count[#]'] = binned_df.apply(lambda x: np.ma.masked_invalid(x['double_diff']).count())
    validation['passed_req.[#]'] = binned_df.apply(lambda x: np.sum(x['requirement']))

    # Add total at the end
    validation = pd.concat([validation, pd.DataFrame(validation.sum(axis=0)).T])
    validation['passed_pc'] = validation['passed_req.[#]'] / validation['total_count[#]']
    validation['success_fail'] = validation['passed_pc'] > threshold
    validation.index.name = 'distance[km]'

    # Rename last row
    validation.rename({validation.iloc[-1].name:'Total'}, inplace=True)

    # Figure
    fig, ax = plt.subplots(1, figsize=(9, 3), layout="none", dpi=200)

    # Plot residuals
    ms = 8 if pair_difference.shape[0] < 1e4 else 0.3
    alpha = 0.6 if pair_difference.shape[0] < 1e4 else 0.2
    ax.scatter(df_nonan.distance, df_nonan.double_diff,
               color='black', s=ms, zorder=1, alpha=alpha, edgecolor='None')

    distance_envelope = np.linspace(*distance_rqmt, num=100)
    requirement_envelope = requirement(distance_envelope)
    ax.fill_between(distance_envelope, 0, requirement_envelope, color='#e6ffe6', zorder=0, alpha=0.6)
    ax.fill_between(distance_envelope, requirement_envelope, 51, color='#ffe6e6', zorder=0, alpha=0.6)
    ax.vlines(bins, 0, 51, linewidth=0.3, color='gray', zorder=1)
    ax.plot(distance_envelope, requirement_envelope, color='k', linestyle='--', zorder=3)

    # Bar plot for each bin
    quantile_th = binned_df.quantile(q=threshold)['double_diff'].values
    for bin_center, quantile, flag in zip(bin_centers,
                                          quantile_th,
                                          validation['success_fail']):
        if flag:
            color = '#227522'
        else:
            color = '#7c1b1b'
        ax.bar(bin_center, quantile, align='center', width=np.diff(bins)[0],
               color='None', edgecolor=color, linewidth=2, zorder=3)

    # Add legend with data info
    legend_kwargs = dict(transform=ax.transAxes, verticalalignment='top')
    props = dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0.4)
    textstr = f"Sensor: {sensor} \n{validation_data}-InSAR point pairs\n"
    textstr += f"Record: {start_date}-{end_date}"

    # Place a text box in upper left in axes coords
    ax.text(0.02, 0.95, textstr, fontsize=8, bbox=props, **legend_kwargs)

    # Add legend with validation info
    textstr = f"{validation_type.capitalize()} requirement\n"
    textstr += f"Site: {site_name}\n"
    if validation.loc['Total']['success_fail']:
        validation_flag = 'PASSED'
        validation_color = '#239d23'
    else:
        validation_flag ='FAILED'
        validation_color = '#bc2e2e'

    props = {**props, **{'facecolor':'none', 'edgecolor':'none'}}
    ax.text(0.818, 0.93, textstr, fontsize=8, bbox=props, **legend_kwargs)
    ax.text(0.852, 0.82,  f"{validation_flag}",
            fontsize=10, weight='bold',
            bbox=props, **legend_kwargs)

    rect = patches.Rectangle((0.8, 0.75), 0.19, 0.2,
                             linewidth=1, edgecolor='black',
                             facecolor=validation_color,
                             transform=ax.transAxes)
    ax.add_patch(rect)

    # Title & labels
    fig.suptitle(f"{validation_type.capitalize()} requirement: {site_name}", fontsize=10)
    ax.set_xlabel("Distance (km)", fontsize=8)
    if validation_data == 'GNSS':
        txt = "Double-Differenced \nVelocity Residual (mm)"
    else:
        txt = "Relative Velocity measurement (mm)"
    ax.set_ylabel(txt, fontsize=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, direction='in', top=False, width=1.5)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xticks(bin_centers, minor=True)
    ax.set_xticks(np.arange(0, 55, 5))
    ax.set_ylim(0, 50)
    ax.set_xlim(*distance_rqmt)

    validation = validation.rename(columns={'success_fail': f'passed_req [>{threshold*100:.1f}%]'})
    
    return validation, fig


def display_transient_validation(pair_distances: list, pair_differences: list, ifgs_dates: list,
                                 site_name: str, distance_rqmt: list = [0.1, 50],
                                 n_bins: int = 10, threshold: float = 0.683, sensor: str = 'Sentinel-1',
                                 validation_data: str = 'GNSS'):
    
    
    '''
    Parameters:
      pair_distances : array     - lis of  1d array of pair distances used in validation
      pair_differences : array   - list of 1d array 0f pair double differenced displacement residuals
      site_name : str            - name of the cal/val site
      start_date  : str          - data record start date, eg. 20190101
      end_date : str             - data record end date, eg. 20200101
      distance_rqmt : list       - distance over requirement is tested, eg. length scales of 0.1-50 km
      n_bins : int               - number of bins
      threshold : float          - threshold represents percentile of Gaussian normal distribution
                                    within residuals are expected to be to pass the test
                                    e.g. 0.683 for 68.3% or 1-sigma limit 
      sensor : str               - sensor used in validation, e.g Sentinel-1 or NISAR
      validation_data : str      - data used to validate against; GNSS or INSAR

   Return
      validation_table : styled_df
      validation_figure : fig
    '''
    validation_type = 'Transient'
    maxY=80 ## Y limit in the subplot
    
    n_ifgs = len(pair_distances) ## Number of interferograms to validate
    
    
    # Data frame initialization
    bins = np.linspace(*distance_rqmt, num=n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    columns = [f'{bins[i]:.2f}-{bins[i + 1]:.2f}' for i in range(n_bins)] + ['Total']
    index = [f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}" for start, end in ifgs_dates]
    n_all = np.zeros([n_ifgs, n_bins + 1], dtype=int)
    n_pass = np.zeros([n_ifgs, n_bins + 1], dtype=int)
    
    ##  requirements per interferogram and each interferogram per bin
    for i in range(n_ifgs):
        inds = np.digitize(pair_distances[i], bins)
        for j in range(1, n_bins + 1):
            mask = inds == j
            rem = np.abs(pair_differences[i][mask])
            rqmt = 3 * (1 + np.sqrt(bins[j - 1]))
            n_all[i, j - 1] = len(rem)
            n_pass[i, j - 1] = np.sum(rem < rqmt)
        n_all[i, -1] = np.sum(n_all[i, :-1])
        n_pass[i, -1] = np.sum(n_pass[i, :-1])

    # Calculation of ratios and success/failure
    ratio = n_pass / np.where(n_all > 0, n_all, 1)
    success_or_fail = ratio > threshold
    
    # Creation of DataFrames for validation table
    n_all_pd = pd.DataFrame(n_all, columns=columns, index=index)
    n_pass_pd = pd.DataFrame(n_pass, columns=columns, index=index)
    ratio_pd = pd.DataFrame(ratio, columns=columns, index=index)
    success_or_fail_str = pd.DataFrame(success_or_fail.astype(str), columns=columns, index=index)

    # # Styling the DataFrame
    def style_specific_cells(val):
        color = '#e6ffe6' if val > threshold else '#ffe6e6'
        return f'background-color: {color}'

    # Apply style to all cells, and bold for 'Total' column
    styled_df = (ratio_pd.style.applymap(style_specific_cells)
                 .apply(lambda x: ['font-weight: bold' if x.name == 'Total' else '' for _ in x], axis=0))

    
    # Start subplot, each subplot represent validation test per interferogram 
    num_cols = 3 ## Can be changed to adjust subplot figure
    num_rows = (n_ifgs + num_cols - 1) // num_cols  # Calculate number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4*num_rows))
    
    axs = np.array(axs).reshape(num_rows, num_cols)

    for i in range(0,n_ifgs):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        ## Data frame for validation of each interferogram
        df = pd.DataFrame(np.vstack([pair_distances[i],
                                     pair_differences [i]]).T,
                                     columns=['distance', 'double_diff']) 
        # remove nans, draw bins and group by distance
        df_nonan = df.dropna(subset=['double_diff'])
        bins = np.linspace(*distance_rqmt, num=n_bins+1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        binned_df = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                     observed=False)[['double_diff']]
        validation = pd.DataFrame(data={
            'total_count[#]': n_all[i],
            'passed_req.[#]': n_pass[i],
        })
        validation['passed_pc'] = validation['passed_req.[#]'] / validation['total_count[#]']
        validation['success_fail'] = validation['passed_pc'] > threshold
        validation.index.name = 'distance[km]'
        validation.rename({validation.iloc[-1].name: 'Total'}, inplace=True)
        # start scatter plot 
        ms = 8 if len(pair_differences[i]) < 1e4 else 0.3
        alpha = 0.6 if len(pair_differences[i]) < 1e4 else 0.2

        ax.scatter(pair_distances[i], pair_differences[i],
                   color='black', s=ms, zorder=1, alpha=alpha, edgecolor='None')

        # Plot validation requirement log fit 
        dist_th = np.linspace(min(pair_distances[i]), max(pair_distances[i]), 100)
        acpt_error = 3 * (1 + np.sqrt(dist_th))
        ax.plot(dist_th, acpt_error, 'r')

        # Vertical lines for bins
        ax.vlines(bins, 0, maxY, linewidth=0.3, color='gray', zorder=1)

        # Bar plot for each bin
        quantile_th = binned_df.quantile(q=threshold)['double_diff'].values
        for bin_center, quantile, flag in zip(bin_centers,
                                              quantile_th,
                                              validation['success_fail']):
            if flag:
                color = '#227522'
            else:
                color = '#7c1b1b'
            ax.bar(bin_center, quantile, align='center', width=np.diff(bins)[0],
                   color='None', edgecolor=color, linewidth=2, zorder=3)
        # Add legend with data info
        legend_kwargs = dict(transform=ax.transAxes, verticalalignment='top')
        props = dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0.4)
        textstr = f'Sensor: {sensor} \n{validation_data}-InSAR point pairs\n'
        textstr +=   f"Record: {index[i]}"

        # Place a text box in upper left in axes coords
        ax.text(0.02, 0.95, textstr, fontsize=7, bbox=props, **legend_kwargs)

        # Add legend with validation info
        textstr = f'{validation_type.capitalize()} Req \n'
        textstr += f'Site: {site_name}\n'
        if validation.loc['Total']['success_fail']:
            validation_flag = 'PASSED'
            validation_color = '#239d23'
        else:
            validation_flag ='FAILED'
            validation_color = '#bc2e2e'

        props = {**props, **{'facecolor':'none', 'edgecolor':'none'}}
        ax.text(0.818, 0.93, textstr, fontsize=8, bbox=props, **legend_kwargs)
        ax.text(0.852, 0.82,  f"{validation_flag}",
                fontsize=10, weight='bold',
                bbox=props, **legend_kwargs)

        # Add colored rectangle indicating validation status
        rect = patches.Rectangle((0.8, 0.75), 0.19, 0.2,
                                 linewidth=1, edgecolor='black',
                                 facecolor=validation_color,
                                 transform=ax.transAxes)
        ax.add_patch(rect)

        # Title & labels
        
        ax.set_xlabel("Distance (km)", fontsize=8)
        if validation_data == 'GNSS':
            txt = "Double-Differenced \n Displacement Residual (mm)"
        else:
            txt = "Relative Velocity measurement (mm/yr)"    
        ax.set_ylabel(txt, fontsize=8)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', length=4, direction='in', top=False, width=1.5)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xticks(bin_centers, minor=True)
        ax.set_xticks(np.arange(0, 55, 5))
        ax.set_ylim(0, maxY)
        ax.set_xlim(*distance_rqmt)
        ax.set_title(f"Residuals \n Date range {index[i]} \n Number of station pairs used: {len(pair_distances[i])} \n Cal/Val Site Los Angeles " )
        
    # Hide unused subplots if there are any
    for idx in range(n_ifgs, num_rows*num_cols):
        axs.flat[idx].axis('off')  
    # figure title
    fig.suptitle(f"{validation_type.capitalize()} requirement for site : {site_name} \n", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    ## In case we want to save the figure
    # plt.savefig(f'transient_validation_{index[i]}.png', bbox_inches='tight', transparent=True)
    
    plt.close()
    return styled_df, fig  

