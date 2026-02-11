"""
Functions to plot stellar SEDs with filter transmission curves and ice absorption.
"""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astroquery.svo_fps import SvoFps
from dust_extinction.averages import CT06_MWGC

from icemodels import atmo_model, absorbed_spectrum
from icemodels.core import composition_to_molweight


def plot_stellar_seds(temperatures, filters, xarr=None, ice_model_table=None, 
                      ice_column=None, molecular_weight=None, ice_labels=None,
                      figsize=None, color_cycle=None, show_ice_absorbed=True,
                      renormalize_insets=False, extinction_Av=None, 
                      extinction_curve=None, fig=None, axes=None, label=None,
                      label_filters=False):
    """
    Plot stellar SEDs at multiple temperatures with filter transmission profiles.
    
    Creates a figure with one large plot across the top showing the full SEDs,
    and N small plots in a bottom row (one per filter) showing zoomed-in views
    of each filter's wavelength range with the transmission profile superposed.
    
    Parameters
    ----------
    temperatures : float or array-like
        Stellar effective temperature(s) in Kelvin. Can be a single value or 
        array/list of multiple temperatures to overlay.
    filters : list of str
        List of filter IDs (e.g., ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']).
        Each filter will get its own zoom subplot.
    xarr : astropy.units.Quantity, optional
        Wavelength array for computing the stellar SED. If None, uses a default
        range from 0.6 to 28 microns with 25000 points.
    ice_model_table : astropy.table.Table or list of Tables, optional
        Ice optical constants table(s) with 'Wavelength' and 'k' columns and 
        'density' in metadata. Can be a single table or list of tables for
        multiple ice species. Required if showing ice-absorbed SEDs.
    ice_column : float/Quantity or list of float/Quantity, optional
        Ice column density in molecules/cm². Can be a single value or list
        (one per ice species). If list, must match length of ice_model_table list.
        Can be plain numbers or Quantities with units (e.g., 1e19*u.cm**-2).
    molecular_weight : astropy.units.Quantity or list of Quantity, optional
        Molecular weight(s) of the ice composition (e.g., 44*u.Da for CO2).
        Can be a single value or list (one per ice species).
        Required if ice_column is provided.
    ice_labels : str or list of str, optional
        Label(s) for ice species in legend (e.g., 'CO2', 'CO'). If None,
        tries to extract from ice table metadata.
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, auto-calculated based
        on the number of filters.
    color_cycle : list, optional
        List of colors to use for different temperatures. If None, uses default
        matplotlib color cycle.
    show_ice_absorbed : bool, optional
        Whether to show ice-absorbed SEDs if ice parameters are provided.
        Default is True.
    renormalize_insets : bool, optional
        If True, adjust the y-axis limits of each inset plot to match the
        data range within that filter's wavelength range (ymin=datamin, ymax=datamax).
        If False (default), use automatic y-axis limits. This increases the
        dynamic range and makes absorption features more visible without
        changing the data normalization.
    extinction_Av : float, optional
        Visual extinction A_V in magnitudes. If provided, extinction will be
        applied to the stellar SEDs. Default is None (no extinction).
    extinction_curve : dust_extinction model instance, optional
        Extinction curve model from dust_extinction package (e.g., CT06_MWGC(), CCM89()).
        If None and extinction_Av is provided, defaults to CT06_MWGC() (Chiar & Tielens 2006).
        Common values: CT06_MWGC(), CCM89(), F99(), etc.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, a new figure is created.
    axes : array of matplotlib.axes.Axes, optional
        Existing axes to plot on: [main_ax, zoom_ax1, zoom_ax2, ...].
        Must match the number of filters. If None, new axes are created.
    label : str, optional
        Label prefix for the legend entries. Useful when calling multiple times
        on the same axes to distinguish different scenarios (e.g., 'No extinction',
        'With extinction'). If None, uses temperature only.
    label_filters : bool, optional
        If True, add text labels for each filter near the top of the main plot
        above their grey shaded regions. Default is False.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    axes : array of matplotlib.axes.Axes
        Array of axes: [main_ax, zoom_ax1, zoom_ax2, ...]
        
    Examples
    --------
    Plot SEDs at 3000K and 5000K for two filters:
    
    >>> temperatures = [3000, 5000]
    >>> filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']
    >>> fig, axes = plot_stellar_seds(temperatures, filters)
    
    Plot with ice absorption:
    
    >>> from icemodels import read_ocdb_file, optical_constants_cache_dir
    >>> ice_file = f'{optical_constants_cache_dir}/55_CO2_(1)_25K_Gerakines.txt'
    >>> ice_table = read_ocdb_file(ice_file)
    >>> fig, axes = plot_stellar_seds(
    ...     temperatures=4000,
    ...     filters=['JWST/NIRCam.F212N'],
    ...     ice_model_table=ice_table,
    ...     ice_column=1e19*u.cm**-2,
    ...     molecular_weight=44*u.Da
    ... )
    """
    # Ensure temperatures is iterable
    if np.isscalar(temperatures):
        temperatures = [temperatures]
    
    # Default wavelength array
    if xarr is None:
        xarr = np.linspace(0.6*u.um, 28.0*u.um, 25000)
    
    # Set up extinction if requested
    if extinction_Av is not None:
        if extinction_curve is None:
            # Default to CT06_MWGC (Chiar & Tielens 2006)
            extinction_curve = CT06_MWGC()
        # Extinction curves expect x = 1/wavelength in units of 1/micron
        # and return A(lambda)/A(V)
        x = 1.0 / xarr.to(u.um).value  # inverse wavelength in 1/micron
        # Only evaluate where x is in valid range
        x_range = extinction_curve.x_range
        valid = (x >= x_range[0]) & (x <= x_range[1])
        ext_alambda_av = np.zeros_like(x)
        if np.any(valid):
            ext_alambda_av[valid] = extinction_curve(x[valid] / u.um)
        # Convert A(lambda)/A(V) to A(lambda) by multiplying by A_V
        ext_mag = ext_alambda_av * extinction_Av
    else:
        ext_mag = None
    
    # Normalize ice parameters to lists
    if ice_column is not None and show_ice_absorbed:
        if ice_model_table is None:
            raise ValueError("ice_model_table must be provided if ice_column is specified")
        
        # Convert single values to lists
        if not isinstance(ice_model_table, list):
            ice_model_table = [ice_model_table]
        if not isinstance(ice_column, list):
            ice_column = [ice_column]
        if molecular_weight is not None and not isinstance(molecular_weight, list):
            molecular_weight = [molecular_weight]
        if ice_labels is not None and not isinstance(ice_labels, list):
            ice_labels = [ice_labels]
        
        # Validate lengths
        n_ices = len(ice_model_table)
        if len(ice_column) == 1 and n_ices > 1:
            ice_column = ice_column * n_ices
        elif len(ice_column) != n_ices:
            raise ValueError(f"ice_column length ({len(ice_column)}) must match "
                           f"ice_model_table length ({n_ices})")
        
        # Handle molecular weights
        if molecular_weight is None:
            molecular_weight = []
            for ice_table in ice_model_table:
                if 'composition' in ice_table.meta:
                    molecular_weight.append(u.Quantity(
                        composition_to_molweight(ice_table.meta['composition']), u.Da))
                else:
                    raise ValueError(
                        "molecular_weight must be provided if ice_column is specified "
                        "and not available in ice_model_table.meta['composition']")
        elif len(molecular_weight) == 1 and n_ices > 1:
            molecular_weight = molecular_weight * n_ices
        elif len(molecular_weight) != n_ices:
            raise ValueError(f"molecular_weight length ({len(molecular_weight)}) must match "
                           f"ice_model_table length ({n_ices})")
        
        # Handle ice labels
        if ice_labels is None:
            ice_labels = []
            for ice_table in ice_model_table:
                if 'composition' in ice_table.meta:
                    ice_labels.append(ice_table.meta['composition'])
                else:
                    ice_labels.append(f"Ice {len(ice_labels)+1}")
        elif len(ice_labels) != n_ices:
            raise ValueError(f"ice_labels length ({len(ice_labels)}) must match "
                           f"ice_model_table length ({n_ices})")
    
    # Get filter transmission data
    transdata = {}
    filter_ranges = {}
    for filt in filters:
        trans = SvoFps.get_transmission_data(filt)
        transdata[filt] = trans
        
        # Convert wavelength to microns if needed
        trans_wl = trans['Wavelength']
        if not hasattr(trans_wl, 'unit'):
            trans_wl = trans_wl * u.AA  # Assume Angstroms if no unit
        trans_wl = trans_wl.to(u.um)
        
        # Find wavelength range where transmission > 50% of peak
        max_trans = trans['Transmission'].max()
        half_max_mask = trans['Transmission'] > 0.5 * max_trans
        if np.any(half_max_mask):
            wl_range = trans_wl[half_max_mask]
            filter_ranges[filt] = (wl_range.min(), wl_range.max())
        else:
            # Fallback if no points above 50%
            filter_ranges[filt] = (trans_wl.min(), trans_wl.max())
    
    # Set up figure layout
    n_filters = len(filters)
    
    # Create or reuse figure and axes
    if fig is None or axes is None:
        if figsize is None:
            # Auto-size: width based on number of filters, reasonable height
            figsize = (4 + 3 * n_filters, 8)
        
        # Create figure with GridSpec for custom layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, n_filters, figure=fig, height_ratios=[2, 1], 
                      hspace=0.3, wspace=0.3)
        
        # Main plot spans entire top row
        ax_main = fig.add_subplot(gs[0, :])
        
        # Zoom plots in bottom row
        ax_zooms = [fig.add_subplot(gs[1, i]) for i in range(n_filters)]
    else:
        # Reuse existing axes
        ax_main = axes[0]
        ax_zooms = axes[1:]
        if len(ax_zooms) != n_filters:
            raise ValueError(f"Number of zoom axes ({len(ax_zooms)}) must match "
                           f"number of filters ({n_filters})")
    
    # Set up colors
    if color_cycle is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Store flux data for potential inset renormalization
    temp_flux_data = {}
    
    # Generate and plot SEDs for each temperature
    for i, temp in enumerate(temperatures):
        color = color_cycle[i % len(color_cycle)]
        
        # Generate stellar atmosphere model
        stellar_model = atmo_model(temp, xarr=xarr)
        wavelengths = xarr.to(u.um)
        fluxes = stellar_model['fnu'].quantity
        
        # Apply extinction if requested
        if ext_mag is not None:
            # Convert extinction magnitude to flux attenuation
            # flux_extincted = flux_original * 10^(-ext_mag/2.5)
            fluxes = fluxes * 10**(-ext_mag / 2.5)
        
        # Normalize flux for plotting (peak = 1)
        flux_norm = fluxes / fluxes.max()
        
        # Store flux data for inset plots
        temp_flux_data[temp] = {
            'wavelengths': wavelengths,
            'fluxes': fluxes,
            'flux_norm': flux_norm
        }
        
        # Plot on main axes
        if label is not None:
            temp_label = f"{label}: {temp:.0f} K"
        elif extinction_Av is not None:
            temp_label = f"{temp:.0f} K (Av={extinction_Av:.1f})"
        else:
            temp_label = f"{temp:.0f} K"
        
        # If ice absorption is requested, we'll plot the ice-absorbed spectrum instead
        # of the base (or extinguished-only) spectrum as the main curve
        if ice_column is not None and show_ice_absorbed:
            # Don't plot the non-ice-absorbed curve yet - wait for ice absorption below
            pass
        else:
            # No ice absorption - plot the spectrum (with or without extinction)
            ax_main.plot(wavelengths, flux_norm, color=color, label=temp_label, 
                         alpha=0.8, linewidth=1.5)
        
        # If ice absorption requested
        if ice_column is not None and show_ice_absorbed:
            # Plot each ice species
            linestyles = ['--', '-.', ':']
            for ice_idx, (ice_table, ice_col, mol_wt, ice_label) in enumerate(
                zip(ice_model_table, ice_column, molecular_weight, ice_labels)):
                
                absorbed_flux = absorbed_spectrum(
                    ice_column=ice_col,
                    ice_model_table=ice_table,
                    spectrum=fluxes,
                    xarr=xarr,
                    molecular_weight=mol_wt
                )
                absorbed_norm = absorbed_flux / fluxes.max()  # Normalize to same scale
                
                # Use dashed/dotted lines for ice-absorbed spectra
                linestyle = linestyles[ice_idx % len(linestyles)]
                ax_main.plot(wavelengths, absorbed_norm, color=color, 
                            linestyle=linestyle, alpha=0.6, linewidth=1.5,
                            label=f"{temp:.0f} K ({ice_label})")
                
                # Store ice-absorbed flux for inset plots
                if f'ice_{ice_idx}' not in temp_flux_data[temp]:
                    temp_flux_data[temp][f'ice_{ice_idx}'] = []
                temp_flux_data[temp][f'ice_{ice_idx}'].append({
                    'absorbed_flux': absorbed_flux,
                    'absorbed_norm': absorbed_norm
                })
        
        # Plot on zoom axes
        for j, filt in enumerate(filters):
            ax_zoom = ax_zooms[j]
            
            # Get wavelength range for this filter
            wl_min, wl_max = filter_ranges[filt]
            wl_min_val = wl_min.value if hasattr(wl_min, 'value') else wl_min
            wl_max_val = wl_max.value if hasattr(wl_max, 'value') else wl_max
            
            # Plot stellar SED (no renormalization)
            ax_zoom.plot(wavelengths, flux_norm, color=color, alpha=0.8, 
                        linewidth=1.5)
            
            # Track data range for potential y-limit adjustment
            if renormalize_insets:
                in_range = (wavelengths.value >= wl_min_val) & (wavelengths.value <= wl_max_val)
                if np.any(in_range):
                    # Store data for later y-limit calculation
                    if not hasattr(ax_zoom, '_ylim_data'):
                        ax_zoom._ylim_data = []
                    ax_zoom._ylim_data.append(flux_norm[in_range])
            
            if ice_column is not None and show_ice_absorbed:
                linestyles = ['--', '-.', ':']
                for ice_idx, (ice_table, ice_col, mol_wt, ice_label) in enumerate(
                    zip(ice_model_table, ice_column, molecular_weight, ice_labels)):
                    
                    absorbed_flux = absorbed_spectrum(
                        ice_column=ice_col,
                        ice_model_table=ice_table,
                        spectrum=fluxes,
                        xarr=xarr,
                        molecular_weight=mol_wt
                    )
                    absorbed_norm = absorbed_flux / fluxes.max()
                    
                    linestyle = linestyles[ice_idx % len(linestyles)]
                    ax_zoom.plot(wavelengths, absorbed_norm, color=color, 
                               linestyle=linestyle, alpha=0.6, linewidth=1.5)                    
                    # Track data range for potential y-limit adjustment
                    if renormalize_insets:
                        if np.any(in_range):
                            ax_zoom._ylim_data.append(absorbed_norm[in_range])    
    # Add filter transmission curves to zoom plots (only if creating new axes)
    # Check if axes are new by seeing if they have lines already
    axes_are_new = axes is None
    for j, filt in enumerate(filters):
        ax_zoom = ax_zooms[j]
        trans = transdata[filt]
        
        # Convert wavelength to microns if needed
        trans_wl = trans['Wavelength']
        if not hasattr(trans_wl, 'unit'):
            trans_wl = trans_wl * u.AA  # Assume Angstroms if no unit
        trans_wl = trans_wl.to(u.um)
        
        # Normalize transmission to 0-1 range
        trans_norm = trans['Transmission'] / trans['Transmission'].max()
        
        # Plot transmission on secondary y-axis (only if axes are new)
        if axes_are_new:
            ax_trans = ax_zoom.twinx()
            ax_trans.fill_between(trans_wl.value, 0, trans_norm, 
                                 color='gray', alpha=0.3, label='Transmission')
            # Only show transmission label on rightmost plot
            if j == len(filters) - 1:
                ax_trans.set_ylabel('Transmission', fontsize=8, color='gray')
            ax_trans.tick_params(axis='y', labelsize=7, colors='gray')
            ax_trans.set_ylim(0, 1.1)
        
        # Set zoom range to filter coverage (only if axes are new)
        if axes_are_new:
            wl_min, wl_max = filter_ranges[filt]
            # Handle both Quantity and plain values
            wl_min_val = wl_min.value if hasattr(wl_min, 'value') else wl_min
            wl_max_val = wl_max.value if hasattr(wl_max, 'value') else wl_max
            padding = 0.1 * (wl_max_val - wl_min_val)
            ax_zoom.set_xlim(wl_min_val - padding, wl_max_val + padding)
        
        # Adjust y-limits if renormalize_insets is True
        if renormalize_insets and hasattr(ax_zoom, '_ylim_data') and ax_zoom._ylim_data:
            # Combine all data for this inset
            all_data = np.concatenate(ax_zoom._ylim_data)
            ymin = np.min(all_data)
            ymax = np.max(all_data)
            # Add small padding (2% of range)
            yrange = ymax - ymin
            ax_zoom.set_ylim(ymin - 0.02*yrange, ymax + 0.02*yrange)
        
        # Format zoom plot (only if axes are new)
        if axes_are_new:
            filter_name = filt.split('.')[-1] if '.' in filt else filt
            ax_zoom.set_title(filter_name, fontsize=10)
            ax_zoom.set_xlabel('Wavelength (μm)', fontsize=8)
            ax_zoom.tick_params(labelsize=7)
            
            # Show y-labels on all plots if renormalizing, otherwise only on leftmost
            if renormalize_insets:
                ax_zoom.set_ylabel('Normalized Flux', fontsize=8)
            elif j == 0:
                ax_zoom.set_ylabel('Normalized Flux', fontsize=8)
            else:
                ax_zoom.set_yticklabels([])
    
    # Format main plot (only if axes are new)
    if axes_are_new:
        ax_main.set_xlabel('Wavelength (μm)', fontsize=12)
        ax_main.set_ylabel('Normalized Flux', fontsize=12)
        ax_main.set_title('Stellar SEDs with Filter Coverage', fontsize=14, fontweight='bold')
        #ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(wavelengths.value.min(), wavelengths.value.max())
        
        # Add filter transmission profiles on main plot
        for filt in filters:
            trans = transdata[filt]
            
            # Convert wavelength to microns if needed
            trans_wl = trans['Wavelength']
            if not hasattr(trans_wl, 'unit'):
                trans_wl = trans_wl * u.AA  # Assume Angstroms if no unit
            trans_wl = trans_wl.to(u.um)
            
            # Normalize transmission to 0-1 range
            trans_norm = trans['Transmission'] / trans['Transmission'].max()
            
            # Plot transmission profile, normalized to the main plot height
            ax_main.fill_between(trans_wl.value, 0, trans_norm, 
                               color='gray', alpha=0.15, linewidth=0)
            
            # Add filter labels if requested
            if label_filters:
                # Extract filter name (e.g., 'F212N' from 'JWST/NIRCam.F212N')
                filter_name = filt.split('.')[-1] if '.' in filt else filt
                # Position label at center of filter range, near top of plot
                wl_min, wl_max = filter_ranges[filt]
                wl_min_val = wl_min.value if hasattr(wl_min, 'value') else wl_min
                wl_max_val = wl_max.value if hasattr(wl_max, 'value') else wl_max
                center_wl = (wl_min_val + wl_max_val) / 2
                # Get current y-axis limits to position text near top
                ymin, ymax = ax_main.get_ylim()
                y_pos = ymax * 0.95  # 95% of the way to the top
                ax_main.text(center_wl, y_pos, filter_name, 
                           ha='center', va='top', fontsize=9, 
                           color='gray', alpha=0.7)
    
    # Update legend (always, to include new lines)
    ax_main.legend(loc='best', fontsize=9)
    
    if axes_are_new:
        plt.tight_layout()
    
    # Return figure and all axes
    all_axes = [ax_main] + list(ax_zooms)
    return fig, np.array(all_axes)
