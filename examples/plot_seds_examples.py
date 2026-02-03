"""
Example script demonstrating the plot_stellar_seds function.
"""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
from icemodels import plot_stellar_seds, optical_constants_cache_dir, read_ocdb_file, download_all_ocdb

# Example 1: Basic plot with multiple temperatures
print("Example 1: Basic stellar SEDs at multiple temperatures")
temperatures = [3000, 4000, 5000, 6000]
filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W', 'JWST/MIRI.F1000W']

fig1, axes1 = plot_stellar_seds(
    temperatures=temperatures,
    filters=filters
)
plt.savefig('example_stellar_seds_basic.png', dpi=150, bbox_inches='tight')
print("Saved: example_stellar_seds_basic.png")
plt.close()

# Example 2: Single temperature with ice absorption
print("\nExample 2: Stellar SED with ice absorption")
try:
    # Ensure OCDB files are downloaded
    if not os.path.exists(optical_constants_cache_dir):
        download_all_ocdb()
    
    # Load CO2 ice optical constants from OCDB
    # Look for any pure CO2 file from Gerakines
    import glob
    co2_files = glob.glob(f'{optical_constants_cache_dir}/*_CO2_(1)_*K_Gerakines*.txt')
    if not co2_files:
        # Try downloading again
        download_all_ocdb()
        co2_files = glob.glob(f'{optical_constants_cache_dir}/*_CO2_(1)_*K_Gerakines*.txt')
    
    if co2_files:
        ice_file = co2_files[0]
        print(f"Using ice file: {ice_file}")
        ice_table = read_ocdb_file(ice_file)
        
        fig2, axes2 = plot_stellar_seds(
            temperatures=4000,
            filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W'],
            ice_model_table=ice_table,
            ice_column=1e19 * u.cm**-2,  # ice column density
            molecular_weight=44*u.Da
        )
        plt.savefig('example_stellar_seds_with_ice.png', dpi=150, bbox_inches='tight')
        print("Saved: example_stellar_seds_with_ice.png")
        plt.close()
    else:
        print("No CO2 ice files found - skipping ice absorption example")
except Exception as e:
    import traceback
    print(f"Could not create ice absorption example: {e}")
    traceback.print_exc()

# Example 3: Stellar SED with extinction
print("\nExample 3: Stellar SED with interstellar extinction")
try:
    from dust_extinction.averages import CT06_MWGC
    
    fig3, axes3 = plot_stellar_seds(
        temperatures=[3000, 4000, 5000],
        filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W', 'JWST/MIRI.F1000W'],
        extinction_Av=17.0,  # 17 magnitudes of visual extinction
        extinction_curve=CT06_MWGC()  # Chiar & Tielens 2006
    )
    plt.savefig('example_stellar_seds_extinction.png', dpi=150, bbox_inches='tight')
    print("Saved: example_stellar_seds_extinction.png")
    plt.close()
except ImportError:
    print("dust_extinction not available - skipping extinction example")

# Example 4: More filters to show layout flexibility
print("\nExample 4: Many filters")
many_filters = [
    'JWST/NIRCam.F182M',
    'JWST/NIRCam.F212N',
    'JWST/NIRCam.F300M',
    'JWST/NIRCam.F444W',
    'JWST/MIRI.F770W'
]

fig4, axes4 = plot_stellar_seds(
    temperatures=[3000, 5000],
    filters=many_filters,
    figsize=(18, 8)
)
plt.savefig('example_stellar_seds_many_filters.png', dpi=150, bbox_inches='tight')
print("Saved: example_stellar_seds_many_filters.png")
plt.close()

# Example 5: Extinction + Ice absorption combined
print("\nExample 5: Stellar SED with both extinction and ice absorption")
try:
    from dust_extinction.averages import CT06_MWGC
    
    if co2_files:
        ice_file = co2_files[0]
        ice_table = read_ocdb_file(ice_file)
        
        fig5, axes5 = plot_stellar_seds(
            temperatures=4000,
            filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W'],
            ice_model_table=ice_table,
            ice_column=1e19 * u.cm**-2,
            molecular_weight=44*u.Da,
            extinction_Av=17.0,
            extinction_curve=CT06_MWGC()
        )
        plt.savefig('example_stellar_seds_extinction_and_ice.png', dpi=150, bbox_inches='tight')
        print("Saved: example_stellar_seds_extinction_and_ice.png")
        plt.close()
    else:
        print("No CO2 ice files found - skipping combined example")
except ImportError:
    print("dust_extinction not available - skipping combined example")

print("\nAll examples completed successfully!")
