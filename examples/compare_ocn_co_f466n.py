"""
Compare OCN- (Gerakines+25 band-strength model) vs pure CO ice in their
effect on the JWST/NIRCam F405N - F466N color.

OCN- absorbs at 4.608 um with A' = 1.51e-16 cm molecule^-1, ~13x stronger
per molecule than the CO 4.67 um fundamental. Result: at the same N, OCN-
produces a much larger blueing of [F405N] - [F466N].
"""
import os
import glob

import numpy as np
import astropy.units as u
from astropy.table import Table
from astroquery.svo_fps import SvoFps

from icemodels import (atmo_model, absorbed_spectrum, fluxes_in_filters,
                       read_ocdb_file, optical_constants_cache_dir)
from icemodels.ocn_gerakines25 import make_ocn_table_synthetic


# Build OCN- table if not cached.
ocn_path = os.path.join(optical_constants_cache_dir,
                        'OCN-_Gerakines2025_synthetic.ecsv')
if os.path.exists(ocn_path):
    ocn_tbl = Table.read(ocn_path)
else:
    ocn_tbl = make_ocn_table_synthetic(save_path=ocn_path)

co_tbl = read_ocdb_file(
    glob.glob(f'{optical_constants_cache_dir}/*_CO_(1)_*K_Gerakines*.txt')[0])

xarr = np.linspace(3.5 * u.um, 5.2 * u.um, 8000)
stellar = atmo_model(4500, xarr=xarr, logg=2.5)
fnu0 = stellar['fnu'].quantity

cf = ['JWST/NIRCam.F405N', 'JWST/NIRCam.F466N']
jf = SvoFps.get_filter_list('JWST')
jf.add_index('filterID')
zp = {f: float(jf.loc[f]['ZeroPoint']) for f in cf}
td = {f: SvoFps.get_transmission_data(f) for f in cf}


def f405_f466(fnu):
    flx = fluxes_in_filters(xarr, fnu, filterids=cf, transdata=td)
    m = {f: -2.5 * np.log10(flx[f] / u.Quantity(zp[f], u.Jy)) for f in cf}
    return float((m[cf[0]] - m[cf[1]]).value)


print(f'4500K baseline:  [F405N] - [F466N] = {f405_f466(fnu0):+.3f}')
print()
print(f'{"N (cm^-2)":>12s}  {"CO":>8s}  {"OCN-":>8s}  ratio')
for N in [1e16, 3e16, 1e17, 3e17, 1e18, 3e18]:
    fnu_co = absorbed_spectrum(
        ice_column=N * u.cm**-2, ice_model_table=co_tbl,
        spectrum=fnu0, xarr=xarr, molecular_weight=28 * u.Da)
    fnu_ocn = absorbed_spectrum(
        ice_column=N * u.cm**-2, ice_model_table=ocn_tbl,
        spectrum=fnu0, xarr=xarr,
        molecular_weight=ocn_tbl.meta['molwt'])
    c_co = f405_f466(fnu_co)
    c_ocn = f405_f466(fnu_ocn)
    base = -0.101
    print(f'{N:>12.0e}  {c_co:>+8.3f}  {c_ocn:>+8.3f}  '
          f'{(c_ocn-base)/(c_co-base):>5.1f}')
