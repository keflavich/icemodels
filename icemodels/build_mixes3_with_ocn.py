"""
Build the ``mixes3`` set: each composition in ``mixes2`` augmented with OCN-
at 5 per cent abundance wrt H2O, using the Gerakines+2025 OCN- optical
constants (``ocn_gerakines25.make_ocn_table_synthetic``).

Steps
-----
1. Generate or load the Gerakines+25 OCN- (n,k) table.
2. Build a moltbls dict that overrides the legacy LIDA OCN entry with the
   new model.
3. For each mixes2 composition, append :OCN with multiplier 0.05*water,
   write the new mix table (k, n) to icemodels/data/mymixes/.
4. Run process_table on the new mix tables to compute filter magnitudes
   per column, on the same column grid as the existing combined dmag table.
5. Append the new rows to combined_ice_absorption_tables.ecsv (preserving
   any existing rows).

Run as a module:
    python -m icemodels.build_mixes3_with_ocn
"""
import os

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astroquery.svo_fps import SvoFps

from icemodels.core import (
    optical_constants_cache_dir, read_ocdb_file, retrieve_gerakines_co,
    load_molecule_univap,
)
from icemodels.absorbance_in_filters import (
    make_mixtable, process_table, get_phx4000, xarr, cols, cmd_x_default,
)
from icemodels.ocn_gerakines25 import make_ocn_table_synthetic


# Same compositions as colorcolordiagrams.mixes2 (kept duplicated here so
# this builder is self-contained).
MIXES2 = [
    'H2O:CO:CO2 (1:1:1)',
    'H2O:CO:CO2 (2:1:1)',
    'H2O:CO:CO2 (3:1:1)',
    'H2O:CO:CO2 (5:1:1)',
    'H2O:CO:CO2 (10:1:1)',
    'H2O:CO:CO2 (10:1:0.5)',
    'H2O:CO:CO2 (15:1:1)',
    'H2O:CO:CO2 (20:1:1)',
    'H2O:CO:CO2:CH3OH (1:1:1:1)',
    'H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:1:1:1)',
]

OCN_FRAC_VS_H2O = 0.05  # 5 per cent of H2O


def _add_ocn_to(comp, frac=OCN_FRAC_VS_H2O):
    """Return composition string with :OCN appended at frac * (H2O part)."""
    mols = comp.split(' ')[0].split(':')
    parts = [float(p) for p in comp.split(' ')[1].strip('()').split(':')]
    if mols[0] != 'H2O':
        raise ValueError(f"first molecule of '{comp}' is not H2O")
    ocn_mult = frac * parts[0]
    new_mols = mols + ['OCN']
    new_parts = parts + [ocn_mult]
    parts_str = ':'.join(_fmt_number(p) for p in new_parts)
    return f"{':'.join(new_mols)} ({parts_str})"


def _fmt_number(x):
    """Format float compactly for composition strings."""
    if x == int(x):
        return str(int(x))
    return f"{x:g}"


def _moltbls_with_gerakines_ocn():
    """Build the molecule-table dict matching make_mymix_tables but with the
    Gerakines+25 OCN- model substituted for the LIDA OCN- entry."""
    water = read_ocdb_file(
        f'{optical_constants_cache_dir}/240_H2O_(1)_25K_Mastrapa.txt')
    co2 = read_ocdb_file(
        f'{optical_constants_cache_dir}/55_CO2_(1)_8K_Gerakines.txt')
    ethanol = load_molecule_univap('ethanol')
    methanol = load_molecule_univap('methanol')
    co = retrieve_gerakines_co()

    ocn_path = os.path.join(optical_constants_cache_dir,
                            'OCN-_Gerakines2025_synthetic.ecsv')
    if os.path.exists(ocn_path):
        ocn = Table.read(ocn_path)
    else:
        ocn = make_ocn_table_synthetic(save_path=ocn_path)
    # make_mixtable indexes the ice-table composition string to compute
    # the per-molecule molwt; for OCN- the cleanest alias is "OCN 1".
    ocn.meta['composition'] = 'OCN 1'

    return {'CO': co, 'H2O': water, 'CO2': co2,
            'CH3OH': methanol, 'CH3CH2OH': ethanol, 'OCN': ocn}


def make_mixes3_tables():
    """Build n,k tables for every mixes3 composition; save under
    icemodels/data/mymixes/. Returns dict {(grouping, index, T): Table}."""
    moltbls = _moltbls_with_gerakines_ocn()
    authors = {mol: tb.meta.get('author', 'unknown')
               for mol, tb in moltbls.items()}
    out = {}
    out_dir = f'{optical_constants_cache_dir}/mymixes'
    os.makedirs(out_dir, exist_ok=True)

    for ii, comp2 in enumerate(MIXES2):
        comp3 = _add_ocn_to(comp2)
        author_tag = ', '.join(authors[m]
                               for m in comp3.split(' ')[0].split(':'))
        tbl = make_mixtable(
            comp3, moltbls, grid=xarr,
            density=1.0 * u.g / u.cm**3, temperature=25 * u.K,
            authors=author_tag, index=10000 + ii)
        # Tag for downstream grouping (mirrors make_mymix_tables convention).
        mol_key = ''.join(['plus' + m if i else m
                           for i, m in enumerate(comp3.split(' ')[0].split(':'))])
        out[(mol_key, 10000 + ii, 25)] = tbl
        save_path = f'{out_dir}/{comp3.replace(" ", "_")}.ecsv'
        tbl.write(save_path, overwrite=True)
        print(f'  wrote {save_path}', flush=True)
    return out


def append_mixes3_to_dmag(combined_path=None):
    """Run process_table on the mixes3 mix tables and append rows to the
    combined ice absorption table at ``combined_path``."""
    if combined_path is None:
        basepath = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        combined_path = os.path.join(
            basepath, 'icemodels', 'data',
            'combined_ice_absorption_tables.ecsv')

    print('building mixes3 tables ...', flush=True)
    tables = make_mixes3_tables()

    print('preparing filter data ...', flush=True)
    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')
    cmd_x = cmd_x_default
    filter_data = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in cmd_x}
    transdata = {fid: SvoFps.get_transmission_data(fid) for fid in cmd_x}

    phx4000 = get_phx4000()
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print('computing dmag rows for mixes3 ...', flush=True)
    new_rows = []
    for key, consts in tables.items():
        args = (key[0], key, consts, xarr, phx4000, cols,
                filter_data, transdata, basepath)
        rows = process_table(args, cmd_x=cmd_x, transdata=transdata)
        new_rows.extend(rows)
        print(f'  {key[0]} index={key[1]}: {len(rows)} rows', flush=True)

    new_tbl = Table(new_rows)
    print(f'new mixes3 rows: {len(new_tbl)}', flush=True)

    if os.path.exists(combined_path):
        existing = Table.read(combined_path)
        print(f'existing combined: {len(existing)} rows', flush=True)
        # Drop any prior mixes3 rows so re-running is idempotent.
        is_mixes3 = np.asarray([str(c) in MIXES3_COMPOSITIONS
                                for c in existing['composition']])
        if is_mixes3.any():
            print(f'  removing {int(is_mixes3.sum())} prior mixes3 rows',
                  flush=True)
            existing = existing[~is_mixes3]
        combined = vstack([existing, new_tbl])
    else:
        combined = new_tbl

    combined.write(combined_path, overwrite=True)
    print(f'wrote {combined_path}: {len(combined)} rows total', flush=True)
    return combined


MIXES3_COMPOSITIONS = [_add_ocn_to(c) for c in MIXES2]


if __name__ == '__main__':
    append_mixes3_to_dmag()
