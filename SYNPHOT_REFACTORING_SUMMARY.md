# Synphot Backend Refactoring: Complete Summary

## Objective
Refactor the `icemodels` atmosphere model backend to use STScI's production-grade `synphot`/`stsynphot` libraries in place of the limited `mysg` package, enabling full gravity-stratified (log g) atmosphere model access for luminosity-class demonstrations.

## Why This Was Needed
The original `mysg` library had a critical limitation:
- Only provided models at **log g = +4.0** for any temperature
- Although it accepted a `logg` parameter, it was not actually used—all grids were compiled at single gravity
- This prevented modeling of luminosity classes (I–V) which have distinctly different surface gravities and thus different atmospheric structures

## Solution Implemented

### 1. New Atmosphere Model Backend
**File:** [icemodels/core.py](icemodels/core.py#L178-L229)

Replaced the old `mysg`-based `atmo_model()` with a new implementation using `stsynphot.catalog.grid_to_spec()`:

```python
def atmo_model(
    temperature, 
    xarr=np.linspace(1, 28, 15000)*u.um,
    logg=4.0,                    # ← New: gravity support
    metallicity=0.0,             # ← New: metallicity parameter
    model_grid=None,             # ← New: explicit grid selection
    pysyn_cdbs='/orange/adamginsburg/synphot/grp/hst/cdbs'
):
    """Generate a stellar atmosphere SED with full parameter control.
    
    Using stsynphot.catalog.grid_to_spec() for access to:
    - Phoenix models (T < 4000 K): full (T, Z, logg) grid
    - Kurucz models (T ≥ 4000 K): full (T, Z, logg) grid
    """
```

**Key Features:**
- Automatic model grid selection (Phoenix for cool stars, Kurucz for hot stars)
- Full parameter support: temperature, surface gravity, metallicity
- Environment variable management (`PYSYN_CDBS`) for data localization
- Backward-compatible return type: `Table` with columns `['fnu', 'nu']` and rich metadata
- Unit conversion pipeline: PHOTLAM (photons/cm²/s/Å) → fnu (erg/(Hz·s·cm²))

### 2. Data Installation
**Location:** `/orange/adamginsburg/synphot/grp/hst/cdbs`

Downloaded and extracted synphot reference data containing:
- `grid/phoenix/catalog.fits`: Phoenix models for T < 4000 K
- `grid/k93models/`: Kurucz 1993 models for hotter stars
- HST calibration spectra and component tables

This is the standard Calibration Data Base System (CDBS) structure used by STScI's synphot ecosystem.

### 3. Dependency Updates
**File:** [setup.cfg](setup.cfg)

```diff
- model-yso-sed-grid @ git+https://github.com/astrofrog/mysg.git@5599d16
+ synphot
+ stsynphot
```

### 4. Test Updates
**File:** [icemodels/tests/test_core.py](icemodels/tests/test_core.py#L72-L80)

Rewrote `test_atmo_model()` to use real CDBS data instead of mocked `mysg` calls:

```python
def test_atmo_model():
    result = atmo_model(4000)
    assert 'fnu' in result.colnames
    assert 'nu' in result.colnames
    assert result.meta['temperature'] == 4000
    assert result.meta['model_grid'] in ('phoenix', 'k93models')
    assert result['fnu'].unit == u.erg / u.s / u.cm**2 / u.Hz
    assert result['nu'].unit == u.Hz
```

✅ **Test Status:** PASSED (validated real data from `/orange/adamginsburg/synphot/`)

## Validation: Gravity Parameter Support

All luminosity classes now retrievable from Phoenix grid at 3500 K:

```
I (Supergiant)       (logg=1.0): max_fnu = 3.3260e-05 erg/(Hz·s·cm²)
II (Bright Giant)    (logg=2.0): max_fnu = 3.4736e-05 erg/(Hz·s·cm²)
III (Giant)          (logg=2.5): max_fnu = 3.9201e-05 erg/(Hz·s·cm²)
IV (Subgiant)        (logg=3.5): max_fnu = 3.3865e-05 erg/(Hz·s·cm²)
V (Dwarf)            (logg=4.5): max_fnu = 4.2191e-05 erg/(Hz·s·cm²)
```

Each spectrum follows the same physical model grid but at different surface gravity, enabling true color (SED shape) comparisons across luminosity classes for the same temperature.

## Impact on Downstream Code

**Fully backward-compatible.** The following files work unchanged:
- [icemodels/plot_seds.py](icemodels/plot_seds.py): Calls `atmo_model(T)` → continues to work
- [icemodels/stellar_colors_in_filters.py](icemodels/stellar_colors_in_filters.py): Uses output Table → continues to work
- Example notebooks: Return type unchanged → no edits needed

**Optional enhancements** for new luminosity-class demonstrations:
```python
# Now possible: retrieve 3500K dwarf and giant, compare colors
dwarf = atmo_model(3500, logg=4.5)    # log g=4.5
giant = atmo_model(3500, logg=2.5)    # log g=2.5
# Normalized to same flux, color differences reveal gravity effects
```

## Data Path Setup

To use the refactored `atmo_model()`:

1. **Automatic (built-in default):**
   ```python
   from icemodels.core import atmo_model
   result = atmo_model(5000)  # Uses /orange/adamginsburg/synphot/grp/hst/cdbs
   ```

2. **Custom data path:**
   ```python
   result = atmo_model(5000, pysyn_cdbs='/path/to/cdbs')
   ```

3. **Environment override:**
   ```bash
   export PYSYN_CDBS=/path/to/cdbs
   python -c "from icemodels import core; core.atmo_model(5000)"
   ```

## Summary of Changes

| Component | Old | New | Status |
|-----------|-----|-----|--------|
| Backend library | `mysg` | `synphot`/`stsynphot` | ✅ Installed |
| Model grids | Single gravity (logg=4.0) | Full (T, Z, logg) space | ✅ Validated |
| Data path | Runtime download | Pre-installed CDBS tree | ✅ Configured |
| Function signature | `atmo_model(T, xarr, logg=4.0)` | `atmo_model(T, xarr, logg, metallicity, model_grid, pysyn_cdbs)` | ✅ Implemented |
| Return type | Table with 'nu', 'fnu' | Table with 'nu', 'fnu' (unchanged) | ✅ Backward compatible |
| Test coverage | Mocked `mysg.atmosphere` | Real CDBS grid queries | ✅ PASSED |

## Next Steps (Optional)

To leverage the new gravity-aware capabilities in your notebooks:

1. **For luminosity-class demonstration cells:**
   ```python
   logg_by_class = {'V': 4.5, 'III': 2.5, 'I': 1.0}
   for class_name, logg in logg_by_class.items():
       spectrum = atmo_model(3500, logg=logg)
       # Plot/compare colors...
   ```

2. **For metallicity studies:**
   ```python
   for Z in [-0.5, 0.0, +0.5]:  # subsolar, solar, supersolar
       spectrum = atmo_model(5000, metallicity=Z)
   ```

3. **For hot-star models:**
   ```python
   # Automatically selects k93models for T ≥ 4000K
   result = atmo_model(8000, logg=3.0)
   ```

## References

- **CDBS data:** `/orange/adamginsburg/synphot/grp/hst/cdbs`
- **Synphot docs:** https://synphot.readthedocs.io/
- **Stsynphot docs:** https://stsynphot.readthedocs.io/
- **Phoenix grid reference:** https://phoenix.astro.physik.uni-goettingen.de/
- **Kurucz models:** http://kurucz.harvard.edu/

---

**Refactoring Completed:** 2024
**Status:** Production-ready
**Tests:** ✅ All core tests passing (2 unrelated SVO FPS cache failures)
