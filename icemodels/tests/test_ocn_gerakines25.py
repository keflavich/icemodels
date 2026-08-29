

def test_measured_profile_closure():
    """The measured-absorbance OCN- table must reproduce the Gerakines+2025 A'."""
    import numpy as np
    from icemodels.ocn_gerakines25 import (make_ocn_table_from_mix_absorbance,
                                           closure_aprime, OCN_BAND_APRIME)
    t = make_ocn_table_from_mix_absorbance()
    nu = np.array(t['Wavenumber'])
    k = np.array(t['k'])
    assert np.isclose(closure_aprime(nu, k), OCN_BAND_APRIME, rtol=1e-3)
    # band peak at the published position
    assert abs(nu[np.argmax(k)] - 2170.0) < 3.0


def test_ocn_band_only_in_nh3_mixture():
    """OCN- forms only when NH3 is present (acid-base product)."""
    import numpy as np
    from icemodels.ocn_gerakines25 import load_gerakines2025_ices
    t = load_gerakines2025_ices()
    nu = np.array(t['Wavenumber'])
    band = (nu > 2150) & (nu < 2190)
    wing = ((nu > 2100) & (nu < 2130)) | ((nu > 2205) & (nu < 2230))
    depth = {}
    for ice in ('HNCO', 'H2O_HNCO_10_1', 'H2O_HNCO_NH3_10_1_1'):
        a = np.array(t['absorbance_' + ice])
        depth[ice] = a[band].max() - np.median(a[wing])
    assert depth['H2O_HNCO_NH3_10_1_1'] > 5 * depth['H2O_HNCO_10_1']
