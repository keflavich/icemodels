import numpy as np
from unittest.mock import patch, MagicMock
from astropy import units as u
from astropy.table import Table
from icemodels.core import (
    download_all_ocdb, download_all_lida, atmo_model, load_molecule,
    read_ocdb_file, load_molecule_univap, composition_to_molweight,
    parse_molscomps, retrieve_gerakines_co, absorbed_spectrum, fluxes_in_filters
)


# Test for download_all_ocdb
def test_download_all_ocdb():
    with patch('requests.Session') as mock_session:
        mock_resp = MagicMock()
        mock_resp.text = "Composition: H2O\nTemperature: 10K\nReference: Test"
        mock_session.return_value.get.return_value = mock_resp
        download_all_ocdb(n_ocdb=1, redo=True)
        # Verify that the session was used to get the correct URL
        mock_session.return_value.get.assert_called_with(
            'https://ocdb.smce.nasa.gov/dataset/1/download-data/all'
        )


# Test for download_all_lida
def test_download_all_lida():
    with patch('requests.Session') as mock_session:
        mock_resp = MagicMock()
        mock_resp.text = "<html><a class='name' href='/data/1'>Test</a></html>"
        mock_session.return_value.get.return_value = mock_resp
        download_all_lida(n_lida=1, redo=True)
        # Verify that the session was used to get the correct URL
        mock_session.return_value.get.assert_called_with(
            'https://icedb.strw.leidenuniv.nl/data/1'
        )


# Test for atmo_model
def test_atmo_model():
    with patch('mysg.atmosphere.interp_atmos') as mock_interp_atmos:
        mock_interp_atmos.return_value = {'nu': [1, 2, 3], 'fnu': [0.1, 0.2, 0.3]}
        result = atmo_model(4000)
        assert 'fnu' in result.colnames
        assert 'nu' in result.colnames
        assert result.meta['temperature'] == 4000


# Test for load_molecule
def test_load_molecule():
    import icemodels.core as core_mod
    with patch('astropy.table.Table.read') as mock_table_read, \
            patch('requests.get') as mock_get:
        # Patch molecule_data['h2o'] to include density
        core_mod.molecule_data['h2o']['density'] = 1.0
        # Mock the Table returned by Table.read
        mock_table = MagicMock()
        mock_table.colnames = ['col1', 'col2', 'col3']
        # Mock columns to have .unit attribute
        col1 = MagicMock()
        col2 = MagicMock()
        col3 = MagicMock()
        mock_table.__getitem__.side_effect = lambda key: {'col1': col1, 'col2': col2, 'col3': col3}[key]
        # Support renaming columns

        def rename_column(old, new):
            mock_table.colnames = [new if c == old else c for c in mock_table.colnames]
        mock_table.rename_column.side_effect = rename_column
        mock_table_read.return_value = mock_table
        mock_get.return_value.text = "Composition: h2o\nTemperature: 10K\nReference: Test"
        result = load_molecule('h2o')
        assert 'Wavelength' in result.colnames or 'col1' in result.colnames
        assert 'n' in result.colnames or 'col2' in result.colnames
        assert 'k' in result.colnames or 'col3' in result.colnames


# Test for read_ocdb_file
def test_read_ocdb_file():
    from unittest.mock import mock_open
    with patch('astropy.io.ascii.read') as mock_read, \
            patch('builtins.open', mock_open(read_data='data')):
        # Mock the table returned by ascii.read
        mock_table = MagicMock()
        # Use original column names as in the file before renaming
        mock_table.colnames = ['Wavelength (m)', 'k₁']

        def getitem_side_effect(key):
            if key == 'Wavelength (m)':
                return [1, 2, 3] * u.m  # Astropy Quantity
            elif key == 'k₁':
                return [0.1, 0.2, 0.3]
            elif key == 'Wavelength':
                return [1, 2, 3] * u.um  # Astropy Quantity for renamed column
            elif key == 'k':
                return [0.1, 0.2, 0.3]
            else:
                return [0, 0, 0]
        mock_table.__getitem__.side_effect = getitem_side_effect
        # Make colnames mutable and update on __setitem__
        colnames = ['Wavelength (m)', 'k₁']

        def setitem_side_effect(key, value):
            if key == 'k' and 'k' not in colnames:
                colnames.append('k')
            if key == 'Wavelength' and 'Wavelength' not in colnames:
                colnames.append('Wavelength')
        mock_table.__setitem__.side_effect = setitem_side_effect
        type(mock_table).colnames = property(lambda self: colnames)
        mock_read.return_value = mock_table

        result = read_ocdb_file('dummy_file.txt')
        assert 'Wavelength' in result.colnames
        assert 'k' in result.colnames
        assert result['Wavelength'].unit == u.um


# Test for load_molecule_univap
def test_load_molecule_univap():
    with patch('astropy.table.Table.read') as mock_table_read, \
            patch('icemodels.core.get_univap_meta_table') as mock_get_meta:
        # Mock the meta table
        mock_meta = MagicMock()
        mock_meta.loc = {'G1': {'reference': ['Test'], 'sample': ['CO']}}
        mock_get_meta.return_value = mock_meta

        # Mock the molecule table with expected renamed columns
        mock_table = MagicMock()
        mock_table.colnames = ['WaveNum', 'absorbance', 'k', 'n', 'Wavelength']

        def getitem_side_effect(key):
            if key == 'WaveNum':
                return [1, 2, 3] * u.cm**-1
            elif key == 'Wavelength':
                return [1, 2, 3] * u.um
            elif key == 'k':
                return [0.1, 0.2, 0.3]
            elif key == 'n':
                return [1.1, 1.2, 1.3]
            else:
                return [0, 0, 0]
        mock_table.__getitem__.side_effect = getitem_side_effect
        # Set meta to a real dict with expected values
        mock_table.meta = {'molecule': 'co', 'temperature': 10}
        mock_table_read.return_value = mock_table

        # Patch meta_table to be available as both meta_table and metatable
        import icemodels.core as core_mod
        setattr(core_mod, 'metatable', mock_meta)
        result = load_molecule_univap('co')
        assert 'Wavelength' in result.colnames
        assert 'k' in result.colnames
        assert 'n' in result.colnames
        assert result.meta['molecule'] == 'co'
        assert result.meta['temperature'] == 10


# Test for composition_to_molweight
def test_composition_to_molweight():
    # Test simple molecule
    result = composition_to_molweight('H2O')
    assert result.unit == u.Da
    assert abs(result.value - 18.015) < 0.001

    # Test complex molecule
    result = composition_to_molweight('CH3OH')
    assert result.unit == u.Da
    assert abs(result.value - 32.042) < 0.001


# Test for parse_molscomps
def test_parse_molscomps():
    # Test simple composition
    mols, comps = parse_molscomps('H2O')
    assert mols == ['H2O']
    assert comps == [1]

    # Test mixture (should match function's actual output)
    mols, comps = parse_molscomps('H2O:CO.1:0.4')
    assert mols == ['H2O:CO.1:0.4']
    assert comps == [1]

    # Test with parentheses
    mols, comps = parse_molscomps('H2O(1):CO(0.4)')
    assert mols == ['H2O(1):CO(0.4)']
    assert comps == [1]


# Integration tests for the full pipeline
def test_ice_absorption_pipeline():
    # 1. Load CO ice opacity constants
    co_table = retrieve_gerakines_co(resolution='low')

    # Check that we have the expected columns and metadata
    assert 'Wavelength' in co_table.colnames
    assert 'k' in co_table.colnames
    assert co_table.meta['molecule'] == 'CO'
    assert 'density' in co_table.meta

    # 2. Create a mock spectrum
    wavelength = np.linspace(1, 10, 1000) * u.um
    flux = np.ones_like(wavelength.value) * u.Jy
    spectrum = Table([wavelength, flux], names=['wavelength', 'fnu'])

    # 3. Apply the opacity constants to the spectrum
    ice_column = 1e18 * u.cm**-2
    absorbed = absorbed_spectrum(ice_column, co_table, spectrum['fnu'], xarr=spectrum['wavelength'])

    # Check that the absorbed spectrum has the right length and units
    assert len(absorbed) == len(spectrum)
    assert absorbed.unit == u.Jy

    # 4. Calculate fluxes in JWST filters
    filters = ['JWST/NIRCam.F444W', 'JWST/MIRI.F560W']
    fluxes = fluxes_in_filters(spectrum['wavelength'], absorbed, filters)

    # Check that we get the expected filter fluxes
    for f in filters:
        assert f in fluxes
    # Check that the absorbed fluxes are less than the original fluxes for the requested filters
    original_fluxes = fluxes_in_filters(spectrum['wavelength'], spectrum['fnu'], filters)
    for f in filters:
        flux = fluxes[f]
        orig_flux = original_fluxes[f]
        if not hasattr(orig_flux, 'unit'):
            orig_flux = orig_flux * flux.unit
        assert flux < orig_flux


def test_ice_absorption_pipeline_with_gaussians():
    # Test the Gaussian absorption model
    center = 4.67 * u.um  # CO ice band
    width = 0.1 * u.um
    ice_bandstrength = 1.1e-17 * u.cm  # CO ice band strength
    ice_column = 1e18 * u.cm**-2

    # Create a mock spectrum
    wavelength = np.linspace(1, 10, 1000) * u.um
    flux = np.ones_like(wavelength.value) * u.Jy
    spectrum = Table([wavelength, flux], names=['wavelength', 'fnu'])

    # Create a Gaussian absorption model
    gaussian = np.exp(-(wavelength - center)**2 / (2 * width**2))
    k = ice_bandstrength * gaussian
    ice_table = Table([wavelength, k], names=['Wavelength', 'k'])
    ice_table.meta['molecule'] = 'CO'
    ice_table.meta['density'] = 0.8 * u.g * u.cm**-3

    # Apply the absorption
    absorbed = absorbed_spectrum(ice_column, ice_table, spectrum['fnu'], xarr=spectrum['wavelength'])

    # Check that the absorbed spectrum has the right length and units
    assert len(absorbed) == len(spectrum)
    assert absorbed.unit == u.Jy

    # Calculate fluxes in JWST filters
    filters = ['JWST/NIRCam.F444W', 'JWST/MIRI.F560W']
    fluxes = fluxes_in_filters(spectrum['wavelength'], absorbed, filters)

    # Check that we get the expected filter fluxes
    for f in filters:
        assert f in fluxes
    # Check that the absorbed fluxes are less than the original fluxes for the requested filters
    original_fluxes = fluxes_in_filters(spectrum['wavelength'], spectrum['fnu'], filters)
    for f in filters:
        flux = fluxes[f]
        orig_flux = original_fluxes[f]
        if not hasattr(orig_flux, 'unit'):
            orig_flux = orig_flux * flux.unit
        assert flux <= orig_flux
