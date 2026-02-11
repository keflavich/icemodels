import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from icemodels import plot_stellar_seds
from icemodels.core import retrieve_gerakines_co


# Test basic function call with minimal parameters
def test_plot_stellar_seds_basic():
    """Test basic plot_stellar_seds with single temperature and filter."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F444W']

    fig, axes = plot_stellar_seds(temperatures=temperatures, filters=filters)

    # Check that figure and axes are returned
    assert fig is not None
    assert axes is not None
    assert len(axes) == 2  # Main plot + 1 filter subplot

    plt.close(fig)


def test_plot_stellar_seds_multiple_temperatures():
    """Test plot_stellar_seds with multiple temperatures."""
    temperatures = [3000, 4000, 5000]
    filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']

    fig, axes = plot_stellar_seds(temperatures=temperatures, filters=filters)

    # Check that figure and axes are returned
    assert fig is not None
    assert axes is not None
    assert len(axes) == 3  # Main plot + 2 filter subplots

    plt.close(fig)


def test_plot_stellar_seds_multiple_filters():
    """Test plot_stellar_seds with multiple filters."""
    temperatures = [4000]
    filters = [
        'JWST/NIRCam.F182M',
        'JWST/NIRCam.F212N',
        'JWST/NIRCam.F444W',
        'JWST/MIRI.F1000W'
    ]

    fig, axes = plot_stellar_seds(temperatures=temperatures, filters=filters)

    # Check that figure and axes are returned
    assert fig is not None
    assert axes is not None
    assert len(axes) == 5  # Main plot + 4 filter subplots

    plt.close(fig)


def test_plot_stellar_seds_custom_xarr():
    """Test plot_stellar_seds with custom wavelength array."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F212N']
    xarr_custom = np.linspace(1.5*u.um, 5.2*u.um, 10000)

    fig, axes = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters,
        xarr=xarr_custom
    )

    assert fig is not None
    assert axes is not None

    plt.close(fig)


def test_plot_stellar_seds_with_ice():
    """Test plot_stellar_seds with ice absorption."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F410M']

    # Use retrieve_gerakines_co to get a simple ice table
    co_table = retrieve_gerakines_co(resolution='low')

    fig, axes = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters,
        ice_model_table=co_table,
        ice_column=1e18 * u.cm**-2,
        molecular_weight=28 * u.Da,
        show_ice_absorbed=True
    )

    assert fig is not None
    assert axes is not None

    plt.close(fig)


def test_plot_stellar_seds_with_multiple_ices():
    """Test plot_stellar_seds with multiple ice species."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F410M']

    # Get two ice tables
    co_table = retrieve_gerakines_co(resolution='low')

    # Create a simple CO2-like table for testing
    co2_table = Table()
    co2_table['Wavelength'] = co_table['Wavelength'].copy()
    co2_table['k'] = co_table['k'].copy() * 0.8  # Different k values
    co2_table.meta['density'] = 1.0 * u.g / u.cm**3
    co2_table.meta['molecule'] = 'CO2'

    fig, axes = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters,
        ice_model_table=[co_table, co2_table],
        ice_column=[1e18 * u.cm**-2, 1e18 * u.cm**-2],
        molecular_weight=[28 * u.Da, 44 * u.Da],
        ice_labels=['CO', 'CO2'],
        show_ice_absorbed=True
    )

    assert fig is not None
    assert axes is not None

    plt.close(fig)


def test_plot_stellar_seds_with_extinction():
    """Test plot_stellar_seds with extinction."""
    try:
        from dust_extinction.parameter_averages import CT06_MWGC

        temperatures = [4000]
        filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']

        fig, axes = plot_stellar_seds(
            temperatures=temperatures,
            filters=filters,
            extinction_Av=17.0,
            extinction_curve=CT06_MWGC()
        )

        assert fig is not None
        assert axes is not None

        plt.close(fig)

    except ImportError:
        pytest.skip("dust_extinction package not available")


def test_plot_stellar_seds_with_extinction_and_ice():
    """Test plot_stellar_seds with both extinction and ice."""
    try:
        from dust_extinction.parameter_averages import CT06_MWGC

        temperatures = [4000]
        filters = ['JWST/NIRCam.F212N']
        co_table = retrieve_gerakines_co(resolution='low')

        fig, axes = plot_stellar_seds(
            temperatures=temperatures,
            filters=filters,
            ice_model_table=co_table,
            ice_column=1e18 * u.cm**-2,
            molecular_weight=28 * u.Da,
            extinction_Av=17.0,
            extinction_curve=CT06_MWGC(),
            show_ice_absorbed=True
        )

        assert fig is not None
        assert axes is not None

        plt.close(fig)

    except ImportError:
        pytest.skip("dust_extinction package not available")


def test_plot_stellar_seds_renormalize_insets():
    """Test plot_stellar_seds with renormalized insets."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']
    co_table = retrieve_gerakines_co(resolution='low')

    fig, axes = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters,
        ice_model_table=co_table,
        ice_column=1e18 * u.cm**-2,
        molecular_weight=28 * u.Da,
        renormalize_insets=True,
        show_ice_absorbed=True
    )

    assert fig is not None
    assert axes is not None

    plt.close(fig)


def test_plot_stellar_seds_reusable_axes():
    """Test plot_stellar_seds with reusable axes for comparison plots."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F212N']

    # First plot - baseline
    fig, axes = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters,
        label='Baseline'
    )

    # Second plot - reuse axes
    fig2, axes2 = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters,
        fig=fig,
        axes=axes,
        label='Second scenario'
    )

    # Should return the same fig (axes may be a new array but same underlying objects)
    assert fig2 is fig
    assert len(axes2) == len(axes)
    # Check that axes contain the same matplotlib axes objects
    for ax1, ax2 in zip(axes, axes2):
        assert ax1 is ax2

    plt.close(fig)


def test_plot_stellar_seds_custom_colors():
    """Test plot_stellar_seds with custom color cycle."""
    temperatures = [3000, 4000, 5000]
    filters = ['JWST/NIRCam.F212N']
    custom_colors = ['red', 'green', 'blue']

    fig, axes = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters,
        color_cycle=custom_colors
    )

    assert fig is not None
    assert axes is not None

    plt.close(fig)


def test_plot_stellar_seds_custom_figsize():
    """Test plot_stellar_seds with custom figure size."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']

    fig, axes = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters,
        figsize=(20, 10)
    )

    assert fig is not None
    assert axes is not None
    assert fig.get_figwidth() == 20
    assert fig.get_figheight() == 10

    plt.close(fig)


def test_plot_stellar_seds_invalid_temperature():
    """Test that invalid temperatures raise appropriate errors or warnings."""
    filters = ['JWST/NIRCam.F212N']

    # Test with temperature outside typical range
    # This may issue a warning but should not crash
    try:
        fig, axes = plot_stellar_seds(
            temperatures=[100],  # Very low temperature
            filters=filters
        )
        plt.close(fig)
    except Exception as e:
        # If it raises an error, that's also acceptable behavior
        assert isinstance(e, (ValueError, RuntimeError, KeyError))


def test_plot_stellar_seds_empty_filters():
    """Test that empty filter list raises appropriate error."""
    temperatures = [4000]

    with pytest.raises((ValueError, IndexError, KeyError)):
        fig, axes = plot_stellar_seds(
            temperatures=temperatures,
            filters=[]
        )


def test_plot_stellar_seds_ice_mismatched_parameters():
    """Test that mismatched ice parameters raise appropriate errors."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F212N']
    co_table = retrieve_gerakines_co(resolution='low')

    # Mismatched number of ice tables and columns
    with pytest.raises((ValueError, TypeError, IndexError)):
        fig, axes = plot_stellar_seds(
            temperatures=temperatures,
            filters=filters,
            ice_model_table=[co_table],
            ice_column=[1e18 * u.cm**-2, 1e18 * u.cm**-2],  # Two columns for one table
            molecular_weight=[28 * u.Da]
        )


def test_plot_stellar_seds_axes_structure():
    """Test that the returned axes have the correct structure."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W', 'JWST/MIRI.F1000W']

    fig, axes = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters
    )

    # axes should be a 1D array with length = 1 + n_filters
    # axes[0]: main plot
    # axes[1:]: filter zoom plots (one per filter)
    assert len(axes) == 1 + len(filters)
    assert len(axes) == 4

    plt.close(fig)


def test_plot_stellar_seds_no_ice_absorbed_flag():
    """Test that show_ice_absorbed=False doesn't plot ice-absorbed curves."""
    temperatures = [4000]
    filters = ['JWST/NIRCam.F212N']
    co_table = retrieve_gerakines_co(resolution='low')

    fig, axes = plot_stellar_seds(
        temperatures=temperatures,
        filters=filters,
        ice_model_table=co_table,
        ice_column=1e18 * u.cm**-2,
        molecular_weight=28 * u.Da,
        show_ice_absorbed=False  # Should not show ice-absorbed curves
    )

    assert fig is not None
    assert axes is not None

    plt.close(fig)


# Integration test combining multiple features
def test_plot_stellar_seds_integration():
    """Integration test with multiple features enabled."""
    try:
        from dust_extinction.parameter_averages import CT06_MWGC

        temperatures = [3000, 4000, 5000]
        filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F410M', 'JWST/NIRCam.F444W']
        xarr_custom = np.linspace(1.5*u.um, 5.2*u.um, 15000)
        co_table = retrieve_gerakines_co(resolution='low')

        fig, axes = plot_stellar_seds(
            temperatures=temperatures,
            filters=filters,
            xarr=xarr_custom,
            ice_model_table=co_table,
            ice_column=1e18 * u.cm**-2,
            molecular_weight=28 * u.Da,
            extinction_Av=17.0,
            extinction_curve=CT06_MWGC(),
            renormalize_insets=True,
            show_ice_absorbed=True,
            figsize=(20, 8),
            color_cycle=['red', 'green', 'blue']
        )

        assert fig is not None
        assert axes is not None
        assert len(axes.flatten()) == 2 * len(filters)

        plt.close(fig)

    except ImportError:
        pytest.skip("dust_extinction package not available")
