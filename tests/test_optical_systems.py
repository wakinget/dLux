from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux import (
    LayeredOpticalSystem,
    AngularOpticalSystem,
    CartesianOpticalSystem,
    ThreePlaneOpticalSystem,
    PointSource,
    Wavefront,
    PSF,
)
from dLux.layers import Optic


@pytest.fixture
def wf_npixels():
    return 16


@pytest.fixture
def diameter():
    return 1.0


@pytest.fixture
def layers():
    return [Optic()]


def _test_model(optics):
    source = PointSource([1e-6])
    assert isinstance(optics.model(source), np.ndarray)
    assert isinstance(optics.model(source, return_wf=True), Wavefront)
    assert isinstance(optics.model(source, return_psf=True), PSF)
    with pytest.raises(ValueError):
        optics.model(source, return_wf=True, return_psf=True)


def _test_propagate(optics):
    wavels = np.ones(2)
    assert isinstance(optics.propagate(wavels), np.ndarray)
    assert isinstance(optics.propagate(wavels, return_wf=True), Wavefront)
    assert isinstance(optics.propagate(wavels, return_psf=True), PSF)

    with pytest.raises(ValueError):
        optics.propagate(wavels, return_wf=True, return_psf=True)
    with pytest.raises(ValueError):
        optics.propagate(wavels, weights=np.ones(3))
    with pytest.raises(ValueError):
        optics.propagate(wavels, offset=np.ones(3))


def _test_propagate_mono(optics):
    assert isinstance(optics.propagate_mono(1e-6), np.ndarray)
    assert isinstance(optics.propagate_mono(1e-6, return_wf=True), Wavefront)


def test_layered_optics(wf_npixels, diameter, layers):
    optics = LayeredOpticalSystem(wf_npixels, diameter, layers)
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)

    # Test getattr
    optics.Optic
    optics.opd
    with pytest.raises(AttributeError):
        optics.not_an_attr

    # Test insert and remove layer
    optics.insert_layer(Optic(), 1)
    optics.remove_layer("Optic")


@pytest.fixture
def psf_npixels():
    return 8


@pytest.fixture
def psf_pixel_scale():
    return 1 / 8


@pytest.fixture
def oversample():
    return 2


@pytest.fixture
def focal_length():
    return 1.0


def test_angular_optics(
    wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample
):
    optics = AngularOpticalSystem(
        wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample
    )
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)


def test_cartesian_optics(
    wf_npixels,
    diameter,
    layers,
    focal_length,
    psf_npixels,
    psf_pixel_scale,
    oversample,
):
    optics = CartesianOpticalSystem(
        wf_npixels,
        diameter,
        layers,
        focal_length,
        psf_npixels,
        psf_pixel_scale,
        oversample,
    )
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)


def test_three_plane_optical_system(
    wf_npixels,
    diameter,
    layers,
    psf_npixels,
    psf_pixel_scale,
    oversample,
):
    """
    Integration test for the `ThreePlaneOpticalSystem` class.

    Verifies correct behavior of layer management, attribute access,
    and partial propagation through both planes. Ensures the class
    behaves consistently with other optical system subclasses while
    correctly implementing its intermediate-plane logic.

    The test covers:
      - **Layer operations:** insertion, removal, invalid plane handling,
        and attribute lookup via `__getattr__`.
      - **Monochromatic propagation:** tests `prop_mono_to_p2` for both PSF
        and `Wavefront` outputs.
      - **Polychromatic propagation:** tests `prop_to_p2` for summed PSF,
        stacked `Wavefront`, and `PSF` object outputs.
      - **API compatibility:** runs shared propagation/model tests to confirm
        interface consistency across optical system types.
    """
    optics = ThreePlaneOpticalSystem(
        wf_npixels=wf_npixels,
        p1_diameter=diameter,
        p2_diameter=0.15,
        p1_layers=layers,
        p2_layers=layers,
        plane_separation=0.9,
        magnification=10.0,
        psf_npixels=psf_npixels,
        psf_pixel_scale=0.051566,
        oversample=oversample,
    )
    print("\n--- Initial ThreePlaneOpticalSystem ---")
    print(optics)

    # ----------------------------
    # Insert new layers on each plane
    # ----------------------------
    optics = optics.insert_layer(("P1Mask", Optic()), index=0, plane_index=0)
    optics = optics.insert_layer(("P2Mask", Optic()), index=0, plane_index=1)
    # unlabeled insert on plane 1
    optics = optics.insert_layer(Optic(), index=0, plane_index=1)

    # invalid plane index should raise
    with pytest.raises(ValueError):
        optics.insert_layer(("P3Mask", Optic()), index=0, plane_index=2)

    print("\n--- After Layer Inserts ---")
    print(optics)
    print("Has P1Mask:", hasattr(optics, "P1Mask"))
    print("Has P2Mask:", hasattr(optics, "P2Mask"))

    # Query by key
    assert isinstance(optics.P1Mask, Optic)
    assert isinstance(optics.P2Mask, Optic)

    # Query nested attribute (e.g. 'opd')
    # Accessing is enough to validate __getattr__
    _ = optics.opd

    # ----------------------------
    # Remove inserted layers and verify gone
    # ----------------------------
    optics = optics.remove_layer("P1Mask", plane_index=0)
    optics = optics.remove_layer("P2Mask", plane_index=1)

    print("\n--- After Layer Removals ---")
    print("Has P1Mask:", hasattr(optics, "P1Mask"))
    print("Has P2Mask:", hasattr(optics, "P2Mask"))

    with pytest.raises(AttributeError):
        _ = optics.P1Mask
    with pytest.raises(AttributeError):
        _ = optics.P2Mask

    # ----------------------------
    # Partial propagation to Plane 2 (monochromatic)
    # ----------------------------
    wl = 1.0e-6

    # Return PSF array at Plane 2
    p2_psf = optics.prop_mono_to_p2(wl, return_wf=False)
    print(
        "\n[mono -> P2] return_wf=False type:",
        type(p2_psf),
        "shape:",
        getattr(p2_psf, "shape", None),
    )
    assert isinstance(p2_psf, np.ndarray)
    assert p2_psf.shape == (wf_npixels, wf_npixels)

    # Return Wavefront at Plane 2
    p2_wf = optics.prop_mono_to_p2(wl, return_wf=True)
    print("[mono -> P2] return_wf=True type:", type(p2_wf))
    # try to introspect common attrs for clarity
    if hasattr(p2_wf, "amplitude"):
        print(
            "Wavefront amplitude.shape:",
            getattr(p2_wf.amplitude, "shape", None),
        )
        print("Wavefront pixel_scale:", getattr(p2_wf, "pixel_scale", None))
    assert isinstance(p2_wf, Wavefront)
    assert p2_wf.amplitude.shape == (wf_npixels, wf_npixels)

    # ----------------------------
    # Partial propagation to Plane 2 (polychromatic)
    # ----------------------------
    wavels = np.array([0.9e-6, 1.0e-6, 1.1e-6])

    # Default: summed PSF array at Plane 2
    p2_poly_psf = optics.prop_to_p2(wavels)
    print(
        "\n[poly -> P2] default type:",
        type(p2_poly_psf),
        "shape:",
        getattr(p2_poly_psf, "shape", None),
    )
    assert isinstance(p2_poly_psf, np.ndarray)
    assert p2_poly_psf.shape == (wf_npixels, wf_npixels)

    # Return a stack of Wavefronts
    p2_poly_wf = optics.prop_to_p2(wavels, return_wf=True)
    print("[poly -> P2] return_wf=True type:", type(p2_poly_wf))
    # Conservative inspection: look at .psf and shapes
    p2_stack = p2_poly_wf.psf
    print("Stacked PSF shape:", getattr(p2_stack, "shape", None))
    assert p2_stack.shape[0] == wavels.shape[0]
    assert p2_stack.shape[1:] == (wf_npixels, wf_npixels)

    # Return PSF object (summed PSF + pixel scale)
    p2_psf_obj = optics.prop_to_p2(wavels, return_psf=True)
    print("[poly -> P2] return_psf=True type:", type(p2_psf_obj))
    if hasattr(p2_psf_obj, "pixel_scale"):
        print("PSF.pixel_scale:", p2_psf_obj.pixel_scale)
    assert isinstance(p2_psf_obj, PSF)
    assert p2_psf_obj.data.shape == (wf_npixels, wf_npixels)

    # ----------------------------
    # Reuse standard checks
    # ----------------------------
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)
