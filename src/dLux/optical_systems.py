from __future__ import annotations
from collections import OrderedDict
from abc import abstractmethod
import jax.numpy as np
from jax import Array
from zodiax import filter_vmap, Base
from typing import Union, Any
import dLux.utils as dlu


__all__ = [
    "BaseOpticalSystem",
    "AngularOpticalSystem",
    "CartesianOpticalSystem",
    "LayeredOpticalSystem",
    "TwoPlaneOpticalSystem",
    "MultiPlaneOpticalSystem",
]

from .layers.optical_layers import OpticalLayer
from .wavefronts import Wavefront
from .sources import BaseSource as Source
from .psfs import PSF


###################
# Private Classes #
###################
class BaseOpticalSystem(Base):
    @abstractmethod
    def propagate_mono(
        self: BaseOpticalSystem,
        wavelength: float,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """

    @abstractmethod
    def propagate(
        self: OpticalSystem,
        wavelengths: Array,
        offset: Array = np.zeros(2),
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, metres
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        weights : Array = None
            The weight of each wavelength. If None, all weights are equal.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.

        """

    @abstractmethod
    def model(
        self: OpticalSystem,
        source: Source,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Models the input Source object through the optics.

        Parameters
        ----------
        source : Source
            The Source object to model through the optics.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """


class OpticalSystem(BaseOpticalSystem):
    """
    Base optics class implementing both the `propagate` and `model` methods that are
    universal to all optics classes.
    """

    def propagate(
        self: OpticalSystem,
        wavelengths: Array,
        offset: Array = np.zeros(2),
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, metres
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        weights : Array = None
            The weight of each wavelength. If None, all weights are equal.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        if return_wf and return_psf:
            raise ValueError(
                "return_wf and return_psf cannot both be True. "
                "Please choose one."
            )

        wavelengths = np.atleast_1d(wavelengths)
        if weights is None:
            weights = np.ones_like(wavelengths) / len(wavelengths)
        else:
            weights = np.atleast_1d(weights)

        # Check wavelengths and weights
        if weights.shape != wavelengths.shape:
            raise ValueError(
                "wavelengths and weights must have the "
                f"same shape, got {wavelengths.shape} and {weights.shape} "
                "respectively."
            )

        # Check offset
        offset = np.array(offset) if not isinstance(offset, Array) else offset
        if offset.shape != (2,):
            raise ValueError(
                "offset must be a 2-element array, got "
                f"shape {offset.shape}."
            )

        # Calculate - note we multiply by sqrt(weight) to account for the
        # fact that the PSF is the square of the amplitude
        prop_fn = lambda wavelength, weight: self.propagate_mono(
            wavelength, offset, return_wf=True
        ).multiply("amplitude", weight**0.5)
        wf = filter_vmap(prop_fn)(wavelengths, weights)

        # Return PSF, Wavefront, or array psf
        if return_wf:
            return wf
        if return_psf:
            return PSF(wf.psf.sum(0), wf.pixel_scale.mean())
        return wf.psf.sum(0)

    def model(
        self: OpticalSystem,
        source: Source,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Models the input Source object through the optics.

        Parameters
        ----------
        source : Source
            The Source object to model through the optics.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        return source.model(self, return_wf, return_psf)


class ParametricOpticalSystem(OpticalSystem):
    """
    Implements the attributes required for an optical system with a specific output
    pixel scale and number of pixels.

    Attributes
    ----------
    psf_npixels : int
        The number of pixels of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    """

    psf_npixels: int
    oversample: int
    psf_pixel_scale: float

    def __init__(
        self: OpticalSystem,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float
            The pixel scale of the final PSF.
        oversample : int = 1.
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        self.psf_npixels = int(psf_npixels)
        self.oversample = int(oversample)
        self.psf_pixel_scale = float(psf_pixel_scale)
        super().__init__(**kwargs)


##################
# Public Classes #
##################
class LayeredOpticalSystem(OpticalSystem):
    """
    A flexible optical system that allows for the arbitrary chaining of OpticalLayers.

    ??? abstract "UML"
        ![UML](../../assets/uml/LayeredOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The size of the initial wavefront to propagate.
    diameter : float, metres
        The diameter of the wavefront to propagate.
    layers : OrderedDict
        A series of `OpticalLayer` transformations to apply to wavefronts.
    """

    wf_npixels: int
    diameter: float
    layers: OrderedDict

    def __init__(
        self: OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer, tuple],
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The size of the initial wavefront to propagate.
        diameter : float
            The diameter of the wavefront to propagate.
        layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        """
        self.wf_npixels = int(wf_npixels)
        self.diameter = float(diameter)
        self.layers = dlu.list2dictionary(layers, True, OpticalLayer)

    def __getattr__(self: OpticalSystem, key: str) -> Any:
        """
        Raises both the individual layers and the attributes of the layers via
        their keys.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the layers dictionary.

        Returns
        -------
        item : object
            The item corresponding to the supplied key in the layers dictionary.
        """
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute " f"{key}."
        )

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        # Initialise wavefront
        wavefront = Wavefront(self.wf_npixels, self.diameter, wavelength)
        wavefront = wavefront.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wavefront *= layer

        # Return PSF or Wavefront
        if return_wf:
            return wavefront
        return wavefront.psf

    def insert_layer(
        self: OpticalSystem, layer: Union[OpticalLayer, tuple], index: int
    ) -> OpticalSystem:
        """
        Inserts a layer into the layers dictionary at a specified index. This function
        calls the list2dictionary function to ensure all keys remain unique. Note that
        this can result in some keys being modified if they are duplicates. The input
        'layer' can be a tuple of (key, layer) to specify a key, else the key is taken
        as the class name of the layer.

        Parameters
        ----------
        layer : Any
            The layer to be inserted.
        index : int
            The index at which to insert the layer.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        return self.set(
            "layers", dlu.insert_layer(self.layers, layer, index, OpticalLayer)
        )

    def remove_layer(self: OpticalLayer, key: str) -> OpticalSystem:
        """
        Removes a layer from the layers dictionary, specified by its key.

        Parameters
        ----------
        key : str
            The key of the layer to be removed.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        return self.set("layers", dlu.remove_layer(self.layers, key))


class AngularOpticalSystem(ParametricOpticalSystem, LayeredOpticalSystem):
    """
    An extension to the LayeredOpticalSystem class that propagates a wavefront to an
    image plane with `psf_pixel_scale` in units of arcseconds.

    ??? abstract "UML"
        ![UML](../../assets/uml/AngularOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, metres
        The diameter of the initial wavefront to propagate.
    layers : OrderedDict
        A series of `OpticalLayer` transformations to apply to wavefronts.
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float, arcseconds
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    """

    def __init__(
        self: OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer, tuple],
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, metres
            The diameter of the initial wavefront to propagate.
        layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, arcseconds
            The pixel scale of the final PSF in units of arcseconds.
        oversample : int
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        wf = super().propagate_mono(wavelength, offset, return_wf=True)

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class CartesianOpticalSystem(ParametricOpticalSystem, LayeredOpticalSystem):
    """
    An extension to the LayeredOpticalSystem class that propagates a wavefront to an
    image plane with `psf_pixel_scale` in units of microns.

    ??? abstract "UML"
        ![UML](../../assets/uml/CartesianOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, metres
        The diameter of the initial wavefront to propagate.
    layers : OrderedDict
        A series of `OpticalLayer` transformations to apply to wavefronts.
    focal_length : float, metres
        The focal length of the system.
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float, microns
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    """

    focal_length: None

    def __init__(
        self: OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer, tuple],
        focal_length: float,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, metres
            The diameter of the initial wavefront to propagate.
        layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        focal_length : float, metres
            The focal length of the system.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, microns
            The pixel scale of the final PSF in units of microns.
        oversample : int
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        self.focal_length = float(focal_length)

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        wf = super().propagate_mono(wavelength, offset, return_wf=True)

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = 1e-6 * true_pixel_scale
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale, self.focal_length)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class TwoPlaneOpticalSystem(ParametricOpticalSystem, LayeredOpticalSystem):
    """
    An extension to the LayeredOpticalSystem class that propagates a wavefront to an
    intermediate plane before propagating to the image plane with `psf_pixel_scale`
    in units of arcseconds.

    # TODO: Modify the unit test for this class to actually test specific methods
    # TODO: Create a UML png image describing the inheritence of this class
    # TODO: Needs a __getattr__ method to access the layers properly

    ??? abstract "UML"
        ![UML](../../assets/uml/AngularOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    p1_diameter : Array, metres
        The diameter of the first plane.
    p2_diameter : Array, metres
        The diameter of the second plane.
    p1_layers : OrderedDict
        A series of `OpticalLayer` transformations to apply at plane 1.
    p2_layers : OrderedDict
            A series of `OpticalLayer` transformations to apply at plane 2.
    separation : float, metres
            The physical distance between plane 1 and plane 2
    magnification : float
            The magnification at plane 1, affects the propagation distance
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float, arcseconds
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    """

    p1_diameter: None
    p2_diameter: None
    p1_layers: OrderedDict
    p2_layers: OrderedDict
    separation: None
    magnification: None

    def __init__(
        self: OpticalSystem,
        wf_npixels: int,
        p1_diameter: float,
        p2_diameter: float,
        p1_layers: list[OpticalLayer, tuple],
        p2_layers: list[OpticalLayer, tuple],
        separation: float,
        magnification: float,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        p1_diameter : Array, metres
            The diameter of the first plane.
        p2_diameter : Array, metres
            The diameter of the second plane.
        p1_layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply at the pupil. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        p2_layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply at plane 2. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        separation : float, metres
            The physical distance between plane 1 and plane 2
        magnification : float
            The magnification at plane 1, affects the propagation distance
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, arcseconds
            The pixel scale of the final PSF in units of arcseconds.
        oversample : int
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """

        self.p1_diameter = float(p1_diameter)
        self.p2_diameter = float(p2_diameter)
        self.p1_layers = dlu.list2dictionary(p1_layers, True, OpticalLayer)
        self.p2_layers = dlu.list2dictionary(p2_layers, True, OpticalLayer)
        self.separation = float(separation)
        self.magnification = float(magnification)

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=p1_diameter,
            layers=[],
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

    def insert_layer(
        self: OpticalSystem,
        layer: Union[OpticalLayer, tuple],
        index: int,
        plane_index: int,  # Updated argument name
    ) -> OpticalSystem:
        """
        Inserts a layer into the specified plane's layers at a given index.

        Parameters
        ----------
        layer : OpticalLayer or tuple
            The layer to insert. Can be a tuple (key, layer) to specify a key.
        index : int
            The index at which to insert the layer.
        plane_index : int
            The index of the plane where the layer should be inserted. `0` or `1`.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        if plane_index == 0:
            updated_layers = dlu.insert_layer(
                self.p1_layers, layer, index, OpticalLayer
            )
            return self.set("p1_layers", updated_layers)
        elif plane_index == 1:
            updated_layers = dlu.insert_layer(
                self.p2_layers, layer, index, OpticalLayer
            )
            return self.set("p2_layers", updated_layers)
        else:
            raise ValueError("Invalid plane_index. Must be 0 or 1.")

    def remove_layer(
        self: OpticalSystem, key: str, plane_index: int
    ) -> OpticalSystem:
        """
        Removes a layer from the specified plane's layers.

        Parameters
        ----------
        key : str
            The key of the layer to remove.
        plane : int
            The plane where the layer should be removed. Must be `0` or `1`.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        if plane_index == 0:
            updated_layers = dlu.remove_layer(self.p1_layers, key)
            return self.set("p1_layers", updated_layers)
        elif plane_index == 1:
            updated_layers = dlu.remove_layer(self.p2_layers, key)
            return self.set("p2_layers", updated_layers)
        else:
            raise ValueError("Invalid plane_index. Must be 0 or 1.")

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """

        # Initialise a Wavefront object
        wf = Wavefront(self.wf_npixels, self.p1_diameter, wavelength)
        wf = wf.tilt(offset)

        # Apply pupil layers
        for layer in list(self.p1_layers.values()):
            wf *= layer

        # Propagate to Plane 2
        prop_dist = self.separation * self.magnification
        wf = wf.propagate_fresnel_AS(prop_dist)

        # Resample at Plane 2
        sampling_factor = (
            self.p2_diameter / self.p1_diameter * self.magnification
        )
        wf = wf.scale_to(self.wf_npixels, wf.pixel_scale * sampling_factor)
        wf *= sampling_factor

        # Apply secondary layers
        for layer in list(self.p2_layers.values()):
            wf *= layer

        # Propagate to Focus
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf

    def propagate_to_plane2(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the first optical plane and
        stops at the second optical plane.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the PSF Array at Plane 2.
            if `return_wf` is True, returns the Wavefront object at Plane 2.
        """

        # Initialise a Wavefront object at Plane 1
        wf = Wavefront(self.wf_npixels, self.p1_diameter, wavelength)
        wf = wf.tilt(offset)

        # Apply Plane 1 layers
        for layer in list(self.p1_layers.values()):
            wf *= layer

        # Propagate to Plane 2
        prop_dist = self.separation * self.magnification
        wf = wf.propagate_fresnel_AS(prop_dist)

        # Resample at Plane 2
        sampling_factor = (
            self.p2_diameter / self.p1_diameter * self.magnification
        )
        wf = wf.scale_to(self.wf_npixels, wf.pixel_scale * sampling_factor)
        wf *= sampling_factor  # Maintain flux

        # Apply Plane 2 layers
        for layer in list(self.p2_layers.values()):
            wf *= layer

        # Return PSF or Wavefront at Plane 2
        if return_wf:
            return wf
        return wf.psf


class MultiPlaneOpticalSystem(ParametricOpticalSystem, LayeredOpticalSystem):
    """
    A generalization of TwoPlaneOpticalSystem that supports multiple optical planes.
    """

    diameter: list[float]
    plane_separations: list[float]
    plane_magnifications: list[float]
    layers: OrderedDict[int, OrderedDict[str, OpticalLayer]]

    def __init__(
        self,
        wf_npixels: int,
        diameter: Union[float, list[float]],
        plane_layers: list[list[OpticalLayer, tuple]],
        plane_separations: Union[float, list[float]],
        plane_magnifications: Union[float, list[float]],
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        """
        Multi-plane optical system initialization.

        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : float or list[float]
            The diameters of each optical plane.
        plane_layers : list[list[OpticalLayer, tuple]]
            A list of lists containing layers for each plane.
        plane_separations : float or list[float]
            The physical separations between planes.
        plane_magnifications : float or list[float]
            The magnification factors for each plane.
        psf_npixels : int
            The number of pixels in the final PSF.
        psf_pixel_scale : float
            The pixel scale of the final PSF in arcseconds.
        oversample : int
            The oversampling factor of the final PSF.
        """

        # Ensure diameter is a list
        if isinstance(diameter, (float, int)):
            diameter = [diameter]

        # Ensure separations and magnifications are lists
        plane_separations = (
            [plane_separations]
            if isinstance(plane_separations, (float, int))
            else plane_separations
        )
        plane_magnifications = (
            [plane_magnifications]
            if isinstance(plane_magnifications, (float, int))
            else plane_magnifications
        )

        # Call parent constructor
        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter[
                0
            ],  # Use first plane's diameter for LayeredOpticalSystem
            layers={},  # Empty since we'll override it
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

        # Store attributes
        self.diameter = diameter
        self.plane_separations = plane_separations
        self.plane_magnifications = plane_magnifications

        # Define layers as a nested OrderedDict
        self.layers = OrderedDict(
            {
                i: dlu.list2dictionary(plane_layers[i], True, OpticalLayer)
                for i in range(len(plane_layers))
            }
        )

    def __getattr__(self, key: str) -> Any:
        """
        Allows accessing layers dynamically from the nested `layers` dictionary.

        If a key exists in multiple planes, users must specify a plane index using:
        `get_layer(plane_index, layer_name)`

        Parameters
        ----------
        key : str
            The key of the attribute to search for.

        Returns
        -------
        object : OpticalLayer, dict
            - If `key` is "layers", returns the full layers dictionary.
            - If `key` is "get_layer", returns a method to access specific layers.
        """
        # Allow direct access to the layers dictionary
        if key == "layers":
            return object.__getattribute__(self, "layers")

        # Provide a method for users to get layers by plane index
        if key == "get_layer":
            return self._get_layer_by_plane

        # Search for key inside each plane's OrderedDict
        matches = [
            layer_dict[key]
            for layer_dict in self.layers.values()
            if key in layer_dict
        ]

        if len(matches) == 1:
            return matches[0]  # Only one match, return directly

        if len(matches) > 1:
            raise AttributeError(
                f"Layer '{key}' exists in multiple planes. "
                f"Use `get_layer(plane_index, '{key}')`."
            )

        raise AttributeError(
            f"{self.__class__.__name__} has no attribute '{key}'."
        )

    def propagate_mono(
        self,
        wavelength: float,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic wavefront through multiple planes.

        Parameters
        ----------
        wavelength : float
            The wavelength of the wavefront.
        offset : Array, optional
            The (x, y) offset from the optical axis.
        return_wf : bool, optional
            Whether to return the final Wavefront object.

        Returns
        -------
        Array or Wavefront
            The propagated wavefront or final PSF.
        """

        # Initialize the wavefront at the first plane
        wf = Wavefront(self.wf_npixels, self.diameter[0], wavelength)
        wf = wf.tilt(offset)

        # Iterate through optical planes
        for i in range(len(self.plane_separations)):
            # Apply layers at this plane
            for layer in self.layers.get(i, {}).values():
                wf *= layer

            # Propagate to the next plane
            prop_dist = (
                self.plane_separations[i] * self.plane_magnifications[i]
            )
            wf = wf.propagate_fresnel_AS(prop_dist)

            # Resample at the next plane
            if i + 1 < len(self.diameter):
                sampling_factor = (
                    self.diameter[i + 1]
                    / self.diameter[i]
                    * self.plane_magnifications[i]
                )
                wf = wf.scale_to(
                    self.wf_npixels, wf.pixel_scale * sampling_factor
                )
                wf *= sampling_factor  # Preserve flux

        # Final propagation to the image plane
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale)

        return wf if return_wf else wf.psf

    def propagate_mono_to_plane(
        self,
        wavelength: float,
        offset: Array = np.zeros(2),
        plane_index: int = -1,  # Default to last plane (full propagation)
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through multiple optical planes
        and stops at the specified plane.

        Parameters
        ----------
        wavelength : float
            The wavelength of the wavefront.
        offset : Array, optional
            The (x, y) offset from the optical axis.
        plane_index : int, optional
            The index of the plane at which to stop propagation.
            Default is -1, which propagates fully to the image plane.
        return_wf : bool, optional
            Whether to return the final Wavefront object.

        Returns
        -------
        Array or Wavefront
            The wavefront (if `return_wf=True`) or the PSF at the specified plane.
        """

        n_planes = len(self.diameter)
        current_plane = 0

        # Adjust for negative index (i.e., full propagation)
        if plane_index < 0:
            plane_index = n_planes  # Last plane index

        if plane_index > n_planes or plane_index < 0:
            raise ValueError(
                f"Invalid plane index: {plane_index}. Must be between 0 and {n_planes}."
            )

        # Initialize the wavefront at the first plane
        wf = Wavefront(self.wf_npixels, self.diameter[0], wavelength)
        wf = wf.tilt(offset)

        # Apply layers at first plane
        for layer in self.layers.get(0, {}).values():
            wf *= layer

        # Stop propagation if we've reached the requested plane
        if plane_index == current_plane:
            return wf if return_wf else wf.psf

        # Iterate through intermediate planes
        n_intermediate_planes = n_planes - 1
        for i in range(n_intermediate_planes):
            prop_dist = (
                self.plane_separations[i] * self.plane_magnifications[i]
            )
            wf = wf.propagate_fresnel_AS(prop_dist)
            current_plane += 1  # Increment current plane

            # Resample at the next plane
            sampling_factor = (
                self.diameter[i + 1]
                / self.diameter[i]
                * self.plane_magnifications[i]
            )
            wf = wf.scale_to(self.wf_npixels, wf.pixel_scale * sampling_factor)
            wf *= sampling_factor  # Preserve flux

            # Apply layers at current plane
            for layer in self.layers.get(i + 1, {}).values():
                wf *= layer

            # Stop if we've reached the requested plane
            if plane_index == current_plane:
                return wf if return_wf else wf.psf

        # Propagate to the image plane
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale)

        return wf if return_wf else wf.psf

    def propagate_to_plane(
        self: OpticalSystem,
        wavelengths: Array,
        offset: Array = np.zeros(2),
        weights: Array = None,
        plane_index: int = -1,  # Default to last plane (full propagation)
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, metres
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        weights : Array = None
            The weight of each wavelength. If None, all weights are equal.
        plane_index : int, optional
            The index of the plane at which to stop propagation.
            Default is -1, which propagates fully to the image plane.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        if return_wf and return_psf:
            raise ValueError(
                "return_wf and return_psf cannot both be True. "
                "Please choose one."
            )

        wavelengths = np.atleast_1d(wavelengths)
        if weights is None:
            weights = np.ones_like(wavelengths) / len(wavelengths)
        else:
            weights = np.atleast_1d(weights)

        # Check wavelengths and weights
        if weights.shape != wavelengths.shape:
            raise ValueError(
                "wavelengths and weights must have the "
                f"same shape, got {wavelengths.shape} and {weights.shape} "
                "respectively."
            )

        # Check offset
        offset = np.array(offset) if not isinstance(offset, Array) else offset
        if offset.shape != (2,):
            raise ValueError(
                "offset must be a 2-element array, got "
                f"shape {offset.shape}."
            )

        # Calculate - note we multiply by sqrt(weight) to account for the
        # fact that the PSF is the square of the amplitude
        prop_fn = lambda wavelength, weight: self.propagate_mono_to_plane(
            wavelength, offset, plane_index, return_wf=True
        ).multiply("amplitude", weight**0.5)
        wf = filter_vmap(prop_fn)(wavelengths, weights)

        # Return PSF, Wavefront, or array psf
        if return_wf:
            return wf
        if return_psf:
            return PSF(wf.psf.sum(0), wf.pixel_scale.mean())
        return wf.psf.sum(0)

    def insert_layer(
        self, layer: Union[OpticalLayer, tuple], index: int, plane_index: int
    ) -> OpticalSystem:
        """
        Inserts a layer into the specified plane's layers at a given index.

        Parameters
        ----------
        layer : OpticalLayer or tuple
            The layer to insert. Can be a tuple (key, layer) to specify a key.
        index : int
            The index at which to insert the layer.
        plane_index : int
            The index of the plane where the layer should be inserted.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        if plane_index not in self.layers:
            raise ValueError(
                f"Invalid plane index: {plane_index}. "
                f"Must be between 0 and {len(self.layers) - 1}."
            )

        # Extract the OrderedDict for the specified plane
        updated_plane_layers = dlu.insert_layer(
            self.layers[plane_index], layer, index, OpticalLayer
        )

        # Create a new layers dictionary to avoid mutability issues
        new_layers = self.layers.copy()
        new_layers[
            plane_index
        ] = updated_plane_layers  # Update only the modified plane

        return self.set("layers", new_layers)

    def remove_layer(self, key: str, plane_index: int) -> OpticalSystem:
        """
        Removes a layer from the specified plane's layers.

        Parameters
        ----------
        key : str
            The key of the layer to remove.
        plane_index : int
            The index of the plane where the layer should be removed.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        if plane_index not in self.layers:
            raise ValueError(
                f"Invalid plane index: {plane_index}. "
                f"Must be between 0 and {len(self.layers) - 1}."
            )

        # Extract the OrderedDict for the specified plane
        plane_layers = self.layers[plane_index]

        if key not in plane_layers:
            raise KeyError(f"Layer '{key}' not found in plane {plane_index}.")

        # Use dlu.remove_layer() to remove the key safely
        updated_plane_layers = dlu.remove_layer(
            plane_layers.copy(), key
        )  # Copy to avoid in-place modification

        # Create a new layers dictionary with the updated plane
        new_layers = self.layers.copy()
        new_layers[
            plane_index
        ] = updated_plane_layers  # Update only the modified plane

        return self.set("layers", new_layers)
