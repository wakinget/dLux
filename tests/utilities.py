"""
utilities.py
------------
This file contains testing utilities to help with the generation of 
safe test wavefronts, propagators, layers and detectors. 

Abstract Classes 
----------------
- Utility (abstract)
- UtilityUser (abstract)

Concrete Classes
----------------
- WavefrontUtility
- PhysicalWavefrontUtility
- AngularWavefrontUtility
- GaussianWavefrontUtility
- PropagatorUtility
- VariabelSamplingUtility
- FixedSamplingUtility
- PhysicalMFTUtility
- PhysicalFFTUtility
- PhysicalFresnelUtility
- AngularMFTUtility
- AngularFFTUtility
- AngularFresnelUtility
- GaussianPropagatorUtility
"""
__author__ = "Jordan Dennis"
__date__ = "28/06/2022"


import jax.numpy as numpy
import dLux
import typing
import abc


Array = typing.NewType("Array", numpy.ndarray)
Wavefront = typing.NewType("Wavefront", object)
Utility = typing.NewType("Utility", object)


class UtilityUser(object):
    """
    The base utility class. These utility classes are designed to 
    define safe constructors and constants for testing. These   
    classes are for testing purposes only. 
    """
    utility : Utility


    def get_utility(self : UtilityUser) -> Utility:
        """
        Accessor for the utility. 

        Returns 
        -------
        : Utility
            The utility
        """
        return self.utility 


class Utility(abc.ABC):
    """
    """
    @abc.abstractmethod
    def __init__(self : Utility) -> Utility:
        """
        Construct a new Utility.

        Returns
        : Utility 
            The utility. 
        """
        pass

    
    @abc.abstractmethod
    def construct(self : Utility) -> dLuxModule:
        """
        Safe constructor for the dLuxModule, associated with 
        this utility.

        Returns
        -------
        : dLuxModule
            A safe dLuxModule for testing.
        """
        pass


    def approx(self : Utility, result : Array, comparator : Array) -> Array:
        """
        Compare two arrays to within floating point precision.

        Parameters
        ----------
        result : Array
            The result that you want to test for nearness to
            comparator.
        comparator : Array
            The comparison array.

        Returns
        -------
        : Array[bool]
            True if the array elements are similar to float 
            error, False otherwise. 
        """
        lower_bound = (result - 0.0005) <= comparator
        upper_bound = (result + 0.0005) >= comparator
        return lower_bound & upper_bound 


class WavefrontUtility(Utility):
    """
    Defines safe state constants and a simple constructor for a safe
    `Wavefront` object. 

    Attributes
    ----------
    offset : Array[float]
        A simple array defining the angular displacement of the 
        wavefront. 
    wavelength : float
        A safe wavelength for the testing wavefronts in meters
    """
    wavelength : float
    offset : Array
    size : int
    amplitude : Array 
    phase : Array
    pixel_scale : float


    def __init__(self : WavefrontUtility, /, 
            wavelength : float = None, 
            offset : Array = None,
            size : int = None,
            amplitude : Array = None,
            phase : Array = None,
            pixel_scale : float = None) -> WavefrontUtility:
        """
        Parameters
        ----------
        wavelength : float = 550.e-09
            The safe wavelength for the utility in meters.
        offset : Array = [0., 0.]
            The safe offset for the utility in meters.
        size : int
            A parameter for defining consistent wavefront pixel arrays 
            without causing errors.
        amplitude : Array[float] 
            A simple array defining electric field amplitudes without 
            causing errors.
        phase : Array[float]
            A simple array defining the pixel phase for a wavefront, 
            defined to be safe. 

        Returns 
        -------
        : WavefrontUtility 
            The new utility for generating test cases.
        """
        self.wavelength = 550.e-09 if not wavelength else wavelength
        self.offset = numpy.array([0., 0.]).astype(float) if not \
            offset else numpy.array(offset).astype(float)           
        self.size = 128 if not size else size
        self.amplitude = numpy.ones((self.size, self.size)) if not \
            amplitude else amplitude
        self.phase = numpy.zeros((self.size, self.size)) if not \
            phase else phase
        self.pixel_scale = 1. if not pixel_scale else pixel_scale

        assert self.size == self.amplitude.shape[0]
        assert self.size == self.amplitude.shape[1]
        assert self.size == self.phase.shape[0]
        assert self.size == self.phase.shape[1]          
 

    def construct(self : WavefrontUtility) -> Wavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        : Wavefront
            The safe testing wavefront.
        """
        return dLux\
            .Wavefront(self.wavelength, self.offset)\
            .update_phasor(self.amplitude, self.phase)\
            .set_pixel_scale(self.pixel_scale)


    def get_wavelength(self : WavefrontUtility) -> float:
        """
        Accessor for the wavelength associated with this utility.

        Returns
        -------
        : int
            The wavelength of the utility and hence any `Wavefront` 
            objects it creates in meters.
        """
        return self.wavelength


    def get_size(self : WavefrontUtility) -> int:
        """
        Accessor for the `size` constant.

        Returns
        -------
        : int
            The side length of a pixel array currently stored.
        """
        return self.size


    def get_amplitude(self : WavefrontUtility) -> Array:
        """
        Accessor for the `amplitude` constant.

        Returns 
        -------
        : Array
            The square array of pixel amplitudes in SI units of 
            electric field.
        """
        return self.amplitude


    def get_phase(self : WavefrontUtility) -> Array:
        """
        Accessor for the `phase` constant.

        Returns
        -------
        : Array
            The square array of pixel phases in radians.
        """
        return self.phase


    def get_offset(self : WavefrontUtility) -> Array:
        """
        Accessor for the `offset` constant.

        Returns
        -------
        : Array
            The angle that the wavefront makes with the x and 
            y planes in radians.
        """
        return self.offset


    def get_pixel_scale(self : WavefrontUtility) -> Array:
        """
        Accessor for the `pixel_scale` constant.

        Returns
        -------
        : Array
            The `pixel_scale` associated with the wavefront.
        """
        return self.pixel_scale


class PhysicalWavefrontUtility(WavefrontUtility):
    """
    Defines useful safes state constants as well as a basic 
    constructor for a safe `PhysicalWavefront`.
    """
    def __init__(self : PhysicalWavefrontUtility, /,
            wavelength : float = None, 
            offset : Array = None,
            size : int = None, 
            amplitude : Array = None, 
            phase : Array = None) -> PhysicalWavefront:
        """
        Parameters
        ----------
        wavelength : float 
            The safe wavelength to use for the constructor in meters.
        offset : Array[float]
            The safe offset to use for the constructor in radians.
        size : int
            The static size of the pixel arrays.
        amplitude : Array[float]
            The electric field amplitudes in SI units for electric
            field.
        phase : Array[float]
            The phases of each pixel in radians. 

        Returns
        -------
        : PhysicalWavefrontUtility 
            A helpful class for implementing the tests. 
        """
        super().__init__(wavelength, offset, size, amplitude, phase)


    def construct(self : PhysicalWavefrontUtility) -> PhysicalWavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        : PhysicalWavefront
            The safe testing wavefront.
        """
        wavefront = dLux\
            .PhysicalWavefront(self.wavelength, self.offset)\
            .update_phasor(self.amplitude, self.phase)\
            .set_pixel_scale(self.pixel_scale)

        return wavefront


class AngularWavefrontUtility(WavefrontUtility):
    """
    Defines useful safes state constants as well as a basic 
    constructor for a safe `PhysicalWavefront`.
    """
    def __init__(self : AngularWavefrontUtility, /,
            wavelength : float = None, 
            offset : Array = None,
            size : int = None, 
            amplitude : Array = None, 
            phase : Array = None) -> AngularWavefrontUtility:
        """
        Parameters
        ----------
        wavelength : float 
            The safe wavelength to use for the constructor in meters.
        offset : Array[float]
            The safe offset to use for the constructor in radians.
        size : int
            The static size of the pixel arrays.
        amplitude : Array[float]
            The electric field amplitudes in SI units for electric
            field.
        phase : Array[float]
            The phases of each pixel in radians. 

        Returns
        -------
        : AngularWavefrontUtility 
            A helpful class for implementing the tests. 
        """
        super().__init__(wavelength, offset, size, amplitude, phase)


    def construct(self : AngularWavefrontUtility) -> AngularWavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        : PhysicalWavefront
            The safe testing wavefront.
        """
        wavefront = dLux\
            .AngularWavefront(self.wavelength, self.offset)\
            .update_phasor(self.amplitude, self.phase)\
            .set_pixel_scale(pixel_scale)
            
        return wavefront


class GaussianWavefrontUtility(PhysicalWavefrontUtility):
    """
    Defines safe state constants and a simple constructor for a 
    safe state `GaussianWavefront` object. 

    Attributes
    ----------
    beam_radius : float
        A safe radius for the GaussianWavefront in meters.
    phase_radius : float
        A safe phase radius for the GaussianWavefront in radians.
    position : float
        A safe position for the GaussianWavefront in meters.
    """
    beam_radius : float 
    phase_radius : float
    position : float


    def __init__(self : GaussianWavefrontUtility, 
            wavelength : float = None,
            offset : Array = None,
            size : int = None,
            amplitude : Array = None,
            phase : Array = None,
            beam_radius : float = None,
            phase_radius : float = None,
            position : float = None) -> GaussianWavefrontUtility:
        """
        Parameters
        ----------
        wavelength : float 
            The safe wavelength to use for the constructor in meters.
        offset : Array[float]
            The safe offset to use for the constructor in radians.
        size : int
            The static size of the pixel arrays.
        amplitude : Array[float]
            The electric field amplitudes in SI units for electric
            field.
        phase : Array[float]
            The phases of each pixel in radians.
        beam_radius : float 
            The radius of the gaussian beam in meters.
        phase_radius : float
            The phase radius of the gaussian beam in radians.
        position : float
            The position of the gaussian beam in meters.

        Returns
        -------
        : GaussianWavefrontUtility 
            A helpful class for implementing the tests. 
        """
        super().__init__(wavelength, offset, size, amplitude, phase)
        self.beam_radius = 1. if not beam_radius else beam_radius
        self.phase_radius = numpy.inf if not phase_radius else phase_radius
        self.position = 0. if not position else position


    # TODO: get_beam_radius and get_position
    def get_phase_radius(self : GaussianWavefrontUtility) -> float:
        """
        """
        return self.phase_radius


    def construct(self : GaussianWavefrontUtility) -> GaussianWavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        : PhysicalWavefront
            The safe testing wavefront.
        """
        wavefront = dLux\
            .GaussianWavefront(self.offset,
               self.wavelength)\
            .update_phasor(self.amplitude, self.phase)\
            .set_pixel_scale(self.pixel_scale)\
            .set_position(0.)\
            .set_phase_radius(numpy.inf)\
            .set_beam_radius(1.)

        return wavefront

class PropagatorUtility(Utility):
    """
    Testing utility for the Propagator (abstract) class.

    Attributes 
    ----------
    inverse : bool
        The directionality of the generated propagators. 
    """
    inverse : bool


    def __init__(self : Utility) -> Utility:
        """
        Initialises a safe state for the Propagator attributes 
        stored as attributes in this Utility.
        """
        self.inverse = False;


    def construct(self : Utility) -> Propagator:
    def is_inverse(self : Utility) -> bool:


class VariableSamplingUtility(PropagatorUtility, UtilityUser):
    utility : WavefrontUtility = WavefrontUtility()
    pixels_out : int
    pixel_scale_out : float
   

    def __init__(self : Utility) -> Utility:
    def construct(self : Utility, /, inverse : bool = False, 
            pixels_out : int = 256, 
            pixel_scale_out : float = 1.e-3) -> Propagator:
    def get_pixels_out(self : Utility) -> int:
    def get_pixel_scale_out(self : Utility) -> float:
 

class FixedSamplingUtility(PropagatorUtility, UtilityUser):
    utility : WavefrontUtility = WavefrontUtility()


    def __init__(self : Utility) -> Utility:
    def construct(self : Utility, inverse : bool = False) -> Propagator


class PhysicalMFTUtility(VaraibleSamplingUtility, UtilityUser):
    utility : PhysicalWavefrontUtility = PhysicalWavefrontUtility()
    focal_length : float


    def __init__(self : Utility) -> Utility:
    def construct(self : Utility, inverse : bool = False, 
            pixels_out : int = 256, 
            pixel_scale_out : float = 1.e-3) -> Propagator:
    def get_focal_length(self : Utility) -> float:


class PhysicalFFTUtility(Utility, UtilityUser):
    utility : PhysicalWavefrontUtility = PhysicalWavefrontUtility()
    focal_length : float


    def __init__(self : Utility) -> Utility:
    def construct(self : Utility, inverse : bool = False, 
            pixels_out : int = 256, 
            pixel_scale_out : float = 1.e-3) -> Propagator:
    def get_focal_length(self : Utility) -> float:


class PhysicalFresnelUtility(Utility, UtilityUser):
    utility : PhysicalWavefrontUtility = PhysicalWavefrontUtility()
    focal_length : float
    focal_shift : float


    def __init__(self : Utility) -> Utility:
    def construct(self : Utility, inverse : bool = False, 
            pixels_out : int = 256, 
            pixel_scale_out : float = 1.e-3) -> Propagator:
    def get_focal_length(self : Utility) -> float:
    def get_focal_shift(self : Utility) -> float:


class AngularMFTUtility(FixedSamplingUtility, UtilityUser):
    utility : PhysicalWavefrontUtility = PhysicalWavefrontUtility()


    def __init__(self : Utility) -> Utility:
    def construct(self : Utility, inverse : bool = False, 
            pixels_out : int = 256, 
            pixel_scale_out : float = 1.e-3) -> Propagator:


class AngularFFTUtility(Utility, UtilityUser):
    utility : PhysicalWavefrontUtility = PhysicalWavefrontUtility()


    def __init__(self : Utility) -> Utility:
    def construct(self : Utility, inverse : bool = False, 
            pixels_out : int = 256, 
            pixel_scale_out : float = 1.e-3) -> Propagator:


class AngularFresnelUtility(Utility, UtilityUser):
    """
    """
    pass


class GaussianPropagatorUtility(Utility, UtilityUser):
    utility : PhysicalWavefrontUtility = PhysicalWavefrontUtility()
    distance : float


    def __init__(self : Utility) -> Utility:
    def construct(self : Utility, distance : float, 
            inverse : bool = False) -> Propagator:
