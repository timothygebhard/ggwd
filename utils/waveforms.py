"""
Provide tools for generating and processing GW waveforms.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import numpy as np
from scipy import signal

from pycbc.distributions import JointDistribution, read_params_from_config, \
    read_constraints_from_config, read_distributions_from_config
from pycbc.transforms import apply_transforms, read_transforms_from_config
from pycbc.workflow import WorkflowConfigParser
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class WaveformParameterGenerator(object):

    def __init__(self,
                 config_file,
                 random_seed):

        # Fix the seed for the random number generator
        np.random.seed(random_seed)

        # Read in the config.ini file
        config_file = WorkflowConfigParser(configFiles=config_file)

        # Extract variable arguments and constraints
        # We don't need the static_args here, hence they do not get amended.
        self.var_args, _ = read_params_from_config(config_file)
        self.constraints = read_constraints_from_config(config_file)

        # Extract distributions
        dist = read_distributions_from_config(config_file)

        # Extract transformations
        self.trans = read_transforms_from_config(config_file)

        # Set up a joint distribution to sample from
        self.pval = JointDistribution(self.var_args,
                                      *dist,
                                      **{'constraints': self.constraints})

    # -------------------------------------------------------------------------

    def draw(self):
        """
        Sample from the joint distribution and construct a dict that
        connects the parameter names with values generated for them.

        Returns:
            A dictionary containing a set of randomly sampled waveform
            parameters (e.g., masses, spins, position, ...).
        """
        values = apply_transforms(self.pval.rvs(), self.trans)[0]
        result = dict(zip(self.var_args, values))

        return result


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def fade_on(timeseries,
            alpha=0.25):
    """
    Take a PyCBC time series and use a one-sided Tukey window to "fade
    on" the waveform (to reduce discontinuities in the amplitude).

    Args:
        timeseries (pycbc.types.timeseries.TimeSeries): The PyCBC
            TimeSeries object to be faded on.
        alpha (float): The alpha parameter for the Tukey window.

    Returns:
        The `timeseries` which has been faded on.
    """

    # Save the parameters from the time series we are about to fade on
    delta_t = timeseries.delta_t
    epoch = timeseries.start_time
    duration = timeseries.duration
    sample_rate = timeseries.sample_rate

    # Create a one-sided Tukey window for the turn on
    window = signal.tukey(int(duration * sample_rate), alpha)
    window[int(0.5*len(window)):] = 1

    # Apply the one-sided Tukey window for the fade-on
    ts = window * np.array(timeseries)

    # Create and return a TimeSeries object again from the resulting array
    # using the original parameters (delta_t and epoch) of the time series
    return TimeSeries(initial_array=ts,
                      delta_t=delta_t,
                      epoch=epoch)


def get_waveform(static_arguments,
                 waveform_params):
    """
    Simulate a waveform (using methods provided by PyCBC / LALSuite)
    based on the `static_arguments` (which define, e.g., the waveform
    model to be used) and the `waveform_params`, which specify the
    physical parameters of the waveform (e.g., the masses and spins).

    Args:
        static_arguments (dict): The static arguments (e.g., the
            waveform approximant and the sampling rate) defined in the
            waveform parameters config file.
        waveform_params (dict): There does not seem to exist any
            comprehensive documentation  of which parameters are
            actually supported by the methods
                `get_td_waveform()` and `get_fd_waveform()`,
            or rather the underlying LALSuite methods.
            So far, we are using the following simulation parameters:
                {'approximant', 'coa_phase', 'delta_f', 'delta_t',
                 'distance', 'f_lower', 'inclination', 'mass1,
                 'mass2', 'spin1z', 'spin2z'}
            NOTE: If you want to use a different waveform model or a
            different parameter space, you may need to edit this
            function according to your needs!

    Returns:
        A tuple `(h_plus, h_cross)` with the two polarization modes of
        the simulated waveform, resized to the desired length.
    """

    # Check if we are using a time domain (TD) or frequency domain (FD)
    # approximant and retrieve the required parameters for the simulation
    if static_arguments['domain'] == 'time':
        simulate_waveform = get_td_waveform
        length = int(static_arguments['td_length'])
    elif static_arguments['domain'] == 'frequency':
        simulate_waveform = get_fd_waveform
        length = int(static_arguments['fd_length'])
    else:
        raise ValueError('Invalid domain! Must be "time" or "frequency"!')

    # Collect all the required parameters for the simulation from the given
    # static and variable parameters
    simulation_parameters = dict(approximant=static_arguments['approximant'],
                                 coa_phase=waveform_params['coa_phase'],
                                 delta_f=static_arguments['delta_f'],
                                 delta_t=static_arguments['delta_t'],
                                 distance=static_arguments['distance'],
                                 f_lower=static_arguments['f_lower'],
                                 inclination=waveform_params['inclination'],
                                 mass1=waveform_params['mass1'],
                                 mass2=waveform_params['mass2'],
                                 spin1z=waveform_params['spin1z'],
                                 spin2z=waveform_params['spin2z'])

    # Perform the actual simulation with the given parameters
    h_plus, h_cross = simulate_waveform(**simulation_parameters)

    # Apply the fade-on filter to them
    h_plus = fade_on(h_plus, alpha=static_arguments['tukey_alpha'])
    h_cross = fade_on(h_cross, alpha=static_arguments['tukey_alpha'])

    # Resize the simulated waveform to the specified length
    h_plus.resize(length)
    h_cross.resize(length)

    return h_plus, h_cross


# -----------------------------------------------------------------------------


def get_detector_signals(static_arguments,
                         waveform_params,
                         event_time,
                         waveform):
    """
    Project the raw `waveform` = `(h_plus, h_cross)` onto the antenna
    patterns of the detectors in Hanford and Livingston. This requires
    the position of the source in the sky, which is contained in
    `waveform_params`.

    Args:
        static_arguments (dict): The static arguments (e.g., the
            waveform approximant and the sampling rate) defined in the
            waveform parameters config file.
        waveform_params (dict): This dictionary must contain at least
            the following parameters:
                - 'ra' = right ascension
                - 'dec' = declination
                - 'polarization' = polarization
                - 'event_time' = event time (by convention the H1 time)
        event_time (int): The (randomly sampled) GPS time for the event.
        waveform (tuple): The tuple `(h_plus, h_cross)` that is usually
            generated by get_waveform().

    Returns:
        A dictionary with keys {'H1', 'L1'} that contains the pure
        signal as it would be observed at Hanford and Livingston.
    """

    # Retrieve the two polarization modes from the waveform tuple
    h_plus, h_cross = waveform

    # Extract the parameters we will need later for the projection
    right_ascension = waveform_params['ra']
    declination = waveform_params['dec']
    polarization = waveform_params['polarization']

    # Store the detector signals we will get through projection
    detector_signals = {}

    # Set up detectors
    detectors = {'H1': Detector('H1'), 'L1': Detector('L1')}

    # Loop over both detectors and calculate the signal we would see there
    for detector_name in ('H1', 'L1'):

        # Set up the detector based on its name
        detector = detectors[detector_name]

        # Calculate the antenna pattern for this detector
        f_plus, f_cross = \
            detector.antenna_pattern(right_ascension=right_ascension,
                                     declination=declination,
                                     polarization=polarization,
                                     t_gps=100)

        # Calculate the time offset from H1 for this detector
        delta_t_h1 = \
            detector.time_delay_from_detector(other_detector=detectors['H1'],
                                              right_ascension=right_ascension,
                                              declination=declination,
                                              t_gps=100)

        # Project the waveform onto the antenna pattern
        detector_signal = f_plus * h_plus + f_cross * h_cross

        # Map the signal from geocentric coordinates to the specific
        # reference frame of the detector. This depends on whether we have
        # simulated the waveform in the time or frequency domain:
        if static_arguments['domain'] == 'time':
            offset = 100 + delta_t_h1 + detector_signal.start_time
            detector_signal = detector_signal.cyclic_time_shift(offset)
            detector_signal.start_time = event_time - 100
        elif static_arguments['domain'] == 'frequency':
            offset = 100 + delta_t_h1
            detector_signal = detector_signal.cyclic_time_shift(offset)
            detector_signal.start_time = event_time - 100
            detector_signal = detector_signal.to_timeseries()
        else:
            raise ValueError('Invalid domain! Must be "time" or "frequency"!')

        # Store the result
        detector_signals[detector_name] = detector_signal

    return detector_signals
