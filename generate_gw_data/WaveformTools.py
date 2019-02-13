"""
Provide tools for generating and handling waveforms.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import sys
import os
import numpy as np
from scipy import signal
from copy import deepcopy

from lal import LIGOTimeGPS
from pycbc.distributions import JointDistribution, read_params_from_config, \
    read_constraints_from_config, read_distributions_from_config
from pycbc.transforms import read_transforms_from_config, apply_transforms
from pycbc.workflow import WorkflowConfigParser
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from pycbc.psd import interpolate
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.filter import sigma
from pycbc.types.timeseries import TimeSeries

# Here we need to add the parent directory to the $PYTHONPATH, because
# apparently Python does not have a less hacky way of importing from a
# sibling directory if you are "just" in a script and not in a package
sys.path.insert(0, os.path.realpath('..'))

# Now we can even import from generate_gw_data without PyCharm complaining!
from generate_gw_data.HDFTools import get_strain_from_hdf_file  # noqa


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class WaveformParameterGenerator(object):

    def __init__(self, config_file, random_seed):

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
            A dictionary containing some randomly sampled waveform
            parameters.
        """
        values = apply_transforms(self.pval.rvs(), self.trans)[0]
        result = dict(zip(self.var_args, values))

        return result


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def fade_on(timeseries, alpha=0.25):
    """
    Take a PyCBC time series and use a one-sided Tukey window to "fade on"
    the waveform (to reduce discontinuities in the amplitude).

    Args:
        timeseries: The PyCBC TimeSeries object to be faded on.
        alpha: The alpha parameter for the Tukey window.

    Returns:
        The timeseries which has been faded on.
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
    return TimeSeries(initial_array=ts, delta_t=delta_t, epoch=epoch)


def get_waveform(static_arguments,
                 waveform_params):
    """
    Takes simulation parameters, such as e.g. the approximant and the
    masses, and passes them internally to some LAL method to perform the
    actual simulation.

    Args:
        static_arguments: dict
            # TODO
        waveform_params: dict
            There does not seem to be any comprehensive documentation of
            which parameters are actually supported by get_td_waveform and
            get_fd_waveform (or rather the underlying LAL methods).
            So far, we are using the following parameters:
                - approximant
                - coa_phase
                - delta_f
                - delta_t
                - distance
                - f_lower
                - inclination
                - mass1
                - mass2
                - spin1z
                - spin2z

    Returns:
        A tuple (h_plus, h_cross) with the two polarization modes of the
        simulated waved, resized to the length specified in
        arguments['td_length] or arguments['fd_length].
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


def amend_static_arguments(static_args_):
    """
    Amend the static_args from the *.ini configuration file by adding the
    parameters that can be computed directly from others (more intuitive
    ones).

    Args:
        static_args_: dict
            The original static_args, as read from the *.ini config file.

    Returns:
        The amended static_args, where implicitly defined variables have
        been added.
    """

    # Create a copy of the original static_args
    static_args = deepcopy(static_args_)

    # If necessary, add delta_t = 1 / target_sampling_rate
    if 'delta_t' not in static_args.keys():
        static_args['delta_t'] = 1.0 / static_args['target_sampling_rate']

    # If necessary, add delta_f = 1 / waveform_length
    if 'delta_f' not in static_args.keys():
        static_args['delta_f'] = 1.0 / static_args['waveform_length']

    # If necessary, add td_length = waveform_length * target_sampling_rate
    if 'td_length' not in static_args.keys():
        static_args['td_length'] = int(static_args['waveform_length'] *
                                       static_args['target_sampling_rate'])

    # If necessary, add fd_length = td_length / 2 + 1
    if 'fd_length' not in static_args.keys():
        static_args['fd_length'] = int(static_args['td_length'] / 2.0 + 1)

    return static_args

# -----------------------------------------------------------------------------


def get_detector_signals(static_arguments,
                         waveform_params,
                         event_time,
                         waveform):
    """
    Project the "raw" waveform = (h_plus, h_cross) onto the antenna patterns
    of the detectors in Hanford and Livingston (this requires the position
    of the source in the sky, which is contained in waveform_params).

    Args:
        static_arguments: dict
            The static arguments (e.g., the waveform approximant and the
            sampling rate) defined in the waveform parameters config file.
        waveform_params: dict
            This dictionary must contain at least the following parameters:
                - 'ra' = right ascension
                - 'dec' = declination
                - 'polarization' = polarization
                - ' event_time' = event time (by convention, the H1 time)
        event_time: int
            The (randomly sampled) GPS time for the event.
        waveform: tuple
            The tuple (h_plus, h_cross) that is usually generated by
            get_waveform().

    Returns:
        A dictionary (keys 'H1', 'L1') that contains the signal as it would
        be observed at Hanford and Livingston.
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

        # Set up the detector from its name
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


# -----------------------------------------------------------------------------


def generate_sample(static_arguments,
                    event_tuple,
                    delta_t,
                    waveform_params=None):
    """

    Args:
        static_arguments:
        event_tuple:
        delta_t:
        waveform_params:

    Returns:

    """

    # -------------------------------------------------------------------------
    # Define shortcuts for some elements of self.static_arguments
    # -------------------------------------------------------------------------

    # Read out frequency-related arguments
    original_sampling_rate = static_arguments['original_sampling_rate']
    target_sampling_rate = static_arguments['target_sampling_rate']
    f_lower = static_arguments['f_lower']
    delta_f = static_arguments['delta_f']
    fd_length = static_arguments['fd_length']

    # Get how many seconds before and after the event time to use
    seconds_before_event = static_arguments['seconds_before_event']
    seconds_after_event = static_arguments['seconds_after_event']

    # Get the event time and the dict containing the HDF file path
    event_time, hdf_file_paths = event_tuple

    # -------------------------------------------------------------------------
    # Get the background noise (either from data or synthetically)
    # -------------------------------------------------------------------------

    # A NOTE OF CAUTION!
    #
    # Please not that 'delta_t' unfortunately has a double meaning here:
    #
    # 1. The delta_t that is passed to this function, i.e. generate_sample(),
    #    refers to the length of the piece of noise:
    #    In case we are sampling real noise, we select the interval
    #    [event_time - delta_t, event_time + delta_t] from the HDF files.
    #    In the case of synthetic noise, we simply generate random noise of
    #    length 2 * delta_t (in seconds).
    # 2. The delta_t that is expected by noise_from_pdf() is simply the
    #    inverse of the target sampling rate, i.e. the time difference between
    #    two consecutive samples!

    # If the event_time is None, we generate synthetic noise
    if hdf_file_paths is None:

        # Create an artificial PSD for the noise
        # TODO: Is this the best choice for this task?
        psd = aLIGOZeroDetHighPower(length=fd_length,
                                    delta_f=delta_f,
                                    low_freq_cutoff=f_lower)

        # Actually generate the noise using the PSD and LALSimulation
        noise = dict()
        for det in ('H1', 'L1'):

            # Generate the noise for this detector
            noise[det] = \
                noise_from_psd(length=(2 * delta_t * target_sampling_rate),
                               delta_t=(1.0 / target_sampling_rate),
                               psd=psd,
                               seed=None)

            # Manually fix the noise start time to match the fake event time
            # that we are using. For some reason, the correct setter method
            # for this property does not work?!
            # noinspection PyProtectedMember
            noise[det]._epoch = LIGOTimeGPS(1234567890 - delta_t)

    # Otherwise we select the noise from the corresponding HDF file
    else:

        kwargs = dict(hdf_file_paths=hdf_file_paths,
                      gps_time=event_time,
                      delta_t=delta_t,
                      original_sampling_rate=original_sampling_rate,
                      target_sampling_rate=target_sampling_rate,
                      as_pycbc_timeseries=True)
        noise = get_strain_from_hdf_file(**kwargs)

    # -------------------------------------------------------------------------
    # If applicable, make an injection
    # -------------------------------------------------------------------------

    # If no waveform parameters are given, we are not making an injection.
    # In this case, there are no detector signals and no injection
    # parameters, and the strain is simply equal to the noise
    if waveform_params is None:
        detector_signals = None
        injection_parameters = None
        strain = noise

    # Otherwise, we need to simulate a waveform for the given waveform_params
    # and add it into the noise to create the strain
    else:

        # ---------------------------------------------------------------------
        # Simulate the waveform with the given injection parameters
        # ---------------------------------------------------------------------

        # Actually simulate the waveform with these parameters
        waveform = get_waveform(static_arguments=static_arguments,
                                waveform_params=waveform_params)

        # Get the detector signals by projecting on the antenna patterns
        detector_signals = \
            get_detector_signals(static_arguments=static_arguments,
                                 waveform_params=waveform_params,
                                 event_time=event_time,
                                 waveform=waveform)

        # ---------------------------------------------------------------------
        # Add the waveform into the noise as is to calculate the NOMF-SNR
        # ---------------------------------------------------------------------

        # Store the dummy strain, the PSDs and the SNRs for the two detectors
        strain_ = {}
        psds = {}
        snrs = {}

        # Calculate these quantities for both detectors
        for det in ('H1', 'L1'):

            # Add the simulated waveform into the noise to get the dummy strain
            strain_[det] = noise[det].add_into(detector_signals[det])

            # Estimate the Power Spectral Density from the dummy strain
            psds[det] = strain_[det].psd(4)
            psds[det] = interpolate(psds[det], delta_f=delta_f)

            # Use the PSD estimate to calculate the optimal matched
            # filtering SNR for this injection and this detector
            snrs[det] = sigma(htilde=detector_signals[det],
                              psd=psds[det],
                              low_frequency_cutoff=f_lower)

        # Calculate the network optimal matched filtering SNR for this
        # injection (which we need for scaling to the chosen injection SNR)
        nomf_snr = (snrs['H1'] ** 2.0 + snrs['L1'] ** 2.0) ** 0.5

        # ---------------------------------------------------------------------
        # Add the waveform into the noise with the chosen injection SNR
        # ---------------------------------------------------------------------

        # Compute the rescaling factor
        injection_snr = waveform_params['injection_snr']
        scale_factor = 1.0 * injection_snr / nomf_snr

        strain = {}
        for det in ('H1', 'L1'):

            # Add the simulated waveform into the noise, using a scaling
            # factor to ensure that the resulting NOMF-SNR equals the chosen
            # injection SNR
            strain[det] = noise[det].add_into(scale_factor *
                                              detector_signals[det])

        # ---------------------------------------------------------------------
        # Store some information about the injection we just made
        # ---------------------------------------------------------------------

        # Store the information we have computed ourselves
        injection_parameters = {'scale_factor': scale_factor,
                                'nomf_snr': nomf_snr,
                                'h1_snr': snrs['H1'],
                                'l1_snr': snrs['L1'],
                                'injection_snr': injection_snr}

        # Also add the waveform parameters we have sampled
        for key, value in waveform_params.iteritems():
            injection_parameters[key] = value

    # -------------------------------------------------------------------------
    # Whiten and bandpass the strain (regardless whether an injection was made)
    # -------------------------------------------------------------------------

    for det in ('H1', 'L1'):

        # Get the whitening parameters
        segment_duration = static_arguments['whitening_segment_duration']
        max_filter_duration = static_arguments['whitening_max_filter_duration']

        # Whiten the strain (using the built-in whitening of PyCBC)
        # We don't need to remove the corrupted samples here, because we
        # crop the strain later on
        strain[det] = \
            strain[det].whiten(segment_duration=segment_duration,
                               max_filter_duration=max_filter_duration,
                               remove_corrupted=False)

        # Get the limits for the bandpass
        bandpass_lower = static_arguments['bandpass_lower']
        bandpass_upper = static_arguments['bandpass_upper']

        # Apply a high-pass to remove everything below `bandpass_lower`;
        # If bandpass_lower = 0, do not apply any high-pass filter.
        if bandpass_lower != 0:
            strain[det] = strain[det].highpass_fir(frequency=bandpass_lower,
                                                   remove_corrupted=False,
                                                   order=512)

        # Apply a low-pass filter to remove everything above `bandpass_upper`.
        # If bandpass_upper = sampling rate, do not apply any low-pass filter.
        if bandpass_upper != target_sampling_rate:
            strain[det] = strain[det].lowpass_fir(frequency=bandpass_upper,
                                                  remove_corrupted=False,
                                                  order=512)

    # -------------------------------------------------------------------------
    # Cut strain (and signal) time series to the pre-specified length
    # -------------------------------------------------------------------------

    for det in ('H1', 'L1'):

        # Define some shortcuts for slicing
        a = event_time - seconds_before_event
        b = event_time + seconds_after_event

        # Cut the strain to the desired length
        strain[det] = strain[det].time_slice(a, b)

        # If we've made an injection, also cut the simulated signal
        if waveform_params is not None:

            # Cut the detector signals to the specified length
            detector_signals[det] = detector_signals[det].time_slice(a, b)

            # Also add the detector signals to the injection parameters
            injection_parameters['h1_signal'] = \
                np.array(detector_signals['H1'])
            injection_parameters['l1_signal'] = \
                np.array(detector_signals['L1'])

    # -------------------------------------------------------------------------
    # Collect all available information about this sample and return results
    # -------------------------------------------------------------------------

    sample = {'event_time': event_time,
              'h1_strain': np.array(strain['H1']),
              'l1_strain': np.array(strain['L1'])}

    return sample, injection_parameters
