"""
Provide tools for generating and handling waveforms.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from scipy import signal


from pycbc.distributions import JointDistribution, read_params_from_config, \
    read_constraints_from_config, read_distributions_from_config
from pycbc.transforms import read_transforms_from_config, apply_transforms
from pycbc.workflow import WorkflowConfigParser
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from pycbc.psd import interpolate
from pycbc.filter import sigma
from pycbc.types.timeseries import TimeSeries


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
        self.var_args, self.static = read_params_from_config(config_file)
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

def fade_on(timeseries, alpha=1.0/4):
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


def get_waveform(waveform_params):
    """
    Takes simulation parameters, such as e.g. the approximant and the
    masses, and passes them internally to some LAL method to perform the
    actual simulation.

    Args:
        waveform_params (dict):
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
    if waveform_params['domain'] == 'time':
        simulate_waveform = get_td_waveform
        length = int(waveform_params['td_length'])
    elif waveform_params['domain'] == 'frequency':
        simulate_waveform = get_fd_waveform
        length = int(waveform_params['fd_length'])
    else:
        raise ValueError('Invalid domain! Must be "time" or "frequency"!')

    # Collect all the required parameters for the simulation from the given
    # static and variable parameters
    simulation_parameters = dict(approximant=waveform_params['approximant'],
                                 coa_phase=waveform_params['coa_phase'],
                                 delta_f=waveform_params['delta_f'],
                                 delta_t=waveform_params['delta_t'],
                                 distance=waveform_params['distance'],
                                 f_lower=waveform_params['f_lower'],
                                 inclination=waveform_params['inclination'],
                                 mass1=waveform_params['mass1'],
                                 mass2=waveform_params['mass2'],
                                 spin1z=waveform_params['spin1z'],
                                 spin2z=waveform_params['spin2z'])

    # Perform the actual simulation with the given parameters
    h_plus, h_cross = simulate_waveform(**simulation_parameters)

    # Apply the fade-on filter to them
    h_plus = fade_on(h_plus)
    h_cross = fade_on(h_cross)

    # Resize the simulated waveform to the specified length
    h_plus.resize(length)
    h_cross.resize(length)

    return h_plus, h_cross


# -----------------------------------------------------------------------------


def get_detector_signals(waveform_params, waveform):
    """
    Project the "raw" waveform = (h_plus, h_cross) onto the antenna patterns
    of the detectors in Hanford and Livingston (this requires the position
    of the source in the sky, which is contained in waveform_params).

    Args:
        waveform_params: (dict)
            This dictionary must contain at least the following parameters:
                - 'ra' = right ascension
                - 'dec' = declination
                - 'polarization' = polarization
                - ' event_time' = event time (by convention, the H1 time)
        waveform: (tuple)
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
    event_time = waveform_params['event_time']

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
        if waveform_params['domain'] == 'time':
            offset = 100 + delta_t_h1 + detector_signal.start_time
            detector_signal = detector_signal.cyclic_time_shift(offset)
            detector_signal.start_time = event_time - 100
        elif waveform_params['domain'] == 'frequency':
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
                    event_time,
                    waveform_params=None):

    # -------------------------------------------------------------------------
    # Define shortcuts for some elements of self.static_arguments
    # -------------------------------------------------------------------------

    # Read out frequency-related arguments
    sampling_rate = static_arguments['sampling_rate']
    f_lower = static_arguments['f_lower']
    delta_f = static_arguments['delta_f']

    # Get how many seconds before and after the event time to use
    seconds_before_event = static_arguments['seconds_before_event']
    seconds_after_event = static_arguments['seconds_after_event']

    # -------------------------------------------------------------------------
    # Select the background noise for the sample
    # -------------------------------------------------------------------------

    # TODO: This still needs to be implemented!
    noise = dict()

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
        waveform = get_waveform(waveform_params=waveform_params)

        # Get the detector signals by projecting on the antenna patterns
        detector_signals = \
            get_detector_signals(waveform_params=waveform_params,
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

        # Whiten the strain (using the built-in whitening of PyCBC)
        strain[det] = strain[det].whiten(4, 4)

        # Get the limits for the bandpass
        bandpass_lower = int(waveform_params['bandpass_lower'])
        bandpass_upper = int(waveform_params['bandpass_upper'])

        # Apply a high-pass to remove everything below `bandpass_lower`;
        # If bandpass_lower = 0, do not apply any high-pass filter.
        if bandpass_lower != 0:
            strain[det] = strain[det].highpass_fir(frequency=bandpass_lower,
                                                   order=512)

        # Apply a low-pass filter to remove everything above `bandpass_upper`.
        # If bandpass_upper = sampling rate, do not apply any low-pass filter.
        if bandpass_upper != sampling_rate:
            strain[det] = strain[det].lowpass_fir(frequency=bandpass_upper,
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
