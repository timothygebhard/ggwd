"""
Provide the :func:`generate_sample()` method, which is at the heart of
the sample generation process.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import numpy as np

from lal import LIGOTimeGPS
from pycbc.psd import interpolate
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.filter import sigma

from .hdffiles import get_strain_from_hdf_file
from .waveforms import get_detector_signals, get_waveform


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def generate_sample(static_arguments,
                    event_tuple,
                    waveform_params=None):
    """
    Generate a single sample (or example) by taking a piece of LIGO
    background noise (real or synthetic, depending on `event_tuple`),
    optionally injecting a simulated waveform (depending on
    `waveform_params`) and post-processing the result (whitening,
    band-passing).
    
    Args:
        static_arguments (dict): A dictionary containing global
            technical parameters for the sample generation, for example
            the target_sampling_rate of the output.
        event_tuple (tuple): A tuple `(event_time, file_path)`, which
            specifies the GPS time at which to make an injection and
            the path of the HDF file which contains said GPS time.
            If `file_path` is `None`, synthetic noise will be used
            instead and the `event_time` only serves as a seed for
            the corresponding (random) noise generator.
        waveform_params (dict): A dictionary containing the randomly
            sampled parameters that are passed as inputs to the
            waveform model (e.g., the masses, spins, position, ...).

    Returns:
        A tuple `(sample, injection_parameters)`, which contains the
        generated `sample` itself (a dict with keys `{'event_time',
        'h1_strain', 'l1_strain'}`), and the `injection_parameters`,
        which are either `None` (in case no injection was made), or a
        dict containing the `waveform_params` and some additional
        parameters (e.g., single detector SNRs).
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

    # Get the width of the noise sample that we either select from the raw
    # HDF files, or generate synthetically
    noise_interval_width = static_arguments['noise_interval_width']

    # Get how many seconds before and after the event time to use
    seconds_before_event = static_arguments['seconds_before_event']
    seconds_after_event = static_arguments['seconds_after_event']

    # Get the event time and the dict containing the HDF file path
    event_time, hdf_file_paths = event_tuple

    # -------------------------------------------------------------------------
    # Get the background noise (either from data or synthetically)
    # -------------------------------------------------------------------------

    # If the event_time is None, we generate synthetic noise
    if hdf_file_paths is None:

        # Create an artificial PSD for the noise
        # TODO: Is this the best choice for this task?
        psd = aLIGOZeroDetHighPower(length=fd_length,
                                    delta_f=delta_f,
                                    low_freq_cutoff=f_lower)

        # Actually generate the noise using the PSD and LALSimulation
        noise = dict()
        for i, det in enumerate(('H1', 'L1')):

            # Compute the length of the noise sample in time steps
            noise_length = noise_interval_width * target_sampling_rate

            # Generate the noise for this detector
            noise[det] = noise_from_psd(length=noise_length,
                                        delta_t=(1.0 / target_sampling_rate),
                                        psd=psd,
                                        seed=(2 * event_time + i))

            # Manually fix the noise start time to match the fake event time.
            # However, for some reason, the correct setter method seems broken?
            start_time = event_time - noise_interval_width / 2
            # noinspection PyProtectedMember
            noise[det]._epoch = LIGOTimeGPS(start_time)

    # Otherwise we select the noise from the corresponding HDF file
    else:

        kwargs = dict(hdf_file_paths=hdf_file_paths,
                      gps_time=event_time,
                      interval_width=noise_interval_width,
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
        nomf_snr = np.sqrt(snrs['H1']**2 + snrs['L1']**2)

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
                                'h1_snr': snrs['H1'],
                                'l1_snr': snrs['L1']}

        # Also add the waveform parameters we have sampled
        for key, value in waveform_params.items():
            injection_parameters[key] = value

    # -------------------------------------------------------------------------
    # Whiten and bandpass the strain (also for noise-only samples)
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

    # The whitened strain is numerically on the order of O(1), so we can save
    # it as a 32-bit float (unlike the original signal, which is down to
    # O(10^-{30}) and thus requires 64-bit floats).
    sample = {'event_time': event_time,
              'h1_strain': np.array(strain['H1']).astype(np.float32),
              'l1_strain': np.array(strain['L1']).astype(np.float32)}

    return sample, injection_parameters
