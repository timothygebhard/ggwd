"""
This script uses the pycbc.catalog module to create a sample file
containing the pre-processed (i.e., re-sampled, whitened and
band-passed) data for all merger events observed so far.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import argparse
import os
import time
import h5py

from utils.configfiles import read_ini_config

from pycbc.catalog import Catalog
from pycbc.types.timeseries import TimeSeries


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start_time = time.time()
    print('')
    print('GENERATE SAMPLE FILE FOR REAL GRAVITATIONAL-WAVE EVENTS')
    print('')

    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser and add arguments
    parser = argparse.ArgumentParser(description='Generate real events file.')
    parser.add_argument('--ini-config-file',
                        help='Name of the *.ini configuration file, whose '
                             '[static_args] section contains information '
                             'about the pre-processing (e.g., sampling rate, '
                             'whitening, band-passing, etc.).'
                             'Default: waveform_params.ini.',
                        default='waveform_params.ini')

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    command_line_arguments = vars(parser.parse_args())
    print('Done!')

    # -------------------------------------------------------------------------
    # Read in INI config file specifying the static_args and variable_args
    # -------------------------------------------------------------------------

    # Build the full path to the waveform params file
    ini_config_name = command_line_arguments['ini_config_file']
    ini_config_path = os.path.join('.', 'config_files', ini_config_name)

    # Read in the static_arguments (and ignore the variable_arguments)
    print('Reading and validating in INI configuration file...', end=' ')
    _, static_arguments = read_ini_config(ini_config_path)
    print('Done!\n')

    # -------------------------------------------------------------------------
    # Create shortcuts for pre-processing parameters, and print their values
    # -------------------------------------------------------------------------

    # Define shortcuts for pre-processing parameters
    bandpass_lower = static_arguments['bandpass_lower']
    bandpass_upper = static_arguments['bandpass_upper']
    original_sampling_rate = static_arguments['original_sampling_rate']
    target_sampling_rate = static_arguments['target_sampling_rate']
    max_filter_duration = static_arguments['whitening_max_filter_duration']
    segment_duration = static_arguments['whitening_segment_duration']

    # Print the pre-processing parameters we will be using
    print('Using the following pre-processing parameters:')
    for param in sorted(['original_sampling_rate', 'target_sampling_rate',
                         'whitening_segment_duration', 'bandpass_lower',
                         'bandpass_upper', 'whitening_max_filter_duration']):
        print('-- {:32}'.format(param + ':'), static_arguments[param])
    print('')

    # -------------------------------------------------------------------------
    # Prepare the HDF file in which we will save the sample
    # -------------------------------------------------------------------------

    # Make sure the output directory exists
    output_dir = os.path.join('.', 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Construct path to results file and open it to ensure its empty
    results_file = os.path.join(output_dir, 'real_events.hdf')
    with h5py.File(results_file, 'w'):
        pass

    # -------------------------------------------------------------------------
    # Create an event catalog and loop over all events
    # -------------------------------------------------------------------------

    # Set up a new catalog
    catalog = Catalog()

    # Loop over the events it contains
    for event in sorted(catalog.names):

        print('Processing', event.upper())
        print(64 * '-')

        # Get the strain for detectors H1 and L1 (if necessary, this will
        # download the  strain from GWOSC)
        strain = dict(H1=catalog[event].strain('H1'),
                      L1=catalog[event].strain('L1'),)

        # ---------------------------------------------------------------------
        # Re-sample to the desired target_sampling_rate
        # ---------------------------------------------------------------------

        print('Re-sampling to {} Hz...'.format(target_sampling_rate), end=' ')

        # Compute the re-sampling factor
        resampling_factor = int(static_arguments['original_sampling_rate'] /
                                static_arguments['target_sampling_rate'])

        # Re-sample the time series for both detectors
        for det in ('H1', 'L1'):
            strain[det] = \
                TimeSeries(initial_array=strain[det][::resampling_factor],
                           delta_t=1.0 / target_sampling_rate,
                           epoch=strain[det].start_time)
        
        print('Done!')

        # ---------------------------------------------------------------------
        # Whiten and band-pass the data
        # ---------------------------------------------------------------------

        for det in ('H1', 'L1'):

            # Whiten the 512 second stretch with a 4 second window
            print('Whitening the data...', end=' ')
            strain[det] = \
                strain[det].whiten(segment_duration=segment_duration,
                                   max_filter_duration=max_filter_duration,
                                   remove_corrupted=False)
            print('Done!')

            # Apply a high-pass to remove everything below `bandpass_lower`
            print('High-passing the data...', end=' ')
            if bandpass_lower != 0:
                strain[det] = \
                    strain[det].highpass_fir(frequency=bandpass_lower,
                                             remove_corrupted=False,
                                             order=512)
            print('Done!')

            # Apply a low-pass to remove everything above `bandpass_upper`
            print('Low-passing the data...', end=' ')
            if bandpass_upper != target_sampling_rate:
                strain[det] = \
                    strain[det].lowpass_fir(frequency=bandpass_upper,
                                            remove_corrupted=False,
                                            order=512)
            print('Done!')

        # ---------------------------------------------------------------------
        # Select interval around event to remove corrupted edges
        # ---------------------------------------------------------------------

        print('Selecting interval around event...', end=' ')
        for det in ('H1', 'L1'):
            strain[det] = strain[det].time_slice(catalog[event].time - 8,
                                                 catalog[event].time + 8)
        print('Done!')

        # ---------------------------------------------------------------------
        # Save the resulting samples as HDF files again
        # ---------------------------------------------------------------------

        print('Saving strain to HDF file...', end=' ')
        with h5py.File(results_file, 'a') as hdf_file:

            # Create a new group for this event
            hdf_file.create_group(name=event)

            # Save the post-processed strain for the event
            hdf_file[event].create_dataset(name='h1_strain',
                                           data=strain['H1'])
            hdf_file[event].create_dataset(name='l1_strain',
                                           data=strain['L1'])
        print('Done!')

        print(64 * '-')
        print('')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    runtime = time.time() - script_start_time
    print('This took {:.1f} seconds in total!'.format(runtime))
    print('')
