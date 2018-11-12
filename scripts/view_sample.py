"""
Inspect the results of the sample generation process.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import argparse
import numpy as np
import os
import sys
import time


# Here we need to add the parent directory to the $PYTHONPATH, because
# apparently Python does not have a less hacky way of importing from a
# sibling directory if you are "just" in a script and not in a package
sys.path.insert(0, os.path.realpath('..'))

from generate_gw_data.SampleFileTools import SampleFile  # noqa

# We need to load a different backend for matplotlib before import plt to
# avoid problems on environments where the $DISPLAY variable is not set.
import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def view_sample(sample_id,
                all_samples,
                before=0.15,
                after=0.05,
                show_plot=True,
                save_plot=None):

    # Select the sample (i.e., the row from the data frame of samples)
    try:
        sample = all_samples.loc[sample_id]
    except KeyError:
        raise KeyError('Given sample_id is too big! Maximum value = {}'.
                       format(len(all_samples) - 1))

    # Check if the sample we have received contains an injection or not
    if 'h1_signal' in sample.keys():
        has_injection = isinstance(sample['h1_signal'], np.ndarray)
    else:
        has_injection = False

    # Read out and construct some necessary values for plotting

    seconds_before_event = float(sample['seconds_before_event'])
    seconds_after_event = float(sample['seconds_after_event'])
    target_sampling_rate = float(sample['target_sampling_rate'])
    sample_length = float(sample['sample_length'])

    # Create a grid on which the sample can be plotted so that the
    # `event_time` is at position 0
    grid = np.linspace(0 - seconds_before_event, 0 + seconds_after_event,
                       int(target_sampling_rate * sample_length))

    # Create subplots for H1 and L1
    fig, axes1 = plt.subplots(nrows=2)

    # If the sample has an injection, we need a second y-axis to plot the
    # (unwhitened) detector signals
    if has_injection:
        axes2 = [ax.twinx() for ax in axes1]
    else:
        axes2 = None

    # Plot the strains for H1 and L1
    for i, (det_name, det_string) in enumerate([('H1', 'h1_strain'),
                                                ('L1', 'l1_strain')]):

        axes1[i].plot(grid, sample[det_string], color='C0')
        axes1[i].set_xlim(-before, after)
        axes1[i].set_ylim(-150, 150)
        axes1[i].tick_params('y', colors='C0', labelsize=8)
        axes1[i].set_ylabel('Amplitude of Whitened Strain ({})'
                            .format(det_name), color='C0', fontsize=8)

    # If applicable, also plot the detector signals for H1 and L1
    if has_injection:

        # Get the maximum value of the detector signal (for norming them)
        maximum = max(np.max(sample['h1_signal']), np.max(sample['l1_signal']))

        for i, (det_name, det_string) in enumerate([('H1', 'h1_signal'),
                                                    ('L1', 'l1_signal')]):
            axes2[i].plot(grid, sample[det_string] / maximum,
                          color='C1')
            axes2[i].set_xlim(-before, after)
            axes2[i].set_ylim(-1.2, 1.2)
            axes2[i].tick_params('y', colors='C1', labelsize=8)
            axes2[i].set_ylabel('Rescaled Amplitude of Simulated\n'
                                'Detector Signal ({})'.format(det_name),
                                color='C1', fontsize=8)

    # Also add the injection parameters
    if has_injection:
        keys = ('mass1', 'mass2', 'spin1z', 'spin2z', 'ra', 'dec',
                'coa_phase', 'inclination', 'polarization', 'injection_snr')
        string = ', '.join(['{} = {:.2f}'.format(_, float(sample[_]))
                            for _ in keys])
    else:
        string = '(sample does not contain an injection)'
    plt.figtext(0.5, 0.9, 'Injection Parameters:\n' + string,
                fontsize=8, ha='center')

    # Add a vertical line at the position of the event (x=0)
    axes1[0].axvline(x=0, color='black', ls='--', lw=1)
    axes1[1].axvline(x=0, color='black', ls='--', lw=1)

    # Set x-labels
    axes1[0].set_xticklabels([])
    axes1[1].set_xlabel('Time from `event_time` (in seconds)')

    # Adjust the size and spacing of the subplots
    plt.gcf().set_size_inches(12, 6, forward=True)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.subplots_adjust(wspace=0, hspace=0)

    # Add a title
    plt.suptitle('Sample #{}'.format(sample_id), y=0.975)

    # If desired, save the resulting plot
    if save_plot is not None:
        plt.savefig(save_plot)

    # If desired, show the resulting plot
    if show_plot:
        plt.show()


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Disable output buffering ('flush' option is not available for Python 2)
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # Start the stopwatch
    script_start_time = time.time()
    print('')
    print('VIEW A GENERATED SAMPLE')
    print('')

    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser
    parser = argparse.ArgumentParser(description='View a generated sample.')

    # Add arguments (and set default values where applicable)
    parser.add_argument('--hdf-file-path',
                        help='Path to the HDF sample file (generated with '
                             '`generate_sample.py`) to be used.',
                        default='../output/default.hdf')
    parser.add_argument('--sample-id',
                        help='ID of the sample to be viewed (an integer '
                             'between 0 and n_injection_samples + '
                             'n_noise_samples).',
                        default=20)
    parser.add_argument('--seconds-before',
                        help='Seconds to plot before the `event_time`.',
                        default=0.15)
    parser.add_argument('--seconds-after',
                        help='Seconds to plot after the `event_time`.',
                        default=0.05)
    parser.add_argument('--show-plot',
                        help='Show the plot using plt.show()?',
                        action='store_true',
                        default=False)
    parser.add_argument('--save-plot',
                        help='Use this option to pass a path to plt.savefig() '
                             'to save the generated plot at the given '
                             'location.',
                        default='view_sample.pdf')

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    arguments = vars(parser.parse_args())
    print('Done!')

    # Set up shortcuts for the command line arguments
    hdf_file_path = str(arguments['hdf_file_path'])
    sample_id = int(arguments['sample_id'])
    seconds_before = float(arguments['seconds_before'])
    seconds_after = float(arguments['seconds_after'])
    show_plot = bool(arguments['show_plot'])
    save_plot = str(arguments['save_plot'])

    # -------------------------------------------------------------------------
    # Read in the sample file and plot the selected sample
    # -------------------------------------------------------------------------

    print('Reading in HDF file...', end=' ')
    data = SampleFile()
    data.read_hdf(hdf_file_path)
    df = data.as_dataframe(injection_parameters=True,
                           static_arguments=True)
    print('Done!')

    print('Plotting sample...', end=' ')
    view_sample(sample_id=sample_id,
                all_samples=df,
                before=seconds_before,
                after=seconds_after,
                show_plot=show_plot,
                save_plot=save_plot)
    print('Done!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    # Print the total run time
    print('')
    print('Total runtime: {:.1f} seconds!'
          .format(time.time() - script_start_time))
    print('')
