"""
Plot the (pre-processed) strains for a real GW event.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import argparse
import h5py
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start_time = time.time()
    print('')
    print('PLOT REAL GRAVITATIONAL-WAVE EVENTS')
    print('')

    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser and add arguments
    parser = argparse.ArgumentParser(description='Plot real GW events.')
    parser.add_argument('--delta-t',
                        help='Seconds around event to plot (default: 0.25).',
                        default=0.25)
    parser.add_argument('--event',
                        help='Name of GW event to plot (default: GW150914).',
                        default='GW150914')
    parser.add_argument('--hdf-file-path',
                        help='Path to the HDF file containing the '
                             'pre-processed real events.'
                             'Default: ./output/real_events.hdf.',
                        default='./output/real_events.hdf')

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    command_line_arguments = vars(parser.parse_args())
    print('Done!')
    
    # Create shortcuts to command line arguments
    delta_t = float(command_line_arguments['delta_t'])
    event = command_line_arguments['event']
    hdf_file_path = command_line_arguments['hdf_file_path']

    # -------------------------------------------------------------------------
    # Load the data from HDF file
    # -------------------------------------------------------------------------

    with h5py.File(hdf_file_path, 'r') as hdf_file:
        h1_strain = np.array(hdf_file[event]['h1_strain'])
        l1_strain = np.array(hdf_file[event]['l1_strain'])

    # -------------------------------------------------------------------------
    # Make the plot
    # -------------------------------------------------------------------------

    # Create new subplots
    fig, axes = plt.subplots(figsize=(10, 3), nrows=2, ncols=1)

    # Plot the strain
    grid = np.linspace(-8, 8, len(h1_strain))
    axes[0].plot(grid, h1_strain)
    axes[1].plot(grid, l1_strain)

    # Add labels
    axes[1].set_xlabel('Time from event time (s)')
    axes[0].set_ylabel('Strain H1')
    axes[1].set_ylabel('Strain L1')

    # Fix limits for x- and y-axes
    axes[0].set_xlim(-delta_t, delta_t)
    axes[1].set_xlim(-delta_t, delta_t)
    axes[0].set_ylim(-150, 150)
    axes[1].set_ylim(-150, 150)

    # Add title and remove space between panels
    fig.suptitle(event, y=0.95, fontweight='bold')
    plt.subplots_adjust(hspace=0)

    # Save the plot as a PDF
    plt.savefig('{}.pdf'.format(event), bbox_inches='tight', pad_inches=0)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    runtime = time.time() - script_start_time
    print('This took {:.1f} seconds in total!'.format(runtime))
    print('')
