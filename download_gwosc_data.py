"""
This script simplifies downloading raw LIGO data from GWOSC.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import os
import argparse
import time
import requests

from utils.progressbar import ProgressBar


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser
    parser = argparse.ArgumentParser(description='Download Raw GWOSC Data.')

    # Add arguments (and set default values for some variables)
    parser.add_argument('--observation-run',
                        help='Observation run for which to download data '
                             '(default: O1).',
                        default='O1')
    parser.add_argument('--gps-start-time',
                        help='Start time (GPS) of data to download '
                             '(default: 1126051217).',
                        default='1126051217')
    parser.add_argument('--gps-end-time',
                        help='End time (GPS) of data to download '
                             '(default: 1137254417).',
                        default='1137254417')
    parser.add_argument('--detector',
                        help='Detector to download data for '
                             '(default: both).',
                        choices=['H1', 'L1', 'both'],
                        default='both')
    parser.add_argument('--destination',
                        help='Target directory for the download '
                             '(default: .).',
                        default='.')
    parser.add_argument('--threshold',
                        help='Minimum percentage of the file that has valid '
                             'data (also known as duty_cycle) '
                             '(default: 0).',
                        default=0)
    parser.add_argument('--dry',
                        help='Do not download; just list files '
                             '(default: False).',
                        action='store_true',
                        default=False)

    # Parse the arguments that were passed when calling this script
    arguments = vars(parser.parse_args())

    # -------------------------------------------------------------------------
    # Define shortcuts for the command line arguments
    # -------------------------------------------------------------------------

    observation_run = arguments['observation_run']
    gps_start_time = arguments['gps_start_time']
    gps_end_time = arguments['gps_end_time']
    detectors = arguments['detector']
    destination = arguments['destination']
    threshold = arguments['threshold']
    dry = arguments['dry']

    # -------------------------------------------------------------------------
    # Start the stop watch to measure how long the download took
    # -------------------------------------------------------------------------

    script_start_time = time.time()
    print('')
    print('DOWNLOAD RAW HDF FILES FROM GRAVITATIONAL WAVE OPEN SCIENCE CENTER '
          '(GWOSC)')
    print('')

    if dry:
        print('Running in dry mode (not downloading anything)!\n')

    # -------------------------------------------------------------------------
    # Loop over specified detectors and download the HDF files
    # -------------------------------------------------------------------------

    if detectors == 'both':
        detectors = ['H1', 'L1']
    else:
        detectors = [detectors]

    for detector in detectors:

        # Define the directory into which we would download the data for
        # the current detector
        directory = os.path.join(destination, detector)
        if not dry and not os.path.exists(directory):
            os.mkdir(directory)

        # Construct the URL to the JSON file containing the links to all files
        urlformat = 'https://gw-openscience.org/archive/links/' \
            '{0}/{1}/{2}/{3}/json/'
        url = urlformat.format(observation_run, detector, gps_start_time,
                               gps_end_time)

        # If we are actually in download-mode, we also want to download the
        # JSON file which specifies the list of all HDF files so we have it
        # available later
        if not dry:

            print('Downloading JSON file for {} from URL...'
                  .format(detector), end=' ')

            # Construct the path to the JSON file and actually download it
            json_file_path = os.path.join(directory, 'table_of_files.json')
            request = requests.get(url)
            with open(json_file_path, "wb") as json_file:
                json_file.write(request.content)

            print('Done!')

        # Even if we are not downloading anything, we still need to read the
        # contents of the JSON file
        print('Reading in JSON file for {} ...'.format(detector), end=' ')
        request = requests.get(url)
        dataset = request.json()['strain']
        print('Done!')

        # Keep only the URLs to the HDF files (drop .gwf files)
        dataset = filter(lambda _: _['url'].endswith('hdf5'), dataset)

        # Keep only entries where valid data percentage is above the threshold
        dataset = filter(lambda _: _['duty_cycle'] >= threshold, dataset)

        # If we are performing a dry run we don't a progress bar
        if not dry:
            dataset = ProgressBar(list(dataset))
            print('Downloading HDF files for {}...\n'.format(detector))
        else:
            dataset = list(dataset)
            print('Available HDF files for {}:'.format(detector))

        # Loop over all HDF files that were specified in the JSON file
        for entry in dataset:

            # Construct the destination path and file path
            file_name = '{}_{}-{}.hdf'.format(detector, entry['GPSstart'],
                                              str(int(entry['GPSstart']) +
                                                  int(entry['duration'])))
            file_path = os.path.join(directory, file_name)

            # Either print the file (dry mode), or actually download it
            if dry:
                print('--', entry['url'], '->', file_path)

            else:

                # Print the progress bar and which file is being downloaded
                # Remember, dataset has been decorated using ProgressBar
                dataset.write(extras=['Downloading: {}'.format(entry['url'])])

                # Perform the actual download
                request = requests.get(entry['url'])
                with open(file_path, "wb") as f:
                    f.write(request.content)

        # Clear up the progress bar and show that downloads are finished!
        if isinstance(dataset, ProgressBar):
            dataset.write(clear_line=True)
        if not dry:
            print('\nDownload for {} is complete!\n'.format(detector))

    # Finally print the overall runtime of the script
    print('This took {:.2f} seconds!'.format(time.time() - script_start_time))
    print('')
