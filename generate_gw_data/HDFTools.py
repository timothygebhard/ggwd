"""
Provide classes and functions for reading and writing HDF files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function
import numpy as np
import h5py
import os
import sys

from Constants import GW150914_TIME, GW151226_TIME, LVT151012_TIME


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_file_paths(directory, extensions=None):
    """
    Take a directory and return the paths to all files in this directory and
    its subdirectories. Optionally filter out only files specific extensions.

    Args:
        directory: str
            Path to a directory.
        extensions: list of str
            List of allowed file extensions, e.g. ['hdf', 'h5']

    Returns:
        List of file paths.
    """

    file_paths = []

    # Walk over the directory and find all files
    for path, dirs, files in os.walk(directory):
        for f in files:
            file_paths.append(os.path.join(path, f))

    # If a list of extensions is provided, only keep the corresponding files
    if extensions is not None:
        file_paths = [_ for _ in file_paths if any([_.endswith(ext) for
                                                    ext in extensions])]

    return file_paths


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class NoiseTimeline:

    def __init__(self,
                 data_directory,
                 random_seed=42,
                 verbose=False):

        # Store the directory of the raw HDF files
        self.data_directory = data_directory

        # Print debug messages or not?
        self.verbose = verbose

        # Store and set the random seed
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Get the list of all HDF files in the specified directory
        self.vprint('Getting HDF file paths...', end=' ')
        self.hdf_file_paths = get_file_paths(self.data_directory,
                                             extensions=['hdf', 'h5'])
        self.vprint('Done!')

        # Read in the meta information and masks from HDF files
        self.vprint('Reading information from HDF files', end=' ')
        self.hdf_files = self._get_hdf_files()
        self.vprint('Done!')

        # Build the timeline for these HDF files
        self.vprint('Building timeline object...', end=' ')
        self.timeline = self._build_timeline()
        self.vprint('Done!')

    # -------------------------------------------------------------------------

    def vprint(self, string, *args, **kwargs):
        """
        verbose print: Wrapper around print() to only call it if self.verbose
        is set to true.

        Args:
            string: String to be printed if self.verbose is True.
            *args: arguments passed to print()
            **kwargs: keyword arguments passed to print()
        """

        if self.verbose:
            print(string, *args, **kwargs)
            sys.stdout.flush()

    # -------------------------------------------------------------------------

    def _get_hdf_files(self):

        # Keep track of all the files whose information we need to store
        hdf_files = []

        # Open every HDF file once to read in the meta information as well
        # as the injection and data quality (DQ) masks
        n_files = len(self.hdf_file_paths)
        for i, hdf_file_path in enumerate(self.hdf_file_paths):
            with h5py.File(hdf_file_path, 'r') as f:

                self.vprint('({:>4}/{:>4})...'.format(i, n_files), end=' ')

                # Select necessary information from the HDF file
                start_time = f['meta']['GPSstart'][()]
                detector = f['meta']['Detector'][()].decode('utf-8')
                duration = f['meta']['Duration'][()]
                inj_mask = np.array(f['quality']['injections']['Injmask'],
                                    dtype=np.int32)
                dq_mask = np.array(f['quality']['simple']['DQmask'],
                                   dtype=np.int32)

                # Perform some basic sanity checks
                assert detector in ['H1', 'L1'], \
                    'Invalid detector {}!'.format(detector)
                assert duration == len(inj_mask) == len(dq_mask), \
                    'Length of InjMask or DQMask does not match the duration!'

                # Collect this information in a dict
                hdf_files.append(dict(file_path=hdf_file_path,
                                      start_time=start_time,
                                      detector=detector,
                                      duration=duration,
                                      inj_mask=inj_mask,
                                      dq_mask=dq_mask))

                self.vprint('\033[15D\033[K', end='')

        # Sort the read in HDF files by start time and return them
        self.vprint('({:>4}/{:>4})...'.format(n_files, n_files), end=' ')
        return sorted(hdf_files, key=lambda _: _['start_time'])

    # -------------------------------------------------------------------------

    def _build_timeline(self):

        # Get the size of the arrays that we need to initialize
        n_entries = self.gps_end_time - self.gps_start_time

        # Initialize the empty timeline
        timeline = dict(h1_inj_mask=np.zeros(n_entries, dtype=np.int32),
                        l1_inj_mask=np.zeros(n_entries, dtype=np.int32),
                        h1_dq_mask=np.zeros(n_entries, dtype=np.int32),
                        l1_dq_mask=np.zeros(n_entries, dtype=np.int32))

        # Add information from HDF files to timeline
        for hdf_file in self.hdf_files:

            # Define some shortcuts
            detector = hdf_file['detector']
            dq_mask = hdf_file['dq_mask']
            inj_mask = hdf_file['inj_mask']

            # Map start/end from GPS time to array indices
            idx_start = hdf_file['start_time'] - self.gps_start_time
            idx_end = idx_start + hdf_file['duration']

            # Add the mask information to the correct detector
            if detector == 'H1':
                timeline['h1_inj_mask'][idx_start:idx_end] = inj_mask
                timeline['h1_dq_mask'][idx_start:idx_end] = dq_mask
            else:
                timeline['l1_inj_mask'][idx_start:idx_end] = inj_mask
                timeline['l1_dq_mask'][idx_start:idx_end] = dq_mask

        # Return the completed timeline
        return timeline

    # -------------------------------------------------------------------------

    def is_valid(self,
                 gps_time,
                 delta_t=256,
                 dq_bits=(0, 1, 2, 3),
                 inj_bits=(0, 1, 2, 4)):
        """
        For a given GPS time, test if is a valid time to sample noise from
        by checking if all data in the interval [gps_time - delta_t,
        gps_time + delta_t] have the specified dq_bits and inj_bits set.

        Args:
            gps_time: int
                The GPS time from which we would like to draw a noise sample.
            delta_t: int
                The number of seconds around `gps_time` which we also want
                to be valid (because the sample will be an interval)
            dq_bits: tuple of int
                The Data Quality Bits which one would like to require, see
                here: https://www.gw-openscience.org/archive/dataset/O1/
                Example: dq_bits=(0, 1, 2, 3) means that the DQ needs to pass
                all tests up to `CAT3`.
            inj_bits: tuple of int
                The Injection Bits which one would like to require, see
                here: https://www.gw-openscience.org/archive/dataset/O1/
                Example: inj_bits=(0, 1, 2, 4) means that only continuous wave
                (CW) injections are permitted; all recordings containing any
                of other type of injection will be invalid for sampling.

        Returns:

        """

        # ---------------------------------------------------------------------
        # Perform some basic sanity checks
        # ---------------------------------------------------------------------

        assert isinstance(gps_time, int), \
            'Received GPS time that is not an integer!'
        assert delta_t >= 0, \
            'Received an invalid value for delta_t!'
        assert set(dq_bits).issubset(set(range(7))), \
            'Invalid Data Quality bit specification passed to is_valid()!'
        assert set(inj_bits).issubset(set(range(5))), \
            'Invalid Injection bit specification passed to is_valid()!'

        # ---------------------------------------------------------------------
        # Check if given time is too close to a real event
        # ---------------------------------------------------------------------

        if any(abs(gps_time - _) <= delta_t for _ in (GW150914_TIME,
                                                      GW151226_TIME,
                                                      LVT151012_TIME)):
            return False

        # ---------------------------------------------------------------------
        # Select the environment around the specified time
        # ---------------------------------------------------------------------

        # Map time to indices
        idx_start = self.gps2idx(gps_time) - delta_t
        idx_end = self.gps2idx(gps_time) + delta_t

        # Select the mask intervals
        environment = \
            dict(h1_inj_mask=self.timeline['h1_inj_mask'][idx_start:idx_end],
                 l1_inj_mask=self.timeline['l1_inj_mask'][idx_start:idx_end],
                 h1_dq_mask=self.timeline['h1_dq_mask'][idx_start:idx_end],
                 l1_dq_mask=self.timeline['l1_dq_mask'][idx_start:idx_end])

        # ---------------------------------------------------------------------
        # Data Quality Check
        # ---------------------------------------------------------------------

        # Compute the minimum data quality
        min_dq = sum([2**i for i in dq_bits])

        # Perform the DQ check for H1
        environment['h1_dq_mask'] = environment['h1_dq_mask'] > min_dq
        if not np.all(environment['h1_dq_mask']):
            return False

        # Perform the DQ check for L1
        environment['l1_dq_mask'] = environment['l1_dq_mask'] > min_dq
        if not np.all(environment['l1_dq_mask']):
            return False

        # ---------------------------------------------------------------------
        # Injection Check
        # ---------------------------------------------------------------------

        # Define an array of ones that matches the length of the environment.
        # This  is needed because for a given number N, we  can check if the
        # K-th bit is set by evaluating the expression: N & (1 << K)
        ones = np.ones(2 * delta_t, dtype=np.int32)

        # For each requested injection bit, check if it is set for the whole
        # environment (for both H1 and L1)
        for i in inj_bits:

            # Perform the injection check for H1
            if not np.all(np.bitwise_and(environment['h1_inj_mask'],
                                         np.left_shift(ones, i))):
                return False

            # Perform the injection check for L1
            if not np.all(np.bitwise_and(environment['l1_inj_mask'],
                                         np.left_shift(ones, i))):
                return False

        # If we have not returned False yet, the time must be valid!
        return True

    # -------------------------------------------------------------------------

    def sample(self,
               delta_t=256,
               dq_bits=(0, 1, 2, 3),
               inj_bits=(0, 1, 2, 4)):

        """
        Randomly sample a time from [gps_start_time, gps_end_time] that
        passes the is_valid() test.

        Args:
            delta_t: int
                For an explanation, see is_valid()
            dq_bits: tuple
                For an explanation, see is_valid()
            inj_bits: tuple
                For an explanation, see is_valid()

        Returns:
            A valid GPS time.
        """

        # Keep sampling random times until we find a valid one...
        while True:

            # Randomly choose a GPS time between the start and end
            gps_time = np.random.randint(self.gps_start_time + delta_t,
                                         self.gps_end_time - delta_t)

            # If it is a valid time, return it
            if self.is_valid(gps_time=gps_time, delta_t=delta_t,
                             dq_bits=dq_bits, inj_bits=inj_bits):
                return gps_time

    # -------------------------------------------------------------------------

    def get_file_paths_for_time(self,
                                gps_time):
        """
        For a given (valid) GPS time, find the two HDF files (for H1 and L1)
        which contain the corresponding recording.

        Args:
            gps_time: int
                A valid GPS time stamp.

        Returns: dict or None
            A dictionary with keys 'H1', 'L1' containing the paths to the HDF
            files, or None, if no such files could be found
        """

        # Keep track of the results, i.e., the paths to the HDF files
        result = dict()

        # Loop over all HDF files to find the ones containing the given time
        for hdf_file in self.hdf_files:

            # Get the start and end time for the current HDF file
            start_time = hdf_file['start_time']
            end_time = start_time + hdf_file['duration']

            # Check if the given GPS time falls into the interval of the
            # current HDF file, and if so, store the file path for it
            if start_time < gps_time < end_time:
                result[hdf_file['detector']] = hdf_file['file_path']

            # If both files were found, we are done!
            if 'H1' in result.keys() and 'L1' in result.keys():
                return result

        # If we didn't both files, return None
        return None

    # -------------------------------------------------------------------------

    def idx2gps(self, idx):
        """
        Map an index to a GPS time by correcting for the start time of the
        observation run, as determined from the HDF files.

        Args:
            idx: int
                An index of a timeseries array (covering an observation run)

        Returns:
            The corresponding GPS time.
        """

        return idx + self.gps_start_time

    # -------------------------------------------------------------------------

    def gps2idx(self, gps):
        """
        Map an GPS time to an index by correcting for the start time of the
        observation run, as determined from the HDF files.

        Args:
            gps: int
                A GPS time belonging to a point in time between the start and
                end of an obversation run

        Returns:
            The corresponding time series index.
        """

        return gps - self.gps_start_time

    # -------------------------------------------------------------------------

    @property
    def gps_start_time(self):
        """
        Returns: The GPS start time of the observation run.
        """

        return self.hdf_files[0]['start_time']

    # -------------------------------------------------------------------------

    @property
    def gps_end_time(self):
        """
        Returns: The GPS end time of the observation run.
        """

        return self.hdf_files[-1]['start_time'] + \
            self.hdf_files[-1]['duration']
