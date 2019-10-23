"""
Provide tools for writing and reading the sample HDF files produced by
the sample generation.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import h5py

from six import iteritems
from pprint import pformat
from warnings import warn


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class SampleFile:
    """
    :class:`SampleFile` objects serve as an abstraction for the result
    files of the sample generation.

    Args:
        data (dict): A dictionary containing the following keys:
            
            .. code-block:: python
            
               {'command_line_arguments', 'static_arguments',
                'injection_samples', 'noise_samples',
                'injection_parameters', 'normalization_parameters'}
            
            The value for every key must again be a dictionary relating
            the names of sample parameters (e.g., 'h1_snr') to a numpy
            array containing the values for that parameter.
    """

    def __init__(self,
                 data=None):

        # Perform sanity checks on data
        self.__check_data(data)

        # If we have received data, store it; else initialize an empty dict
        if data is not None:
            self.data = data
        else:
            self.data = dict(command_line_arguments=dict(),
                             static_arguments=dict(),
                             injection_samples=dict(),
                             noise_samples=dict(),
                             injection_parameters=dict(),
                             normalization_parameters=dict())

    # -------------------------------------------------------------------------

    @staticmethod
    def __check_data(data):
        """
        Run some sanity checks on `data`. Raises an assertion error if
        the data fail any of these sanity checks.

        Args:
            data (dict): A dictionary as specified in the ``__init__``
                of this class, that is, a dictionary containing the
                following keys:
                
                .. code-block:: python
                
                    {'command_line_arguments', 'static_arguments',
                     'injection_samples', 'noise_samples',
                     'injection_parameters', 'normalization_parameters'}
        """

        assert isinstance(data, dict) or data is None, \
            'data must be either dict or None!'

        if data is not None:

            assert 'command_line_arguments' in data.keys(), \
                'data must provide key "command_line_arguments"!'
            assert 'static_arguments' in data.keys(), \
                'data must provide key "static_arguments"!'
            assert 'injection_samples' in data.keys(), \
                'data must provide key "injection_samples"!'
            assert 'noise_samples' in data.keys(), \
                'data must provide key "noise_samples"!'
            assert 'injection_parameters' in data.keys(), \
                'data must provide key "injection_parameters"!'
            assert 'normalization_parameters' in data.keys(), \
                'data must provide key "normalization_parameters"!'

    # -------------------------------------------------------------------------

    def __repr__(self):

        return pformat(self.data, indent=4)

    # -------------------------------------------------------------------------

    def __str__(self):

        return pformat(self.data, indent=4)

    # -------------------------------------------------------------------------

    def __getitem__(self, item):

        return self.data[item]

    # -------------------------------------------------------------------------

    def __setitem__(self, key, value):

        self.data[key] = value

    # -------------------------------------------------------------------------

    def read_hdf(self, file_path):
        """
        Read in an existing HDF sample file (e.g., to use an instance
        of :class:`SampleFile` as a convenience wrapper for accessing
        the contents of an HDF samples file).

        Args:
            file_path (str): The path to the HDF file to be read into
                the :class:`SampleFile` object.
        """

        # Clear the existing data
        self.data = {}

        with h5py.File(file_path, 'r') as hdf_file:

            # Read in dict with command_line_arguments
            self.data['command_line_arguments'] = \
                dict(hdf_file['command_line_arguments'].attrs)
            self.data['command_line_arguments'] = \
                {key: value.decode('ascii') for key, value in
                 iteritems(self.data['command_line_arguments'])}

            # Read in dict with static_arguments
            self.data['static_arguments'] = \
                dict(hdf_file['static_arguments'].attrs)
            self.data['static_arguments'] = \
                {key: value.decode('ascii') for key, value in
                 iteritems(self.data['static_arguments'])}

            # Read in group containing injection samples
            self.data['injection_samples'] = dict()
            for key in ('event_time', 'h1_strain', 'l1_strain'):
                try:
                    self.data['injection_samples'][key] = \
                        np.array(hdf_file['injection_samples'][key])
                except TypeError:
                    self.data['injection_samples'][key] = np.array(None)

            # Read in group containing noise samples
            self.data['noise_samples'] = dict()
            for key in ('event_time', 'h1_strain', 'l1_strain'):
                try:
                    self.data['noise_samples'][key] = \
                        np.array(hdf_file['noise_samples'][key])
                except TypeError:
                    self.data['noise_samples'][key] = np.array(None)

            # Read in injection parameters
            self.data['injection_parameters'] = dict()
            for key in hdf_file['/injection_parameters'].keys():
                try:
                    self.data['injection_parameters'][key] = \
                        np.array(hdf_file['injection_parameters'][key])
                except TypeError:
                    self.data['injection_parameters'][key] = np.array(None)

            # Read in dict with normalization parameters
            self.data['normalization_parameters'] = \
                dict(hdf_file['normalization_parameters'].attrs)
            self.data['normalization_parameters'] = \
                {key: float(value) for key, value in
                 iteritems(self.data['normalization_parameters'])}

    # -------------------------------------------------------------------------

    def to_hdf(self, file_path):

        with h5py.File(file_path, 'w') as hdf_file:

            # Create group for command_line_arguments and save the values of
            # the dict as attributes of the group
            group = hdf_file.create_group('command_line_arguments')
            for key, value in iteritems(self.data['command_line_arguments']):
                group.attrs[key] = str(value)

            # Create group for static_arguments and save the values of
            # the dict as attributes of the group
            group = hdf_file.create_group('static_arguments')
            for key, value in iteritems(self.data['static_arguments']):
                group.attrs[key] = str(value)

            # Create group for injection_samples and save every item of the
            # dict as a new dataset
            group = hdf_file.create_group('injection_samples')
            for key, value in iteritems(self.data['injection_samples']):
                dtype = 'float64' if key == 'event_time' else 'float32'
                if value is not None:
                    group.create_dataset(name=key,
                                         shape=value.shape,
                                         dtype=dtype,
                                         data=value)
                else:
                    group.create_dataset(name=key,
                                         shape=None,
                                         dtype=dtype)

            # Create group for noise_samples and save every item of the
            # dict as a new dataset
            group = hdf_file.create_group('noise_samples')
            for key, value in iteritems(self.data['noise_samples']):
                dtype = 'float64' if key == 'event_time' else 'float32'
                if value is not None:
                    group.create_dataset(name=key,
                                         shape=value.shape,
                                         dtype=dtype,
                                         data=value)
                else:
                    group.create_dataset(name=key,
                                         shape=None,
                                         dtype=dtype)

            # Create group for injection_parameters and save every item of the
            # dict as a new dataset
            group = hdf_file.create_group('injection_parameters')
            for key, value in iteritems(self.data['injection_parameters']):
                if value is not None:
                    group.create_dataset(name=key,
                                         shape=value.shape,
                                         dtype='float64',
                                         data=value)
                else:
                    group.create_dataset(name=key,
                                         shape=None,
                                         dtype='float64')

            # Create group for normalization_parameters and save every item
            # of the dict as a new attribute
            group = hdf_file.create_group('normalization_parameters')
            for key, value in iteritems(self.data['normalization_parameters']):
                group.attrs[key] = float(value)

    # -------------------------------------------------------------------------

    def as_dataframe(self,
                     injection_parameters=False,
                     static_arguments=False,
                     command_line_arguments=False,
                     split_injections_noise=False):
        """
        Return the contents of the :class:`SampleFile` as a ``pandas``
        data frame.

        Args:
            injection_parameters (bool): Whether or not to return
                the `injection parameters` for every sample.
            static_arguments (bool): Whether or not to return
                the `static_arguments` for every sample.
            command_line_arguments (bool): Whether or not to return
                the `command_line_arguments` for every sample.
            split_injections_noise (bool): If this is set to True, a
                separate data frame will be returned for both the
                samples with and without an injection.

        Returns:
            One (or two, if `split_injections_noise` is set to `True`)
            pandas data frame containing the sample stored in the
            :class:`SampleFile` object.
        """

        # Create a data frame for the samples containing an injection
        injection_samples = []
        if self.data['injection_samples']['event_time'].shape != ():
            for i in range(len(self.data['injection_samples']['event_time'])):
                _ = {k: v[i] for k, v in
                     iteritems(self.data['injection_samples'])}
                injection_samples.append(_)
            df_injection_samples = pd.DataFrame().append(injection_samples,
                                                         ignore_index=True,
                                                         sort=True)
        else:
            df_injection_samples = pd.DataFrame()

        # Create a data frame for the samples not containing an injection
        noise_samples = []
        if self.data['noise_samples']['event_time'].shape != ():
            for i in range(len(self.data['noise_samples']['event_time'])):
                _ = {k: v[i] for k, v in
                     iteritems(self.data['noise_samples'])}
                noise_samples.append(_)
            df_noise_samples = pd.DataFrame().append(noise_samples,
                                                     ignore_index=True,
                                                     sort=True)
        else:
            df_noise_samples = pd.DataFrame()

        # If requested, create a data frame for the injection parameters and
        # merge it with the data frame containing the injection samples
        if injection_parameters:
            injection_params = []

            # Check if we even have any injection parameters
            if self.data['injection_parameters']['mass1'].shape != ():
                for i in range(len(df_injection_samples)):
                    _ = {k: v[i] for k, v in
                         iteritems(self.data['injection_parameters'])}
                    injection_params.append(_)
                df_injection_params = pd.DataFrame().append(injection_params,
                                                            ignore_index=True,
                                                            sort=True)
            else:
                df_injection_params = pd.DataFrame()

            df = pd.concat([df_injection_samples, df_injection_params],
                           axis=1, sort=True)

        else:
            df = df_injection_samples

        # If requested, add the static_arguments to the data frame
        # containing the injections, and a smaller subset of the
        # static_arguments also to the data frame containing the noise
        # samples (only those arguments that make sense there)
        if static_arguments:
            for key, value in iteritems(self.data['static_arguments']):
                df[key] = value
                if key in ('random_seed', 'target_sampling_rate',
                           'bandpass_lower', 'bandpass_upper',
                           'seconds_before_event', 'seconds_after_event',
                           'sample_length'):
                    df_noise_samples[key] = value

        # Merge the data frames for the samples with and without injections
        df = df.append(df_noise_samples, ignore_index=True, sort=True)

        # If requested, add the command line arguments that were used in the
        # creation of the sample file to the combined data frame
        if command_line_arguments:
            for key, value in iteritems(self.data['command_line_arguments']):
                df[key] = value

        # Ensure the `event_time` variable is an integer
        try:
            df['event_time'] = df['event_time'].astype(int)
        except KeyError:
            warn('\nNo key "event_time": Data frame is probably empty!')

        # Either split into two data frames for injection and noise samples
        if split_injections_noise:
            df_injections = df[df.h1_signal.notnull()]
            df_noise = df[~df.h1_signal.notnull()]
            return df_injections, df_noise

        # Or just return a single data frame containing both types of samples
        else:
            return df
