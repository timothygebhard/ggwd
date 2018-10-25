"""
Provide tools for writing and reading sample files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import h5py

from six import iteritems
from pprint import pformat


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class SampleFile:

    def __init__(self, data=None):

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
                             injection_parameters=dict())

    # -------------------------------------------------------------------------

    @staticmethod
    def __check_data(data):

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
            self.data['injection_samples']['event_time'] = \
                np.array(hdf_file['/injection_samples/event_time'])
            self.data['injection_samples']['h1_strain'] = \
                np.array(hdf_file['/injection_samples/h1_strain'])
            self.data['injection_samples']['l1_strain'] = \
                np.array(hdf_file['/injection_samples/l1_strain'])

            # Read in group containing noise samples
            self.data['noise_samples'] = dict()
            self.data['noise_samples']['event_time'] = \
                np.array(hdf_file['/noise_samples/event_time'])
            self.data['noise_samples']['h1_strain'] = \
                np.array(hdf_file['/noise_samples/h1_strain'])
            self.data['noise_samples']['l1_strain'] = \
                np.array(hdf_file['/noise_samples/l1_strain'])

            # Read in injection parameters
            self.data['injection_parameters'] = dict()
            for key in hdf_file['/injection_parameters'].keys():
                self.data['injection_parameters'][key] = \
                    np.array(hdf_file['/injection_parameters/{}'.format(key)])

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
                group.create_dataset(name=key,
                                     shape=value.shape,
                                     dtype='f4',
                                     data=value)

            # Create group for noise_samples and save every item of the
            # dict as a new dataset
            group = hdf_file.create_group('noise_samples')
            for key, value in iteritems(self.data['noise_samples']):
                group.create_dataset(name=key,
                                     shape=value.shape,
                                     dtype='f4',
                                     data=value)

            # Create group for injection_parameters and save every item of the
            # dict as a new dataset
            group = hdf_file.create_group('injection_parameters')
            for key, value in iteritems(self.data['injection_parameters']):
                group.create_dataset(name=key,
                                     shape=value.shape,
                                     dtype='f4',
                                     data=value)

    # -------------------------------------------------------------------------

    def as_dataframe(self, injection_parameters=False, static_arguments=False,
                     command_line_arguments=False):

        injection_samples = []
        for i in range(len(self.data['injection_samples']['event_time'])):
            _ = {k: v[i] for k, v in iteritems(self.data['injection_samples'])}
            injection_samples.append(_)
        df_injection_samples = pd.DataFrame().append(injection_samples,
                                                     ignore_index=True)

        noise_samples = []
        for i in range(len(self.data['noise_samples']['event_time'])):
            _ = {k: v[i] for k, v in iteritems(self.data['noise_samples'])}
            noise_samples.append(_)
        df_noise_samples = pd.DataFrame().append(noise_samples,
                                                 ignore_index=True)

        if injection_parameters:
            injection_params = []
            for i in range(len(df_injection_samples)):
                _ = {k: v[i] for k, v in
                     iteritems(self.data['injection_parameters'])}
                injection_params.append(_)
            df_injection_params = pd.DataFrame().append(injection_params,
                                                        ignore_index=True)

            df = pd.concat([df_injection_samples, df_injection_params], axis=1)
        else:
            df = df_injection_samples

        if static_arguments:
            for key, value in iteritems(self.data['static_arguments']):
                df[key] = value
                if key in ('random_seed', 'sampling_rate', 'bandpass_lower',
                           'bandpass_upper'):
                    df_noise_samples[key] = value

        df = df.append(df_noise_samples, ignore_index=True)

        if command_line_arguments:
            for key, value in iteritems(self.data['command_line_arguments']):
                df[key] = value

        df['event_time'] = df['event_time'].astype(int)

        return df
