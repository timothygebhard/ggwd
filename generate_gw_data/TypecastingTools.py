"""
Provide tools that are needed for typecasting.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import copy


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def typecast_static_args(static_args):
    """
    Take the static_args dict as it is read in from the PyCBC config file
    (i.e., all values are strings) and typecast the variables as needed. Also
    add useful variables that can be trivially calculated from others.

    Args:
        static_args: dict
            The raw static_args dict as it is read from the waveform
            parameters config file.
    Returns:
        The static_args dict with proper types for all variables, and some
        useful variables added.
    """

    args = copy.deepcopy(static_args)

    # Cast variables to integer that need to be integers
    args['bandpass_lower'] = int(args['bandpass_lower'])
    args['bandpass_upper'] = int(args['bandpass_upper'])
    args['fd_length'] = int(args['fd_length'])
    args['td_length'] = int(args['td_length'])
    args['waveform_length'] = int(args['waveform_length'])
    args['original_sampling_rate'] = int(args['original_sampling_rate'])
    args['target_sampling_rate'] = int(args['target_sampling_rate'])
    args['whitening_segment_duration'] = \
        float(args['whitening_segment_duration'])
    args['whitening_max_filter_duration'] = \
        int(args['whitening_max_filter_duration'])

    # Cast variables to float that need to be floats
    args['delta_f'] = float(args['delta_f'])
    args['delta_t'] = float(args['delta_t'])
    args['distance'] = float(args['distance'])
    args['f_lower'] = float(args['f_lower'])
    args['seconds_before_event'] = float(args['seconds_before_event'])
    args['seconds_after_event'] = float(args['seconds_after_event'])

    # Add useful variables derived from others
    args['sample_length'] = (args['seconds_before_event'] +
                             args['seconds_after_event'])

    return args
