"""
Provide tools that are needed for amending and typecasting the static
arguments from an `*.ini` configuration file, which controls the
waveform simulation process.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import copy


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def amend_static_args(static_args):
    """
    Amend the static_args from the `*.ini` configuration file by adding
    the parameters that can be computed directly from others (more
    intuitive ones). Note that the static_args should have been
    properly typecast first; see :func:`typecast_static_args()`.

    Args:
        static_args (dict): The static_args dict after it has been
            typecast by :func:`typecast_static_args()`.

    Returns:
        The amended `static_args`, where implicitly defined variables
        have been added.
    """

    # Create a copy of the original static_args
    args = copy.deepcopy(static_args)
    
    # If necessary, compute the sample length
    if 'sample_length' not in args.keys():
        args['sample_length'] = \
            args['seconds_before_event'] + args['seconds_after_event']

    # If necessary, add delta_t = 1 / target_sampling_rate
    if 'delta_t' not in args.keys():
        args['delta_t'] = 1.0 / args['target_sampling_rate']

    # If necessary, add delta_f = 1 / waveform_length
    if 'delta_f' not in args.keys():
        args['delta_f'] = 1.0 / args['waveform_length']

    # If necessary, add td_length = waveform_length * target_sampling_rate
    if 'td_length' not in args.keys():
        args['td_length'] = \
            int(args['waveform_length'] * args['target_sampling_rate'])

    # If necessary, add fd_length = td_length / 2 + 1
    if 'fd_length' not in args.keys():
        args['fd_length'] = int(args['td_length'] / 2.0 + 1)

    return args


def typecast_static_args(static_args):
    """
    Take the `static_args` dictionary as it is read in from the PyCBC
    configuration file (i.e., all values are strings) and cast the
    values to the correct types (`float` or `int`).

    Args:
        static_args (dict): The raw `static_args` dictionary as it is
            read from the `*.ini` configuration file.
            
    Returns:
        The `static_args` dictionary with proper types for all values.
    """

    args = copy.deepcopy(static_args)

    # Cast variables to integer that need to be integers
    args['bandpass_lower'] = int(args['bandpass_lower'])
    args['bandpass_upper'] = int(args['bandpass_upper'])
    args['waveform_length'] = int(args['waveform_length'])
    args['noise_interval_width'] = int(args['noise_interval_width'])
    args['original_sampling_rate'] = int(args['original_sampling_rate'])
    args['target_sampling_rate'] = int(args['target_sampling_rate'])
    args['whitening_segment_duration'] = \
        float(args['whitening_segment_duration'])
    args['whitening_max_filter_duration'] = \
        int(args['whitening_max_filter_duration'])

    # Cast variables to float that need to be floats
    args['distance'] = float(args['distance'])
    args['f_lower'] = float(args['f_lower'])
    args['seconds_before_event'] = float(args['seconds_before_event'])
    args['seconds_after_event'] = float(args['seconds_after_event'])

    return args
