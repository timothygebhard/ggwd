.. _configuration-files:

Configuration files
===================

This section aims to give a more detailed overview of how the configuration 
files can be used to steer and control the sample generation process, and to 
explain which options are available.

Very fundamentally, there are two different types of configuration files
(JSON and INI files), which control different aspects of the sample generation
process.
The JSON files control the sample generation process itself, that is, they
determine for example the number of examples to generated, or the number of
concurrent processes to use multiprocessing.
Inside a JSON file, you will need to specify a ``waveform_params`` file, which
uses the INI file format.
These INI files then control the simulation of synthetic waveforms, which are 
used to make injections.
This concerns both the technical aspects of the simulation (e.g., which
waveform model to use), as well as the physical aspects, that is, the 
parameter space from which the simulation inputs are sampled.

Both types of configuration files should be placed in a folder 
``/config_files``, which resides in the same directory as the
``generate_samples.py`` script.
The repository already comes with such a folder, which contains a
``default.json`` and a ``waveform_params.ini``, which should specify
reasonable default values for everything and are a good starting point
for custom changes.

The following subsections will now outline the exact configuration file 
specifications in more detail.

.. note::
   As briefly mentioned in the introduction, the scripts in this repository
   were written primarily with the goal of using the generated samples for 
   a machine learning application.
   Most of the default values were chosen with this in mind, and for some
   options, we have added additional comments to this documentation about 
   what we think is "good to know" when using this code to create training / 
   testing data for machine learning experiments.





JSON files
----------

As explained above, the `*.json` configuration files are used to control the 
sample generation process itself. 
These configuration files accept the following options:

``random_seed``
  The seed for the random number generator, which ensures (or at least 
  improves) the reproducibility of the results.
  For example, if you want to generate a training and a test data set for your 
  application, it might make sense to have two `*.json` files that use 
  different random seeds.

  .. note::
     Note that the simulation of waveforms is not always 100% deterministic.
     This means that if you run the sample generation process twice using the
     same configuration, the resulting files will probably not be exactly
     identical; however, the differences should usually only be on the level
     of floating-point precision.

``background_data_directory``
  The path to the directory that contains the raw HDF files with the LIGO 
  recordings as downloaded from GWOSC.
  The script automatically and recursively searches the subdirectories of the 
  given path for `*.hdf` or `*.h5` files, so if your files are sorted in 
  ``/some/path/H1`` and ``/some/path/L1``, it is sufficient to give 
  ``/some/path`` as the ``background_data_directory``.
  In case you only want to work with simulated LIGO noise, you can also use 
  ``null`` (the JSON equivalent of ``None``) as a value here. 
  This is also the default setting to ensure everything runs "out of the box".

``dq_bits`` 
  The *Data Quality Bits* which you want to be set for all LIGO recordings 
  that are selected to be used as background noise. 
  The definitions of these DQ bits can be found on 
  `GWOSC <https://www.gw-openscience.org/archive/dataset/O1>`_. 
  The bits correspond to the first column of the table there. 
  More information about the meaning of the different categories 
  `is also available here <https://www.gw-openscience.org/O1>`_.

    **Example:**
    Setting ``dq_bits: [0, 1, 2, 3]`` in the JSON config file means all data 
    that  is used to inject waveforms into has to at least pass all quality 
    tests up to ``CBC CAT3``.
    To put it simply: The more ``dq_bits`` you request, the better your data 
    quality will be, but the less data will be available.

``inj_bits``
  The *Injection Bits* which you want to be set for the data that can be used 
  to inject waveforms into. 
  The meaning of these bits is given in this `table on GWOSC 
  <https://www.gw-openscience.org/archive/dataset/O1>`_. 
  More information about *Hardware Injections* can be found `on this website
  <https://www.gw-openscience.org/o1_inj>`_.

    **Example:** 
    Setting ``inj_bits: [0, 1, 2, 4]`` in the JSON config file means that the 
    only type of hardware injection that is permitted in the data used for 
    generating samples are *continuous wave injections*.

``waveform_params_file_name``
  The name of the ``*.ini`` file to be used to this sample generation process.
  This file needs to be placed in the same folder as the JSON config file.

``max_runtime``
  The maximum runtime (in seconds) for the generation of a single example.
  This is useful, because the LALSuite routines which are used internally for 
  the simulation of waveforms sometimes seem to get stuck for some parameter
  combinations.
  The exact time it takes to simulate a single example will of course depend
  on your hardware, but usually it is expected to be on the other of 10 to
  15 seconds.
  By default, the ``max_runtime`` is therefore set to 60 seconds.

``n_injection_samples``
  The number of samples containing an injection to be generated.

``n_noise_samples``
  The number of samples *not* containing an injection to be generated
  (i.e., samples that consists purely of whitened background noise). 
  If you only want to generate samples with waveform signals in them, simply 
  set this value to 0.

``n_processes``
  The number of (concurrent) processes to be used for the sample generation 
  process to speed it up by parallelizing the generation of examples.

``output_file_name``
  The name that will be given to the final output HDF file that contains all 
  generated examples (with and without injections). 
  The file ending (`*.hdf` or `*.h5`) must be included here.





INI files
---------

Also as explained before, the `*.ini` files are used primarily to control 
the process of simulating GW waveforms with PyCBC. 
Each such files consists of three sections:

* ``[variable_args]``: Here you have to declare the names of the parameters 
  whose values will vary between samples. 
  This is mostly the "physics" of the process, that is, the parameters of the 
  coalescence such as the masses of the compact objects, or the position in 
  the sky. 
  The values for these parameters will be randomly sampled from a probability 
  distribution (see below) for every waveform that is simulated.
* ``[static_args]``: These are the fixed parameters which are the same for all 
  samples. 
  This covers basically the technical side, such as the sampling rate or the 
  waveform model that is used for the simulation.
* ``[prior-*]``: These sections are used to define the probability 
  distributions for the parameters declared in ``[variable_args]``. 
  More information about the available distributions can be found 
  `in the PyCBC documentation 
  <http://pycbc.org/pycbc/latest/html/pycbc.distributions.html>`_ of the
  ``distributions`` module.

In the following, more information about the parameters of each section will 
be provided.



Variable Arguments
~~~~~~~~~~~~~~~~~~

As explained above, variable arguments are the parameters that are specific 
to the particular waveform that is being simulated. 
They determine the physical aspect of the process, such as the masses and
distance of the binary mergers, or its location in the sky. 
Variable arguments are sampled at random for every waveform from a joint 
distribution over the full parameter space.

In the following, we list the the variable arguments and their default values:

``mass1`` and ``mass2``
  The masses of the compact objects in the simulated binary coalescence. 
  The values are given in solar masses, and are per default sampled 
  independently and uniformly at random from the range 
  :math:`[10\ \textrm{M}_\odot, 80\ \textrm{M}_\odot]`.

``spin1z`` and ``spin2z``
  The spins of the two black holes (or neutron stars; depending on the chosen 
  waveform model) in the merger. 
  The values are sampled independently and uniformly from :math:`[0, 0.998]`. 
  This is because values too close to 1 can lead to numerical instabilities 
  during the simulation (this, of course, also depends on the waveform model).

``ra``
  The *right ascension* is one of the two angles that determine the position 
  of a source in the sky when using the equatorial coordinate system. 
  It is defined as *"the angular distance of the source's hour circle east of 
  the vernal equinox when measured along the celestial equator"*, or more 
  simply: 
  the right ascension is the celestial equivalent of terrestrial longitude. 
  Like the longitude, it takes on values in :math:`[0, 2\pi]`. 
  For reasons that are explained below, the ``ra`` is sampled randomly but 
  in conjunction with the declination parameter ``dec``.

``dec``
  The *declination* is the other angle determining a source's sky position.
  It is defined as the angular distance from the celestial equator 
  (alternatively: from the celestial North pole), measured along the hour 
  circle of the source. 
  Hence it takes on values in :math:`[-\pi/2, \pi/2]` (or :math:`[0, \pi]` 
  when using the alternative definition), similar to the latitude in the 
  geographic coordinate system.

  To ensure the sources of the simulated waveforms are distributed 
  isotropically in the sky, ``ra`` and ``dec`` are sampled jointly from a 
  uniform distribution over a sphere. 
  Conveniently, PyCBC already provides the ``uniform_sky`` distribution to 
  sample ``ra`` and ``dec`` jointly in this fashion.

``polarization``
  This is the polarization angle, which is one of the three Euler angles that 
  relate the *radiation frame*, which is the reference frame in which the 
  gravitational wave propagates in the *z*-direction, to  the reference frame 
  of the detector. 
  It is sampled uniformly at random from :math:`[0, 2\pi]`.

``coa_phase``
  To understand the significance of this angle, one needs to introduce a 
  third reference frame beside the detector and radiation frame, namely, 
  the reference frame of the source itself. 
  In the case of a binary coalescence, this source reference frame is chosen 
  such that its *z*-axis is perpendicular to the plane in which the two black 
  holes (or neutron stars) orbit each other. 
  Then, the ``coa_phase`` is one of the two angles that specify the location 
  in the sky of the detector as seen from this source frame. 
  Its value is chosen uniformly at random from :math:`[0, 2\pi]`.

``inclination``
  This is the other polar angle that, together with the ``coa_phase``, 
  defines  the location of the detector in the sky as observed from the 
  reference frame of the source. 
  It is sampled randomly from a *cosine distribution* to ensure that the sky 
  position is isotropically distributed on a sphere (cf. how the ``ra`` 
  and ``dec`` parameters are sampled) and takes on values in :math:`[0, \pi]`.

  .. note::
     Neither the `coa_phase` nor the `inclination` have a significant effect 
     on the simulation result unless precessing or higher-order waveforms are 
     considered; they just scale the signal-to-noise ratio (SNR).

``injection_snr``
  This is the network signal-to-noise ratio with which the simulated waveform 
  should later be observed when it is injected into the background noise. 
  It is not directly a parameter for the waveform generation (i.e., it is not 
  passed to any of the PyCBC simulation routines); however, its value is also 
  randomly generated for each waveform. 
  Per default, the injection SNR is sampled uniformly from :math:`[5, 20]`.



Static Arguments
~~~~~~~~~~~~~~~~

As mentioned before, *static arguments* are the global parameters for the 
waveform simulation, that is, they are the same for all waveforms. 
They do not contain any information about the physical system that generates 
the GW signal, but specify the technical aspects of the simulation process.

In the following, we list the the static arguments and their default values:

``approximant``
  The waveform model (also called approximant) to be used. 
  There exists a wide range of such waveform models. 
  The default that was chosen here is ``SEOBNRv4``, a state-of-the-art 
  approximant that is suitable for simulating spinning, non-precessing 
  binary black holes by using a time-domain effective-one-body (EOB) model 
  (see `Bohé et al, 2017 <https://doi.org/10.1103/PhysRevD.95.044028>`_).

  .. note:: 
     Using only a single approximant for training a ML model can involve some 
     risk of overfitting to that particular approximant. 
     If this is a concern for you, you may want to consider generating 
     multiple samples with different values for the ``approximant``, and then 
     manually mix the results to build your training / testing set.

``domain``
  For some approximants, two versions exist: one in the time- and one in the 
  frequency domain. 
  This parameter (which has to be either ``time`` or ``frequency``) 
  resolves this ambiguity.

``distance``
  The distance between the Earth and the source (in Megaparsec). 
  This parameter must necessarily be specified for the simulation. 
  It is, however, rather irrelevant for the sample generation process here, 
  because the distance only acts as a scaling factor on the waveform 
  amplitude, and the simulated waveforms are later rescaled again to a match 
  a given network SNR (the ``injection_snr``), which is a more meaningful 
  quantity than the distance. 
  For this reason, a fixed value of 100 Mpc is used for the distance.

``f_lower``
  The frequency at which to begin the simulation of the waveform. 
  The lower this frequency is chosen, the longer the resulting waveform, and 
  also the simulation time. 
  Since the LIGO detectors are not sensitive to signals below ~20 Hz, it does 
  not make a lot of sense to choose a value much lower than that. 
  Per default, the value for ``f_lower`` is 18 Hz.

``waveform_length``
  Not to be confused with the ``sample_length``, this  parameter specifies 
  the length (in seconds) up to which waveforms are simulated, or---if the 
  simulation result is shorter---are resized by padding them with zeros. 
  The default value of this parameter is set to 128 seconds.

``noise_interval_width``
  When selecting the background noise for an example, we choose an interval
  that is longer than the desired ``sample_length``, since the whitening
  procedure that is part of the sample generation process will corrupt the
  edges of an example, and these artifacts need to be cropped off.
  Per default, for samples with a length of 8 seconds, this value was chosen
  as 16 seconds, such that we can crop off 4 seconds on each side after
  making the injection and whitening the exampels.

``original_sampling_rate``
  The sampling rate of the LIGO recordings that you downloaded from 
  GWOSC (in Hertz).
  In most cases, this value should be 4096 (Hertz).

``target_sampling_rate``
  The sampling rate (or frequency) of the waveforms to be generated. 
  This has to match the sampling rate of the background noise into which the 
  simulated waveform is later injected. 
  When choosing this value, you are essentually trading off the resulting 
  sample size (in terms of memory) against the resolution in time. 
  For technical reasons, the value of ``target_sampling_rate`` has to be a 
  factor (divisor) of ``original_sampling_rate``. 
  The default value here is 2048 Hz, for the following reason:

  .. note:: 
     According to the Nyquist-Shannon sampling theorem, a sampling rate of 
     *N* Hz allows to reconstruct signals with a frequency of up to *N*/2 Hz 
     (Nyquist frequency). 
     Signals from compact binary coalescences (CBCs) are mostly expected in a 
     range of up to a few hundred Hertz, meaning a Nyquist frequency of 
     1024 Hz should be sufficient for resolving them. 
     Therefore, we have chosen a default value of 2048 Hz for the
     `target_sampling_rate`.

``whitening_segment_duration``
  Parameter that is passed to PyCBC when whitening the examples.
  Check the `PyCBC documentation <https://pycbc.org/pycbc/latest/html/
  pycbc.types.html#pycbc.types.timeseries.TimeSeries.whiten>`_  of the 
  ``whiten()`` method provided by the ``TimeSeries`` class.
  The default values for this parameter is 4.

``whitening_max_filter_duration``
  Parameter that is passed to PyCBC when whitening the examples.
  Check the `PyCBC documentation <https://pycbc.org/pycbc/latest/html/
  pycbc.types.html#pycbc.types.timeseries.TimeSeries.whiten>`_ of the 
  ``whiten()`` method provided by the ``TimeSeries`` class.
  The default values for this parameter is 4.

``bandpass_lower``
  The cutoff-frequency for the high-pass that is applied after whitening. 
  A value of 20 Hz was chosen as the default.
  This is slightly higher than ``f_lower``, which helps to suppress the 
  non-physical turn-on effects of the simulation.

``bandpass_upper``
  The cutoff-frequency for the low-pass that is applied after whitening. 
  Per default, no low-pass is used for the sample generation. 
  This is realized by choosing ``bandpass_upper`` equal to 
  ``target_sampling_rate``.

``seconds_before_event``
  The number of seconds between the start of the sample and the event time 
  (i.e., peak of the waveform signal) in the ``H1`` channel. 
  A value of 5.5 was chosen as the default. 
  This defines the location of the coalescence within the sample.

  .. note:: 
     If you are training a machine learning model that is sensitive to the 
     *absolute* position of the injection in the sample, and you want to 
     avoid overfitting to this, you have a few different options:

     1. You can generate a sample file using the scripts in the repository
        "as is" and then subsquently randomly crop off parts of each example
        on the left and right.
     2. You can generate multiple sample files with different values for 
        ``seconds_before_event``, and combine the results manually into your 
        training / test data set.
     3. You can turn ``seconds_before_event`` into a *variable argument*, and 
        specify a probability distribution for it. 
        This will require a few more changes in the code (in particular the 
        ``waveforms.py`` file), but should be mostly straight-forward.

``seconds_after_event``
  The number of seconds between the event time in the ``H1`` channel and 
  the end of the sample. 
  A value of 2.5 seconds was chosen for the default. 
  Together with ``seconds_before_event``, this parameter implicitly defines 
  the ``sample_length`` (which is simply their sum --- in the default case, 
  this means 8 seconds).

``tukey_alpha``
  To reduce any amplitude discontinuities when injecting simulated waveforms 
  into the background noise, there is the option to "fade-on" the amplitude 
  of the waveform by multiplying it with a "one-sided" `Tukey window 
  <https://en.wikipedia.org/wiki/Window_function#Tukey_window>`_. 
  The parameter ``tukey_alpha`` is passed to the ``scipy.signal.tukey`` 
  function (as `alpha`; `see also the scipy documentation for more information 
  <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/
  scipy.signal.tukey.html>`_.) to control the shape of the Tukey window. 
  It takes on values between 0 and 1, with a default value of 0.25. 
  To disable this fade-on, simple set ``tukey_alpha=0``.

    **Example:** 
    The effect of "fading on" a waveform using this procedure is also 
    illustrated by the following graphic (compare the start of the two 
    waveforms):
    
    .. image:: images/tukey_alpha.png
       :align: center
