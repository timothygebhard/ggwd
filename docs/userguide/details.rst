Getting into details
====================

In case you want to get even deeper into the inner workings of this 
repository, this section is here to help you get started. 
Maybe the most useful starting point is an overview of the sample generation 
itself, which is provided by the flowchart in the next subsection.

In the subsequent parts, we provide more detailed information about some 
aspects of the sample generation which may be useful if you want to start
making your own modifications.





Flowchart of the sample generation process
------------------------------------------

In principle, the generation of a single example works as illustrated by this
flowchart:

.. image:: images/sample_generation.png
   :align: center
   :width: 480





Finding valid noise times
-------------------------

In order to find a piece of background recording into which we can inject a 
simulated waveform, we first need to find a *valid* noise time. 
But what does that exactly mean? 
Well, a given time `t` is called valid, if a `delta_t` interval around it 
satisfies the following constraints:

* Both detectors (H1 and L1) have data in that interval.
* The entire interval has at least the data quality specified by the 
  ``dq_bits`` parameter.
* The entire interval only contains hardware injections of the types allowed 
  by the ``inj_bits``.
* The interval does not contain any *real* GW events.
  The coincidence check for this makes use of the :class:`pycbc.catalog` 
  package, which `is documented here 
  <https://pycbc.org/pycbc/latest/html/pycbc.catalog.html>`_.
* The interval does not span over multiple raw HDF files, that is the noise 
  time is at least ``delta_t`` seconds aways from the edge of the HDF file 
  that contains it.
  (This last restriction is purely for convenience reasons and may be dropped 
  if you adjust the :func:`utils.hdffiles.get_strain_from_hdf_file()` method
  accordingly.)

The entire functionality for finding and sampling valid noise times is 
contained in the :class:`utils.hdffiles.NoiseTimeline` class.
The process works as follows: 

1. When an instance of that class is instantiated, the class-internal method
   :func:`_get_hdf_files()` first collects a list of all the raw LIGO 
   recordings in the given ``background_data_directory``. 
2. Then, the method :func:`_build_timeline()` loops over these files, reads 
   in the ``dq_bits`` and ``inj_bits`` arrays of the HDF files, and combines 
   them all into a single big timeline.
   Depending in the number of background files, this may take a few minutes.
3. This ``timeline`` is then used by the method
   :func:`utils.hdffiles.NoiseTimeline.is_valid()` to check the above 
   conditions for a given ``gps_time`` and a ``delta_t``. 
4. The :func:`utils.hdffiles.NoiseTimeline.sample()` method then basically 
   only generates random times between the start and the end of the 
   ``timeline`` until it finds one that passes the tests of the
   :func:`is_valid()` method.





Structure of the output HDF files
---------------------------------

You can easily get an overview of the structure of the generated HDF files 
by running the following command (assuming you have the `HDF5 command line
tools <https://portal.hdfgroup.org/display/HDF5/HDF5+Tools+by+Category>`_
installed):

.. code-block:: bash

   h5ls -r <output_file>.hdf


The output for the default configuration should look like this:

.. code-block:: bash

   /                                        Group
   /command_line_arguments                  Group
   /injection_parameters                    Group
       /injection_parameters/coa_phase      Dataset {32}
       /injection_parameters/dec            Dataset {32}
       /injection_parameters/h1_signal      Dataset {32, 16384}
       /injection_parameters/h1_snr         Dataset {32}
       /injection_parameters/inclination    Dataset {32}
       /injection_parameters/injection_snr  Dataset {32}
       /injection_parameters/l1_signal      Dataset {32, 16384}
       /injection_parameters/l1_snr         Dataset {32}
       /injection_parameters/mass1          Dataset {32}
       /injection_parameters/mass2          Dataset {32}
       /injection_parameters/polarization   Dataset {32}
       /injection_parameters/ra             Dataset {32}
       /injection_parameters/scale_factor   Dataset {32}
       /injection_parameters/spin1z         Dataset {32}
       /injection_parameters/spin2z         Dataset {32}
   /injection_samples                       Group
       /injection_samples/event_time        Dataset {32}
       /injection_samples/h1_strain         Dataset {32, 16384}
       /injection_samples/l1_strain         Dataset {32, 16384}
   /noise_samples                           Group
       /noise_samples/event_time            Dataset {16}
       /noise_samples/h1_strain             Dataset {16, 16384}
       /noise_samples/l1_strain             Dataset {16, 16384}
   /normalization_parameters                Group
   /static_arguments                        Group

The generated output files are standard HDF files and can be read and 
handled as such. 
However,  we also provide the :class:`utils.samplefiles.SampleFile` class as 
a convenience wrapper, which for example allows to easily read in a generated 
sample file into a ``pandas`` data frame.
