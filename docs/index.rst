Welcome to ggwd's documentation!
================================

The purpose of this repository is to provide a starting point for generating
realistic samples of synthetic gravitational-wave data which can then be used, 
for example, for machine learning experiments. 
By "samples", we mean time series data which either does or does not contain 
a gravitational-wave waveform corresponding to a compact binary coalescence
(CBC), like for example a binary black hole merger event.

The main idea behind the data generating process is listed the following.
Of course in practice, each of these steps involves numerous subleties and 
complications, which are not always too obvious:

1. Randomly select a piece of real LIGO recording to serve as background 
   noise (or generate a piece of synthetic noise with the correct power 
   spectral density of the instrument),
2. Randomly select the parameters of a compact binary coalescence (CBC) from 
   a given parameter space and simulate the corresponding waveform,
3. Add ("inject") this simulated waveform into the background noise.

The scripts in this repository are mostly just convenience wrappers around 
the great and mighty `PyCBC software package <https://pycbc.org>`_, which 
itself relies in parts on the `LIGO Algorithm Library (LALSuite) 
<https://wiki.ligo.org/Computing/DASWG/LALSuite>`_.
However, due to the reason mentioned above, we believe that the scripts and 
methods in  repository will still be of use to you (or at least serve as an 
educational resource), especially when you are not yet familiar with the
process of generationg synthetic GW data.





.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   userguide/setup
   userguide/get-data
   userguide/sample-creation
   userguide/plot-results
   userguide/real-events
   userguide/configfiles
   userguide/details

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   modules/configfiles
   modules/hdffiles
   modules/progressbar
   modules/samplefiles
   modules/samplegeneration
   modules/staticargs
   modules/waveforms
