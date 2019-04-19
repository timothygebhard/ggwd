.. _plot-results:

Plotting the results
====================

As soon as the sample generation process is complete, you can manually inspect 
the results to get a feel for what they look like. 
To this end, we have prepared the script ``plot_sample.py`` in the root 
directory of the repository. 
You can run it as follows:

.. code-block:: bash

   python plot_sample.py --hdf-file-path=/path/to/hdf/sample/file --sample-id=N --plot-path=/where/to/save/the/result.pdf


If you haven't changed the configuration, you don't even need to specify the 
``--hdf-file-path`` option; per default it will use ``./output/default.hdf``.

The ``--sample-id`` refers to the example in the sample file that you want 
to plot. 
This should be an integer between 0 and 
``n_injection_samples + n_noise_samples - 1``, as specified in the `*.json` 
configuration file:
If you choose a value between 0 and ``n_injection_samples - 1``, you will get 
a sample containing an injection; if you choose a value between 
``n_injection_samples - 1`` and ``n_injection_samples + n_noise_samples - 1``, 
you will get a sample that does not contain an injection (i.e., that is just 
whitened background noise).

Finally, you should specify the location where you want to store the resulting 
plot using the ``--plot-path`` flag. 
Note that you can always learn more about the possible command line options 
by running ``python plot_sample.py —help``.

If everything worked as expected, you should get a result plot that looks 
something like the following:

.. image:: images/sample_with_injection.png
   :align: center

*Heureka!* It looks like a gravitational wave! :-)
In blue, you can see the whitened strain, that is, the result from adding a 
simulated waveform into real noise and subsequently *whitening* the result. 
In orange, you have the pure (i.e., unwhitened) simulated waveform before 
adding it into the background noise. 
It has been rescaled to an arbitrary scale to allow a better comparison with 
the whitened strain in blue. 
We do not expect a perfect match here!

Note also how the detector signal (in orange) is *different* for the two 
detectors (upper and lower panel). 
This is because the relative position and orientation of the two detectors 
causes the signal to be detected with a different phase and amplitude, 
depending on the position of the source in the sky. 
This dependency is described by the *antenna patterns* of the 
interferometer --- just one of the subleties of generating realistic 
gravitational-wave data that the scripts in this repository are automatically
taking care of.

Finally, if you run the ``plot_sample.py`` script for a sample that does 
*not* contain an injection, you will of course only get the whitened strain
without any not gravitational-wave signals:

.. image:: images/sample_without_injection.png
   :align: center

This is it! 
Now you should know how to use the tools in this repository to generate your 
own synthetic gravitational-wave data, combining real LIGO recordings with 
simulated waveforms to make injections. 
Keep on reading to learn more about the details of the process and how you 
can customize things to suit your needs.

