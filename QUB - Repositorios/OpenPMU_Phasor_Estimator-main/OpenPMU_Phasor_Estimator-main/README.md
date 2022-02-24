# OpenPMU Phasor Estimator
The OpenPMU Phasor Estimator converts a stream of sampled values from an ADC into a synchrophasor.

A stream of sampled values (SV) representing the waveform of an AC voltage or current may be acquired by an analogue to digital converter (ADC) or through simulation.  This software fits the waveform data to a cosine model, yielding a phasor representation of the waveform.  That is to say, an estimate of the best fit amplitude and phase angle of the cosine model to the sampled values.  Time and other metadata is preserved, enabling a synchrophasor to be produced. Frequency and ROCOF are also calculated.

![Screenshot](/images/OpenPMU_Phasor_Estimator.png)

Licensed under GPLv3. Collaborations towards development of this code and other aspects of the OpenPMU project are very welcome.

Requires Python 3.7 or later.  Has been tested using [WinPython](https://winpython.github.io/) 3.9.4 and on [Raspberry Pi OS](https://www.raspberrypi.org/software/operating-systems/) (March 4th 2021).

## Getting Started:

The code is run from the command line using `python StartEstimation.py`.  You may be asked to install some dependencies.

Configuration is stored in `config.json`.  The config file consists of the following settings:  

| Key | Value | Description |
|-----|-------|-------------|
|"receiveIP": | "192.168.7.1", | IP Address from which Sampled Values are received |
|"receivePort": | 48001, | UDP Port on which Sampled Values are received |
|"sendIP": | "127.0.0.1", | IP Address to which Synchrophasors are sent |
|"sendPort": | 48003, | UDP Port to which Synchrophasors are sent |
|"estimationMethod": | "leastsquareoffline", | Estimation Algorithm |
|"estimationFrequency": | 50 | Nominal Frequency of the Power System |

On first run, when using "leastsquaresoffline", an inverse matrix lookup table is calculated.  This usually consumes circa 200 to 300 MB.  Unless you change parameters, this should not need to be recreated.

Usually, the sampled values will be at a sampling rate of 256 samples per nominal cycle, yielding 12,800 Hz for 50 Hz systems, and 15,360 Hz for 60 Hz systems.  The sampling rate is fixed, or constant, and it should not vary.  The estimator should accept other sampling rates but this is not widely tested.

If you do not have ADC hardware, you can use the [OpenPMU XML SV Simulator](https://github.com/OpenPMU/OpenPMU_XML_SV_Simulator) to get started.  

* If you are running the [OpenPMU XML SV Simulator](https://github.com/OpenPMU/OpenPMU_XML_SV_Simulator)  on the same machine as the [OpenPMU Phasor Estimator](https://github.com/OpenPMU/OpenPMU_Phasor_Estimator), then the `receiveIP` should be set to '127.0.0.1' in `config.json`.  
* If you are acquiring SVs from an [OpenPMU ADC based on the BeagleBone Black](https://ieeexplore.ieee.org/document/7931698) platform over its USB Ethernet adaptor, then usually the `receiveIP` is set to '192.168.7.1' in `config.json`.  On some systems, it will be '192.168.6.1'.

## Cite this work:

When citing this work, we recommend citing the following publications:

> X. Zhao, D. M. Laverty, A. McKernan, D. J. Morrow, K. McLaughlin and S. Sezer, "[GPS-Disciplined Analog-to-Digital Converter for Phasor Measurement Applications](https://ieeexplore.ieee.org/document/7931698)," in _IEEE Transactions on Instrumentation and Measurement_, vol. 66, no. 9, pp. 2349-2357, Sept. 2017, doi: 10.1109/TIM.2017.2700158.

> D. M. Laverty, J. Hastings, D. J. Morrow, R. Khan, K. Mclaughlin and S. Sezer, "[A modular phasor measurement unit design featuring open data exchange methods](https://ieeexplore.ieee.org/document/8273986)," _2017 IEEE Power & Energy Society General Meeting_, 2017, pp. 1-5, doi: 10.1109/PESGM.2017.8273986.

Or take your pick at www.OpenPMU.org/publications
