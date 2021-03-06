NDDriverStdArrays Releases
==========================

The latest untagged master branch can be obtained at
https://github.com/areaDetector/NDDriverStdArrays.

Tagged source code releases can be obtained at 
https://github.com/areaDetector/NDDriverStdArrays/releases.

The versions of EPICS base, asyn, and other synApps modules used for each release can be obtained from 
the EXAMPLE_RELEASE_PATHS.local, EXAMPLE_RELEASE_LIBS.local, and EXAMPLE_RELEASE_PRODS.local
files respectively, in the configure/ directory of the appropriate release of the 
[top-level areaDetector](https://github.com/areaDetector/areaDetector) repository.


Release Notes
=============

R1-4 (January XXX, 2020)
====================
* Fixed initialization problem in the driver.

R1-3 (December 19, 2018)
====================
* Fixed layout of medm screen.
* Added op/edl/autoconvert, op/opi/autoconvert, and op/ui/autoconvert files.
* Fixed Db/NDDriverStdArrays_settings.req for misnamed PV and missing PVs.

R1-2 (July 5, 2017)
====================
* Changed layout of medm screen for ADCore R3-0.


R1-1 (February 22, 2017)
====================
* Added new parameters ADSerialNumber, ADFirmwareVersion, ADSDKVersion. Minor change to
  driver and medm screen.


R1-0 (November 21, 2016)
====================
* Initial release. This release requires changes to ADCore that were made after R2-5.  
  They will be present in R2-6, but until R2-6 is released the master branch of ADCore should be used.

