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

R1-1 (December XXX, 2016)
====================
* Added new parameters ADSerialNumber, ADFirmwareVersion, ADSDKVersion. Minor change to
  driver and medm screen.


R1-0 (November 21, 2016)
====================
* Initial release. This release requires changes to ADCore that were made after R2-5.  
  They will be present in R2-6, but until R2-6 is released the master branch of ADCore should be used.

