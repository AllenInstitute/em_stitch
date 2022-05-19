[![Build Status](https://travis-ci.org/AllenInstitute/em_stitch.svg?branch=master)](https://travis-ci.org/AllenInstitute/em_stitch)
[![codecov](https://codecov.io/gh/AllenInstitute/em_stitch/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenInstitute/em_stitch)


# em_stitch

Lens correction solver and stitching utilities for EM images.

This repo has overlap with
https://github.com/AllenInstitute/asap-modules

It is meant to be independent of asap-modules and a running render server.

## Installation

This package is pip-installable with
```pip install em-stitch```

installing from this repo can be accomplished with 
```python setup.py install```


## Optional dependencies
The performance of em-stitch can benefit from the installation of `pandas` and will use it when available.  Bigfeta will use `petsc4py` to implement its solver if available, which can improve memory utilization and performance.


# support

We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support, as it is under active development. The community is welcome to submit issues, but you should not expect an active response.

# Acknowledgement of Government Sponsorship

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior / Interior Business Center (DoI/IBC) contract number D16PC00004. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DoI/IBC, or the U.S. Government.
