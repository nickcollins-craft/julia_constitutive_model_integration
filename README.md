# julia_constitutive_model_integration
## Repository for codes that integrate a constitutive model in Julia.
This repository contains code that (in principle) integrates the constitutive model described in the paper [A Cosserat Breakage Mechanics for brittle granular media](https://www.sciencedirect.com/science/article/pii/S0022509620302106?via%3Dihub) (the open access version is available [here](https://hal.science/hal-03120686v1)), as well as the linear stability analysis of this time-integration to determine whether a horizontal shear band forms.

The intention of the repository is that a user (that's you!) modifies the code to suit their particular needs. At this point in time, the code is structured assuming that it is integrating a model that:
1. is embedded in the Cosserat continuum, and
2. is a Breakage Mechanics model.
As such, it is unlikely that the code as-is suits your model needs, however hopefully it is not a long way from code that can work for you (e.g. adaptation to second gradient should be easy, similar principles can be followed to apply the codes to non-Breakage Mechanics (or more complicated Breakage Mechanics) models).

### Attention!
This code is very infrequently maintained, and it is not subject to any rigorous testing, or guarantees that it works on a given version of Julia. If it doesn't work for you, it shouldn't require much by way of debugging.

### Citing, help, contact etc
If you use these codes in your work, I would appreciate a citation to this repository, and the paper above. If you need help, want to discuss the code, or want to let me know about an error, I'm always contactable at nicholas[dot]collins-crat[at]inria[dot]fr
