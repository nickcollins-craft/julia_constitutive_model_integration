# julia_constitutive_model_integration
## Repository for codes that integrate a constitutive model in Julia.
This repository contains code that (in principle) integrates the constitutive model described in the paper [A Cosserat Breakage Mechanics for brittle granular media](https://www.sciencedirect.com/science/article/pii/S0022509620302106?via%3Dihub) (the open access version is available [here](https://hal.science/hal-03120686v1)), as well as the linear stability analysis of this time-integration to determine whether a horizontal shear band forms.

The intention of the repository is that a user (that's you!) modifies the code to suit their particular needs. At this point in time, the code is structured assuming that it is integrating a model that:
  1. is embedded in the (small strain) Cosserat continuum, and
  2. is a Breakage Mechanics model.  
As such, it is unlikely that the code as-is suits your model needs, however hopefully it is not a long way from code that can work for you (e.g. adaptation to second gradient should be easy, similar principles can be followed to apply the codes to non-Breakage Mechanics (or more complicated Breakage Mechanics) models).

## What is in the repository
The repository contains code that specifies material parameters following the above paper (specifically, the defaults are set to those contained in Table 2 of the paper) and two separate integrators of the same underlying model, one in classical hyperplasticity using an implicit predictor--corrector method, and one using an explicit fourth order Runge-Kutta scheme to implement an h²plasticity version of the model. There is also code that symbolically implements a bifurcation matrix, using the notation established in the paper, but is otherwise generic. Finally, there is code that predicts the width of a localisation, assuming the band produced is perfectly horizontal.

Secondly, there is also a folder labelled replication. This folder contains a Manifest.toml file, a Project.toml file, and another README file detailing how to use these files to activate a version of Julia that is precisely identical to what I used to create the replication files. There are also four subfolders, each specifying an integrator and a load condition. Running the code in these folders with the Julia environment activated should generate files that are identical (plus or minus some differences related to operating systems and chip architecture) to the files that I created, which are stored on [Zenodo](https://zenodo.org/records/10926006). This will allow you to verify that the system works as expected, if you so desire.

## What you need to install in Julia
If you are running the codes in the repository with your own version of Julia, you will need to add the packages Einsum, FileIO, JLD2, LinearAlgebra, NLopt, PolynomialRoots, ProgressBars and SymEngine. You may well already have some or all of these installed!

## How to run the code
Provided you are in the same folder, the codes can be called very simply by:
```
include("h_squared_element_integration.jl")
```
or
```
include("hyperplastic_element_integration.jl")
```
to integrate the consitutive model (generally fast), and then:
```
include("horizontal_band_time_bifurcation.jl")
```
to get the bifurcation analysis (more time consuming). The process works equally well in the replication folders, the relevant codes have been included at that level for ease of use.

### Changing loading conditions
You can easily change the material parameters by editing their value in the corresponding file (although Julia will complain if you have already declared their value, this can be avoided by closing and relaunching the REPL). Similarly, you can easily change the initial elastic strains, and the applied loadings. By default, the system is assumed to be fully strain-controlled (that is the rate of strain in each entry is specified), but it is simple to change to (partially) stress-controlled (that is the rate of stress in a given entry is specified) by changing the corresponding Boolean value in the stress (or couple) control vector. The reproduction folder demonstrates the system integrated under constant volume shearing and constant confinining stress (in the 11 direction) shearing.

### Recommended usage
It should be highlighted that while the h² integrator is capable of doing stress control (and an example of this has been provided), it suffers from quite noticeable stress drift, so for cases of stress control, the hyperplastic integrator is strongly recommmended. This integrator is also stable with much larger time steps, so unless you really need the h² behaviour, the hyperplastic integrator is recommended. Correspondingly, the bifurcation analysis is much faster with fewer time steps.

## Attention!
This code is infrequently maintained, and it is not subject to any rigorous testing, or guarantees that it works on a version of Julia other than that contained in the Manifest and Project toml files. However, if it doesn't work for you, it shouldn't require much by way of debugging.

## Citing, help, contact etc
If you use these codes in your work, I would appreciate a citation to this repository, and the paper above. If you need help, want to discuss the code, or want to let me know about an error, I'm always contactable at nicholas[dot]collins-crat[at]inria[dot]fr

## Funding
This software has been partially developed during my Marie Skłodowska-Curie Actions Individual Fellowship. I acknowledge the support of the Marie Skłodowska-Curie Actions program under the Horizon Europe research and innovation framework program (Grant agreement ID 101064805 LEMMA). Views and opinions expressed are however those of the author only and do not necessarily reflect those of the European Union or Horizon Europe. Neither the European Union nor the granting authority can be held responsible for them.
