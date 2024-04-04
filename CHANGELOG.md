# Changelog
## Version 1.1.0
 - Removed dependence on TensorOperations to construct the bifurcation matrix and build the model codes, as it had completely broken compatibility with SymEngine. This functionality has been replaced with Einsum.
 - New model codes based on two dimensional matrices and single vectors rather than four dimensional tensors and two dimensional tensors implemented, substantially speeding up the system.
 - Integration in time is now carried out manually, rather than using DifferentEquations.jl. This reduces flexibility, but is made up for by the much faster integration speed due to the specialisation of the schemes.
 - A classical hyperplastic implementation of the model is now provided (and is the recommended way of using the code).
 - NLopt has replaced Optim as the optimisation solver to determine the wavelength in the bifurcation analysis, for reasons of coherence with the hyperplastic solver.
 - A replication folder has been provided, that provides four example systems (the two integrators, and two loading conditions each), and Project.toml and Manifest.toml files that allow the user to have precisely the same system as me when running the systems. The outputs of these simulations are stored on Zotero for reasons of maintaining a small repository size.
 - The repository has been brought to Software Heritage standards, and is now archived there.
