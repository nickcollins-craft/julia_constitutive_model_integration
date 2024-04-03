#= The configuration file that sets the material parameters. The values are those given in Table 2 of the paper "A Cosserat Breakage Mechanics model for brittle granular media",
 by N.A. Collins-Craft, I. Stefanou, J. Sulem & I. Einav, in the Journal of the Mechanics and Phyics of Solids (2020). 
 Publsihed version: https://www.sciencedirect.com/science/article/pii/S0022509620302106
 Open access version: https://hal.science/hal-03120686v1 =#

# Set the the grain size distribution parameters
const dmin::Float64 = 0.001  # Minimum grain size
const α::Float64 = 2.6  # Power law term (between 2 and 3)
const dmax::Float64 = 0.105  # Maximum grain size

# Set the material values
const K::Float64 = 13833.  # Bulk stiffness (MPa)
const G::Float64 = 7588.  # Shear stiffness (MPa)
const ζ::Float64 = 1.  # Elastic-to-plastic length scale ratio
const h1::Float64 = 6/5.  # First 3D Cosserat kinematic model parameter
const h2::Float64 = 3/10.  # Second 3D Cosserat kinematic model parameter
const h3::Float64 = 6/(5*(ζ^2))  # Third 3D Cosserat kinematic model parameter
const h4::Float64 = 3/(10*(ζ^2))  # Fourth 3D Cosserat kinematic model parameter
const Gc::Float64 = 3*G/(2*(h1 - h2))  # First Cosserat shear stiffness (MPa)
const L::Float64 = 0.000001  # Cosserat torsional stiffenss (zero, but given a small value for numerical reasons)
const H::Float64 = 3*G/(2*(h3 + h4))  # Second Cosserat shear stiffness (MPa)
const Hc::Float64 = 3*G/(2*(h3 - h4))  # Third Cosserat shear stiffness (MPa)
const θ_γ::Float64 = 1 - ((3 - α)/(5 - α))*((1 - (dmin/dmax)^(5 - α))/(1 - (dmin/dmax)^(3 - α)))  # Grain size parameter for the strains
const θ_κ::Float64 = 1 - ((3 - α)/(7 - α))*((1 - (dmin/dmax)^(7 - α))/(1 - (dmin/dmax)^(3 - α)))  # Grain size parameter for the curvatures
const θ_I::Float64 = 1 - ((3 - α)/(8 - α))*((1 - (dmin/dmax)^(8 - α))/(1 - (dmin/dmax)^(3 - α)))  # Grain size parameter for the inertias
const Ec::Float64 = 4.65  # Grain crushing strength (MPa)
const M::Float64 = 1.7  # Critical state line slope in the p-q plane
const ω::Float64 = 70.  # Plastic coupling angle that controls the ratio between the plastic volumetric shear and the grain breakage
const x_r::Float64 = dmax  # Reference length for Cosserat
const ξ::Float64 = 100.  # Power exponent for h^2 plasticity
const ρ_s::Float64 = 2.5*10^(-3)  # Solid grain density (g/mm^3), which will be treated as the material density in this model

# Set the initial values that we care about
const p_frac::Float64 = 0.3  # Initial confining pressure
const B_nought::Float64 = 0.0  # Initial breakage value

# Set the time integration parameters
tmax::Float64 = 80.0  # Maximum integration time
h_step::Float64 = 0.005  # Time step size
n_timesteps::Int = Int(ceil(tmax/h_step))  # Number of time steps

# Optimisation parameters for finding the value of Λ
const Λmax_limit::Float64 = 200.0  # Largest wavelength to consider for localisation
const optimisation_iterations::Int = 20  # Number of iterations for the optimisation solver
const optimisation_x_tol::Float64 = 0.001  # Tolerance of the optimisation solver in the x direction (Λ)

# Set the save file for the data in Julia's version of pickling (in this case we use the save folder to merely add a prefix to the file)
save_folder = "h_squared_constant_volume_"
save_filename = save_folder * "integration_results.jld2"
