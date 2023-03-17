# The configuration file that sets the material parameters

# Set the dimensionality
const ndim = 3

# Set the things we require to calculate the thetas
const dmin = 0.001
const α = 2.6
const dmax = 0.105

# Set the material values
const K = 13833
const G = 7588
const ζ = 1
const h1 = 6/5.
const h2 = 3/10.
const h3 = 6/(5*(ζ^2))
const h4 = 3/(10*(ζ^2))
const Gc = 3*G/(2*(h1-h2))
const L = 0.000001
const H = 3*G/(2*(h3+h4))
const Hc = 3*G/(2*(h3-h4))
const θ_γ = 1 - ((3-α)/(5-α))*((1-(dmin/dmax)^(5-α))/(1-(dmin/dmax)^(3-α)))
const θ_κ = 1 - ((3-α)/(7-α))*((1-(dmin/dmax)^(7-α))/(1-(dmin/dmax)^(3-α)))
const θ_I = 1 - ((3-α)/(8-α))*((1-(dmin/dmax)^(8-α))/(1-(dmin/dmax)^(3-α)))
const Ec = 4.65
const M = 1.7
const ω = 70.
const x_r = dmax
const β = 1.
const ξ = 100.
const ρ = 2.5*10^(-3)

# Set the initial values that we care about
const p_frac = 0.3
const B_nought = 0.0

# Strain targets
const γ_target = 0.2
const n_timesteps = 100
const strain_rate = γ_target/n_timesteps

# Numerical integration parameters (only dtmax useful for this model)
const dtmax = 0.5
const rate_independent = true
const rate_norm_tol = 1e-6

# Optimisation parameters for finding the values of n and Λ
const Λmax_limit = 200.0
const optimisation_iterations = 20
const optimisation_x_tol = 0.001

# Set the save file for the data in julia's version of pickling
save_folder = "YOUR_SAVE_FOLDER_HERE"
save_filename = save_folder * "integration_results.jld2"
