#= This is a code to integrate the Cosserat Breakage Mechanics model under
arbitrary boundary conditions i.e. with stress, strain or mixed controls=#

using FileIO
using DifferentialEquations
include("material_parameters.jl")
include("model_functions.jl")

p_crit_zero = sqrt(2*K*Ec/θ_γ)

#= Work out the initial confining elastic strain taking into account the
possibility of non-zero breakage =#
elastic_strain_initial = p_frac*p_crit_zero/(K*(1-θ_γ*B_nought))

# Stress control matrix. If 1, stress controlled entry, if 0 strain controlled
stress_control = falses(ndim, ndim)
couple_control = falses(ndim, ndim)
# INSERT STRESS CONTROLS HERE IF NECESSARY
stress_control[1, 1] = true
#=couple_control[INDEX] = 1=#

# Weld together into general control vector
control = [reshape(stress_control, ndim^2); reshape(couple_control, ndim^2)]
# (False) loading rates (we will recalculate ones subject to stress control)
γtotdot = [0 strain_rate 0; strain_rate 0 0; 0 0 0]
κtotdot = [0. 0. 0.; 0. 0. 0.; 0. 0. 0.]
# Stress and couple *rates*
stressdot = [0 0 0; 0 0 0; 0 0 0]
coupledot = [0 0 0; 0 0 0; 0 0 0]

# Initial conditions
γ_e = [elastic_strain_initial 0 0; 0 0 0; 0 0 0]
κ_e = [0. 0. 0.; 0. 0. 0.; 0. 0. 0.]
B = B_nought

# Generalised strain vector
generalised_strain_rate = vcat(reshape(γtotdot, (ndim^2)), reshape(κtotdot, (ndim^2)))
# Generalised stress vector
generalised_stress_rate = vcat(reshape(stressdot, (ndim^2)), reshape(coupledot, (ndim^2)))

# Create the vector listing the points where we need to save the results
save_times = [i for i in 0:n_timesteps]

# Tell the algorithm how long to integrate for
time_span = (convert(Float64, 0), convert(Float64, n_timesteps))
#= Make the initial vector of state variables (interpreting strain and stress
rates as state variables so that we have their value over time)=#
u0 = vcat(B, reshape(γ_e, (ndim^2)), reshape(κ_e, (ndim^2)), generalised_strain_rate, generalised_stress_rate)
# Create the problem
prob = ODEProblem(evolution_laws_control!, u0, time_span, control)
#= Specify the algorithm and tell it not to use autodiff, which throws a
type error for all the implicit methods=#
alg = Rodas5(autodiff=false)
# Solve that bad boy, giving a dtmax so that it doesn't overshoot and run into problems
sol = solve(prob, alg, saveat=save_times, dtmax=0.5)
# Get the state variables out and reshape them to their proper tensor form
Bvals = sol[1, :]
γ_e_vals = reshape(sol[2:10, :], (ndim, ndim, n_timesteps+1))
κ_e_vals = reshape(sol[11:19, :], (ndim, ndim, n_timesteps+1))
γdot_vals = reshape(sol[20:28, :], (ndim, ndim, n_timesteps+1))
κdot_vals = reshape(sol[29:37, :], (ndim, ndim, n_timesteps+1))
τdot_vals = reshape(sol[38:46, :], (ndim, ndim, n_timesteps+1))
μdot_vals = reshape(sol[47:55, :], (ndim, ndim, n_timesteps+1))

# Calculate the resultant stresses etc
function associated_values()
    stresses = zeros(Float64, (ndim, ndim, n_timesteps+1))
    couples = zeros(Float64, (ndim, ndim, n_timesteps+1))
    ps = zeros(Float64, (n_timesteps+1))
    qs = zeros(Float64, (n_timesteps+1))
    ymix_vals = zeros(Float64, (n_timesteps+1))
    Eep_res = zeros(Float64, (ndim, ndim, ndim, ndim, n_timesteps+1))
    Fep_res = zeros(Float64, (ndim, ndim, ndim, ndim, n_timesteps+1))
    Kep_res = zeros(Float64, (ndim, ndim, ndim, ndim, n_timesteps+1))
    Mep_res = zeros(Float64, (ndim, ndim, ndim, ndim, n_timesteps+1))
    Ires = zeros(Float64, (n_timesteps+1))
    total_strains = zeros(Float64, (ndim, ndim, n_timesteps+1))
    total_curvatures = zeros(Float64, (ndim, ndim, n_timesteps+1))
    for t in 1:n_timesteps+1
        stresses[:, :, t] = τ(Bvals[t], γ_e_vals[:, :, t])
        couples[:, :, t] = μ(Bvals[t], κ_e_vals[:, :, t])
        ps[t] = p(Bvals[t], γ_e_vals[:, :, t])
        qs[t] = q(Bvals[t], γ_e_vals[:, :, t], κ_e_vals[:, :, t])
        ymix_vals[t] = ymix(Bvals[t], γ_e_vals[:, :, t], κ_e_vals[:, :, t])
        Eep_res[:, :, :, :, t] = Eep(Bvals[t], γ_e_vals[:, :, t], κ_e_vals[:, :, t])
        Fep_res[:, :, :, :, t] = Fep(Bvals[t], γ_e_vals[:, :, t], κ_e_vals[:, :, t])
        Kep_res[:, :, :, :, t] = Kep(Bvals[t], γ_e_vals[:, :, t], κ_e_vals[:, :, t])
        Mep_res[:, :, :, :, t] = Mep(Bvals[t], γ_e_vals[:, :, t], κ_e_vals[:, :, t])
        Ires[t] = I_rotational(Bvals[t])
        if t == 1
            total_strains[:, :, t] = γ_e_vals[:, :, t]
            total_curvatures[:, :, t] = κ_e_vals[:, :, t]
        else
            total_strains[:, :, t] = total_strains[:, :, t-1] + γdot_vals[:, :, t]
            total_curvatures[:, :, t] = total_curvatures[:, :, t-1] + κdot_vals[:, :, t]
        end
    end
    return stresses, couples, ps, qs, ymix_vals, Eep_res, Fep_res, Kep_res, Mep_res, Ires, total_strains, total_curvatures
end
results = associated_values()
stresses = results[1]
couples = results[2]
ps = results[3]
qs = results[4]
ymix_vals = results[5]
Eep_res = results[6]
Fep_res = results[7]
Kep_res = results[8]
Mep_res = results[9]
Ires = results[10]
total_strains = results[11]
total_curvatures = results[12]
plastic_strains = total_strains - γ_e_vals
plastic_curvatures = total_curvatures - κ_e_vals

# Now save them
save(save_filename, Dict("Bres"=>Bvals, "γeres"=>γ_e_vals, "κeres"=>κ_e_vals,
"γdotres"=>γdot_vals, "κdotres"=>κdot_vals, "τdotes"=>τdot_vals,
"μdotres"=>μdot_vals, "τres"=>stresses, "μres"=>couples, "pres"=>ps, "qres"=>qs,
"yres"=>ymix_vals, "Eepres"=>Eep_res, "Fepres"=>Fep_res, "Kepres"=>Kep_res,
"Mepres"=>Mep_res, "γtotres"=>total_strains, "κtotres"=>total_curvatures,
"γpres"=>plastic_strains, "κpres"=>plastic_curvatures))
