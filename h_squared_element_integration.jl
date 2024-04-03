#= This is a code to integrate the Cosserat Breakage Mechanics model under arbitrary boundary conditions i.e. with stress, strain or mixed controls. 
 The file contains an h²plasticity model, both described in the paper "A Cosserat Breakage Mechanics model for brittle granular media",
 by N.A. Collins-Craft, I. Stefanou, J. Sulem & I. Einav, in the Journal of the Mechanics and Phyics of Solids (2020). 
 Publsihed version: https://www.sciencedirect.com/science/article/pii/S0022509620302106
 Open access version: https://hal.science/hal-03120686v1 =#

 # Import the necessary packages
using FileIO
using ProgressBars


# Import the necessary functions and parameters
include("material_parameters.jl")
include("model_functions.jl")

# Calculate the critical confining pressure at which breakage occurs (assuming B=0)
p_crit_zero = sqrt(2*K*Ec/θ_γ)

# Work out the initial confining elastic strain taking into account the possibility of non-zero breakage
elastic_strain_initial = p_frac*p_crit_zero/(K*(1 - θ_γ*B_nought))

# Stress control vector. If 1, stress controlled entry, if 0 strain controlled
stress_control = falses(9)
couple_control = falses(9)
# INSERT STRESS CONTROLS HERE IF NECESSARY
#stress_control[1] = true
#stress_control[5] = true
#stress_control[9] = true
#=couple_control[INDEX] = true=#

# Weld together into general control vector
stress_control_vector = vcat(stress_control, couple_control)
# (False) loading rates (we will recalculate ones subject to stress control)
γ_t_dot_applied = 0.0025
γ_t_dot = transpose([0. γ_t_dot_applied 0. γ_t_dot_applied 0. 0. 0. 0. 0.])
κ_t_dot = transpose([0. 0. 0. 0. 0. 0. 0. 0. 0.])
# Stress and couple *rates*
ε_t_dot = vcat(γ_t_dot, κ_t_dot)
τ_dot = transpose([0. 0. 0. 0. 0. 0. 0. 0. 0.])
μ_dot = transpose([0. 0. 0. 0. 0. 0. 0. 0. 0.])
σ_dot = vcat(τ_dot, μ_dot)
# Create dumny elastic and plastic loading rates
γ_e_dot = γ_t_dot
γ_p_dot_val = zeros(Float64, (9))

# Initial conditions
γ_e = transpose([elastic_strain_initial 0. 0. 0. 0. 0. 0. 0. 0.])
κ_e = transpose([0. 0. 0. 0. 0. 0. 0. 0. 0.])
B = 0.0
λ = 0.0


# Specify the h²plasticity model loading protocol, using explicit integration
function loading_protocol_h_squared_explicit(γ_e, κ_e, γ_t_dot, κ_t_dot, σ_dot, stress_control_vector, B)
    τ_vals = zeros(Float64, (9, n_timesteps + 1))
    μ_vals = zeros(Float64, (9, n_timesteps + 1))
    γ_e_vals = zeros(Float64, (9, n_timesteps + 1))
    κ_e_vals = zeros(Float64, (9, n_timesteps + 1))
    y_vals = zeros(Float64, (n_timesteps + 1))
    γ_e_dot_vals = zeros(Float64, (9, n_timesteps + 1))
    γ_p_dot_vals = zeros(Float64, (9, n_timesteps + 1))
    γ_t_dot_vals = zeros(Float64, (9, n_timesteps + 1))
    γ_p_acc_vals = zeros(Float64, (9, n_timesteps + 1))
    γ_t_acc_vals = zeros(Float64, (9, n_timesteps + 1))
    κ_e_dot_vals = zeros(Float64, (9, n_timesteps + 1))
    κ_p_dot_vals = zeros(Float64, (9, n_timesteps + 1))
    κ_t_dot_vals = zeros(Float64, (9, n_timesteps + 1))
    κ_p_acc_vals = zeros(Float64, (9, n_timesteps + 1))
    κ_t_acc_vals = zeros(Float64, (9, n_timesteps + 1))
    p_vals = zeros(Float64, (n_timesteps + 1))
    q_vals = zeros(Float64, (n_timesteps + 1))
    B_vals = zeros(Float64, (n_timesteps + 1))
    B_dot_vals = zeros(Float64, (n_timesteps + 1))
    EB_vals = zeros(Float64, (n_timesteps + 1))
    λ_vals = zeros(Float64, (n_timesteps + 1))
    I_vals = zeros(Float64, (3, 3, n_timesteps + 1))
    # Declare a holding value for the plastic and total strains and curvatures
    γ_p_accumulated = zeros(Float64, (9))
    γ_t_accumulated = zeros(Float64, (9))
    κ_p_accumulated = zeros(Float64, (9))
    κ_t_accumulated = zeros(Float64, (9))
    # Declare storage matrices for the incremental stiffnesses
    Eep_vals = zeros(Float64, (9, 9, n_timesteps + 1))
    Fep_vals = zeros(Float64, (9, 9, n_timesteps + 1))
    Kep_vals = zeros(Float64, (9, 9, n_timesteps + 1))
    Mep_vals = zeros(Float64, (9, 9, n_timesteps + 1))

    # If there is stress control, pre-allocate the stiffness matrices we will use to estimate the total strain rates
    if sum(stress_control_vector) >= 1
        # First allocate a big stiffness matrix
        stiffness = zeros(Float64, (18, 18))
        # Create a generic vector combining the stress and strain rates
        ε_dot = vcat(γ_t_dot, κ_t_dot)
    end

    # Loop through time
    for time_step in ProgressBar(1:n_timesteps + 1)
        # Do the required calculations if there is stress control to determine the strain and curvature rates to feed to the RK4 integrator
        if sum(stress_control_vector) >= 1
            stiffness[1:9, 1:9] = Eep_h_squared(B, γ_e, κ_e)
            stiffness[1:9, 10:18] = Fep_h_squared(B, γ_e, κ_e)
            stiffness[10:18, 1:9] = Kep_h_squared(B, γ_e, κ_e)
            stiffness[10:18, 10:18] = Mep_h_squared(B, γ_e, κ_e)
            # Then chop it up into the bits we need (we don't need all submatrices)
            stiffness_KK = stiffness[stress_control_vector, .!stress_control_vector]
            stiffness_KU = stiffness[stress_control_vector, stress_control_vector]
            stiffness_KU_inv = inv(stiffness_KU)
            # First calculate the unknown strain rates corresponding to known stress rates
            ε_dot_u = stiffness_KU_inv*((σ_dot[stress_control_vector] - stiffness_KK*ε_t_dot[.!stress_control_vector]))
            # Replace unknown strain rate with calculated result
            j = 1
            for i = 1:size(ε_dot)[1]
                if stress_control_vector[i] == 1
                    ε_dot[i] = ε_dot_u[j]
                    j = j + 1
                end
            end
            # Now get the strain and curvature rates back out
            γ_t_dot = ε_dot[1:9]
            κ_t_dot = ε_dot[10:18]
        end
        integration_results = RK4_explicit(B, γ_e, κ_e, γ_t_dot, κ_t_dot, h_step)
        B_update = integration_results[1]
        γ_e_update = integration_results[2]
        κ_e_update = integration_results[3]
        γ_e_dot = (γ_e_update - γ_e)/h_step
        γ_p_dot_val = γ_t_dot - γ_e_dot
        κ_e_dot = (κ_e_update - κ_e)/h_step
        κ_p_dot_val = κ_t_dot - κ_e_dot
        B_dot_val = (B_update - B)/h_step
        B = B_update
        γ_e = γ_e_update
        κ_e = κ_e_update
        
        γ_p_accumulated = γ_p_accumulated + h_step*γ_p_dot_val
        γ_t_accumulated = γ_t_accumulated + h_step*γ_t_dot
        κ_p_accumulated = κ_p_accumulated + h_step*κ_p_dot_val
        κ_t_accumulated = κ_t_accumulated + h_step*κ_t_dot
 
        # Call the functions to calculate the incremental elasto-plastic stiffnesses
        Eep_val = Eep_h_squared(B, γ_e, κ_e)
        Fep_val = Fep_h_squared(B, γ_e, κ_e)
        Kep_val = Kep_h_squared(B, γ_e, κ_e)
        Mep_val = Mep_h_squared(B, γ_e, κ_e)

        # Save everything
        τ_vals[:, time_step] = τ(B, γ_e)
        μ_vals[:, time_step] = μ(B, κ_e)
        γ_e_vals[:, time_step] = γ_e
        κ_e_vals[:, time_step] = κ_e
        y_vals[time_step] = ymix(B, γ_e, κ_e)
        γ_e_dot_vals[:, time_step] = γ_e_dot
        γ_p_dot_vals[:, time_step] = γ_p_dot_val
        γ_t_dot_vals[:, time_step] = γ_t_dot
        γ_p_acc_vals[:, time_step] = γ_p_accumulated
        γ_t_acc_vals[:, time_step] = γ_t_accumulated
        κ_e_dot_vals[:, time_step] = κ_e_dot
        κ_p_dot_vals[:, time_step] = κ_p_dot_val
        κ_t_dot_vals[:, time_step] = κ_t_dot
        κ_p_acc_vals[:, time_step] = κ_p_accumulated
        κ_t_acc_vals[:, time_step] = κ_t_accumulated
        p_vals[time_step] = p(B, γ_e)
        q_vals[time_step] = q(B, γ_e, κ_e)
        B_vals[time_step] = B
        B_dot_vals[time_step] = B_dot_val
        EB_vals[time_step] = EB(γ_e, κ_e)
        λ_vals[time_step] = λfunc(B, γ_e, κ_e, γ_t_dot, κ_t_dot)
        I_vals[:, :, time_step] = (π/60)*(1 - θ_I*B)*ρ_s*(dmax^5)*I(3)
        # Save all the incremental tensors as well
        Eep_vals[:, :, time_step] = Eep_val
        Fep_vals[:, :, time_step] = Fep_val
        Kep_vals[:, :, time_step] = Kep_val
        Mep_vals[:, :, time_step] = Mep_val
    end
    return τ_vals, μ_vals, γ_e_vals, κ_e_vals, y_vals, γ_e_dot_vals, γ_p_dot_vals, γ_t_dot_vals, γ_p_acc_vals, γ_t_acc_vals, κ_e_dot_vals, κ_p_dot_vals, κ_t_dot_vals, κ_p_acc_vals, κ_t_acc_vals, p_vals, q_vals, B_vals, B_dot_vals, EB_vals, λ_vals, I_vals, Eep_vals, Fep_vals, Kep_vals, Mep_vals
end


# Get the results of the simulation
results = loading_protocol_h_squared_explicit(γ_e, κ_e, γ_t_dot, κ_t_dot, σ_dot, stress_control_vector, B)
τres = reshape(results[1], (3, 3, n_timesteps + 1))
μres = reshape(results[2], (3, 3, n_timesteps + 1))
γeres = reshape(results[3], (3, 3, n_timesteps + 1))
κeres = reshape(results[4], (3, 3, n_timesteps + 1))
yres = results[5]
γedotres = reshape(results[6], (3, 3, n_timesteps + 1))
γpdotres = reshape(results[7], (3, 3, n_timesteps + 1))
γtdotres = reshape(results[8], (3, 3, n_timesteps + 1))
γpres = reshape(results[9], (3, 3, n_timesteps + 1))
γtotres = reshape(results[10], (3, 3, n_timesteps + 1))
κedotres = reshape(results[11], (3, 3, n_timesteps + 1))
κpdotres = reshape(results[12], (3, 3, n_timesteps + 1))
κtdotres = reshape(results[13], (3, 3, n_timesteps + 1))
κpres = reshape(results[14], (3, 3, n_timesteps + 1))
κtotres = reshape(results[15], (3, 3, n_timesteps + 1))
pres = results[16]
qres = results[17]
Bres = results[18]
Bdotres = results[19]
EBres = results[20]
λres = results[21]
Ires = results[22]
Eepres = reshape(results[23], (3, 3, 3, 3, n_timesteps + 1))
Fepres = reshape(results[24], (3, 3, 3, 3, n_timesteps + 1))
Kepres = reshape(results[25], (3, 3, 3, 3, n_timesteps + 1))
Mepres = reshape(results[26], (3, 3, 3, 3, n_timesteps + 1))
# Create a vector of densities (which don't change in this model, but can in general)
ρres = ones(Float64, (n_timesteps + 1))*ρ_s

# Now save them
save(save_filename, Dict("Bres"=>Bres, "γeres"=>γeres,"κeres"=>κeres, "τres"=>τres, "μres"=>μres, "pres"=>pres, "qres"=>qres,"yres"=>yres, "EBres"=>EBres, "Eepres"=>Eepres, "Fepres"=>Fepres, "Kepres"=>Kepres, "Mepres"=>Mepres, "γtotres"=>γtotres, "κtotres"=>κtotres, "γpres"=>γpres, "κpres"=>κpres, "γedotres"=>γedotres, "γpdotres"=>γpdotres, "γtdotres"=>γtdotres, "κedotres"=>κedotres, "κpdotres"=>κpdotres, "κtdotres"=>κtdotres, "Bdotres"=>Bdotres, "λres"=>λres, "Ires"=>Ires, "ρres"=>ρres))
