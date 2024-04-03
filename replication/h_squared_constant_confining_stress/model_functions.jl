#= This file holds the model functions from the paper "A Cosserat Breakage Mechanics model for brittle granular media",
 by N.A. Collins-Craft, I. Stefanou, J. Sulem & I. Einav, in the Journal of the Mechanics and Phyics of Solids (2020). 
 Publsihed version: https://www.sciencedirect.com/science/article/pii/S0022509620302106
 Open access version: https://hal.science/hal-03120686v1 =#

# Import the necessary packages
using NLopt
using LinearAlgebra


# Get the material parameters
include("material_parameters.jl")


# Kronecker delta function
function KroneckerDelta(i::Int, j::Int)::Int
    if i == j
        kron = 1
    else
        kron = 0
    end
    return kron
end


# Macaulay bracket function
function mac(x)
    if x >= 0
        y = x
    else
        y = 0
    end
    return y
end


# Fill out the elastic stiffness tensors
Celastic = zeros(Float64, (3, 3, 3, 3))
Delastic = zeros(Float64, (3, 3, 3, 3))
for i = 1:3
    for j = 1:3
        for k = 1:3
            for l = 1:3
                Celastic[i, j, k, l] = (K - 2*G/3)*KroneckerDelta(i, j)*KroneckerDelta(k, l) + (G + Gc)*KroneckerDelta(i, k)*KroneckerDelta(j, l) + (G - Gc)*KroneckerDelta(i, l)*KroneckerDelta(j, k)
                Delastic[i, j, k, l] = (L - 2*H/3)*KroneckerDelta(i, j)*KroneckerDelta(k, l) + (H + Hc)*KroneckerDelta(i, k)*KroneckerDelta(j, l) + (H - Hc)*KroneckerDelta(i, l)*KroneckerDelta(j, k)
            end
        end
    end
end
# Reshape the tensors to be matrices
C_matrix = reshape(Celastic, (9, 9))
D_matrix = reshape(Delastic, (9, 9))


# τ function
function τ(B, γ_e)
    τval = (1 - θ_γ*B)*C_matrix*γ_e
    return τval
end


# Mean stress function
function p(B, γ_e)
    τ_val = τ(B, γ_e)
    p = (1/3)*(τ_val[1] + τ_val[5] + τ_val[9])
    return p
end


# Deviatoric stress function
function s_dev(B, γ_e)
    s_val = τ(B, γ_e)
    p_val = p(B, γ_e)
    s_val[1] = s_val[1] - p_val
    s_val[5] = s_val[5] - p_val
    s_val[9] = s_val[9] - p_val
    return s_val
end


# Couple stress function
function μ(B, κ_e)
    μval = (x_r^2)*(1 - θ_κ*B)*D_matrix*κ_e
    return μval
end


# Mean couple stress function
function p_μ(B, κ_e)
    μ_val = μ(B, κ_e)
    p_μ_val = (1/3)*(μ_val[1] + μ_val[5] + μ_val[9])
    return p_μ_val
end


# Deviatoric couple function
function m_dev(B, κ_e)
    m_val = μ(B, κ_e)
    p_μ_val = p_μ(B, κ_e)
    m_val[1] = m_val[1] - p_μ_val
    m_val[5] = m_val[5] - p_μ_val
    m_val[9] = m_val[9] - p_μ_val
    return m_val
end


# Elastic length scale
function l_e(B)
    # Add guards so the implicit solver doesn't blow up
    if B < 0.0
        B_use = 0.0
    elseif B > 1.0
        B_use = 1.0
    elseif B == NaN
        B_use = 0.0
    else
        B_use = B
    end
    ell = x_r*sqrt(1 - θ_κ*B_use)
    return ell
end


# Second deviatoric stress invariant
function q(B, γ_e, κ_e)
    s_val = reshape(s_dev(B, γ_e), (3, 3))
    m_val = reshape(m_dev(B, κ_e), (3, 3))
    q_sum = 0
    for i = 1:3
        for j = 1:3
            q_sum = q_sum + h1*s_val[i, j]*s_val[i, j] + h2*s_val[i, j]*s_val[j, i] + (1/(l_e(B))^2)*(h3*m_val[i, j]*m_val[i, j] + h4*m_val[i, j]*m_val[j, i])
        end
    end
    q_val = sqrt(q_sum)
    return q_val
end


# Total stress-like variables
function σ(B, ε_e)
    σ_val = vcat(τ(B, ε_e[1:9]), μ(B, ε_e[10:18]))
    return σ_val
end


# Breakage energy
function EB(γ_e, κ_e)
    first_product = C_matrix*γ_e
    second_product = D_matrix*κ_e
    EB_val = (θ_γ/2)*transpose(first_product)*γ_e + (θ_κ/2)*(x_r^2)*transpose(second_product)*κ_e
    return EB_val[1]
end


# Yield surface
function ymix(B, γ_e, κ_e)
    ymix = ((EB(γ_e, κ_e)*((1 - B)^2))/Ec) + ((q(B, γ_e, κ_e)/(M*p(B, γ_e)))^2) - 1
    return ymix
end


# functions for the flow rules (minus λ)
function dyΦdτ(B, γ_e, κ_e)
    sbit = reshape(s_dev(B, γ_e), (3, 3))
    sbit_T = transpose(sbit)
    deviatorpart = 2*(h1*sbit + h2*sbit_T)/((M*p(B, γ_e))^2)
    tracepart = ((2*EB(γ_e, κ_e)*((1 - B)^2)*(sind(ω))^2)/(3*p(B, γ_e)*Ec))*Matrix(1.0I, 3, 3)
    dyΦdτ_val = tracepart + deviatorpart
    return reshape(dyΦdτ_val, (9, 1))
end


function dyΦdμ(B, γ_e, κ_e)
    mbit = reshape(m_dev(B, κ_e), (3, 3))
    mbit_T = transpose(mbit)
    dyΦdμ_val = (2*(h3*mbit + h4*mbit_T))/((l_e(B)*M*p(B, γ_e))^2)
    return reshape(dyΦdμ_val, (9, 1))
end


function dyΦdEB(B)
    dyΦdEB_val = (2*((1 - B)^2)*(cosd(ω))^2)/Ec
    return dyΦdEB_val
end


# Group them together as flow rates
function flow_rates(state_vec)
    B = state_vec[1]
    γ_e = state_vec[2:10]
    κ_e = state_vec[11:19]
    # Negatives for the B and φ, because they evolve positively with λ (whereas γ_e and κ_e will reduce in a given time step)
    flow_set = vcat([-dyΦdEB(B)], dyΦdτ(B, γ_e, κ_e), dyΦdμ(B, γ_e, κ_e))
    return flow_set
end


# Now we write a series of helper functions on the way to λ
function dydτ(B, γ_e, κ_e)
    sbit = reshape(s_dev(B, γ_e), (3, 3))
    sbit_T = transpose(sbit)
    deviatorpart = 2*(h1*sbit + h2*sbit_T)/((M*p(B, γ_e))^2)
    tracepart = ((-2*(q(B, γ_e, κ_e)^2))/(3*(M^2)*(p(B, γ_e)^3)))*Matrix(1.0I, 3, 3)
    dydτ_val = tracepart + deviatorpart
    return reshape(dydτ_val, (9, 1))
end


function dydμ(B, γ_e, κ_e)
    mbit = reshape(m_dev(B, κ_e), (3, 3))
    mbit_T = transpose(mbit)
    dydμ_val = (2/((l_e(B)*M*p(B, γ_e))^2))*(h3*mbit + h4*mbit_T)
    return reshape(dydμ_val, (9, 1))
end


function dydEB(B)
    dydEB_val = ((1 - B)^2)/Ec
    return dydEB_val
end


function dydB(B, γ_e, κ_e)
    dydB_val = (-2*(1 - B)*EB(γ_e, κ_e))/Ec
    return dydB_val
end


function dτdγ(B)
    dτdγ_val = (1 - θ_γ*B)*C_matrix
    return dτdγ_val
end


function dτdB(γ_e)
    dτdB_val = -θ_γ*C_matrix*γ_e
    return dτdB_val
end


function dμdκ(B)
    dμdκ_val = (x_r^2)*(1 - θ_κ*B)*D_matrix
    return dμdκ_val
end


function dμdB(κ_e)
    dμdB_val = -θ_κ*(x_r^2)*D_matrix*κ_e
    return dμdB_val
end


function dEBdγ(γ_e)
    dEBdγ_val = θ_γ*C_matrix*γ_e
    return dEBdγ_val
end


function dEBdκ(κ_e)
    dEBdκ_val = (x_r^2)*θ_κ*D_matrix*κ_e
    return dEBdκ_val
end


function λ_numerator_γ_part(B, γ_e, κ_e)
    γpart_val = dτdγ(B)*dydτ(B, γ_e, κ_e) + dydEB(B)*dEBdγ(γ_e)
    return γpart_val
end


function λ_numerator_κ_part(B, γ_e, κ_e)
    κpart_val = dμdκ(B)*dydμ(B, γ_e, κ_e) + dydEB(B)*dEBdκ(κ_e)
    return κpart_val
end


function λ_denominator_stress_part(B, γ_e, κ_e)
    stress_part_val = transpose(dydτ(B, γ_e, κ_e))*(dτdγ(B)*dyΦdτ(B, γ_e, κ_e) - dτdB(γ_e)*dyΦdEB(B))
    return stress_part_val[1, 1]
end


function λ_denominator_couple_part(B, γ_e, κ_e)
    couple_part_val = transpose(dydμ(B, γ_e, κ_e))*(dμdκ(B)*dyΦdμ(B, γ_e, κ_e) - dμdB(κ_e)*dyΦdEB(B))
    return couple_part_val[1, 1]
end


function λ_denominator_energy_part(B, γ_e, κ_e)
    energy_part_val = dydEB(B)*(transpose(dEBdγ(γ_e))*dyΦdτ(B, γ_e, κ_e) + transpose(dEBdκ(κ_e))*dyΦdμ(B, γ_e, κ_e))
    return energy_part_val[1, 1]
end


function λ_denominator_breakage_part(B, γ_e, κ_e)
    breakagepart = dydB(B, γ_e, κ_e)*dyΦdEB(B)
    return breakagepart
end


function λ_denominator(B, γ_e, κ_e)
    λdenom = λ_denominator_stress_part(B, γ_e, κ_e) + λ_denominator_couple_part(B, γ_e, κ_e) + λ_denominator_energy_part(B, γ_e, κ_e) - λ_denominator_breakage_part(B, γ_e, κ_e)
    return λdenom
end


function λγ(B, γ_e, κ_e)
    λγ_val = λ_numerator_γ_part(B, γ_e, κ_e)/λ_denominator(B, γ_e, κ_e)
    return λγ_val
end


function λκ(B, γ_e, κ_e)
    λκ_val = λ_numerator_κ_part(B, γ_e, κ_e)/λ_denominator(B, γ_e, κ_e)
    return λκ_val
end


# Finally we function the plastic multiplier λ
function λfunc(B, γ_e, κ_e, γtotdot, κtotdot)
    λ_val = dot(λγ(B, γ_e, κ_e), γtotdot) + dot(λκ(B, γ_e, κ_e), κtotdot)
    return λ_val[1]
end


# We write the flow rules explicitly
function B_dot(B, γ_e, κ_e, γtotdot, κtotdot)
    bdot_val = λfunc(B, γ_e, κ_e, γtotdot, κtotdot)*dyΦdEB(B)
    return bdot_val
end


function γpdot(B, γ_e, κ_e, γtotdot, κtotdot)
    γpdot_val = λfunc(B, γ_e, κ_e, γtotdot, κtotdot)*dyΦdτ(B, γ_e, κ_e)
    return γpdot_val
end


function κpdot(B, γ_e, κ_e, γtotdot, κtotdot)
    κpdot_val = λfunc(B, γ_e, κ_e, γtotdot, κtotdot)*dyΦdμ(B, γ_e, κ_e)
    return κpdot_val
end


# We can n also write the flow rules in in h²plasticity form
function Bdot_h_squared(B, γ_e, κ_e, γtotdot, κtotdot)
    bdot = mac(λfunc(B, γ_e, κ_e, γtotdot, κtotdot))*((1 + ymix(B, γ_e, κ_e))^ξ)*dyΦdEB(B)
    return bdot
end


function γpdot_h_squared(B, γ_e, κ_e, γtotdot, κtotdot)
    γpdot = mac(λfunc(B, γ_e, κ_e, γtotdot, κtotdot))*((1 + ymix(B, γ_e, κ_e))^ξ)*dyΦdτ(B, γ_e, κ_e)
    return γpdot
end


function κpdot_h_squared(B, γ_e, κ_e, γtotdot, κtotdot)
    κpdot = mac(λfunc(B, γ_e, κ_e, γtotdot, κtotdot))*((1 + ymix(B, γ_e, κ_e))^ξ)*dyΦdμ(B, γ_e, κ_e)
    return κpdot
end


# It is convenient to write the evolution laws for the state variables other than B
function γ_e_dot_func(B, γ_e, κ_e, γtotdot, κtotdot)
    γedot = γtotdot - γpdot(B, γ_e, κ_e, γtotdot, κtotdot)
    return γedot
end


function κ_e_dot_func(B, γ_e, κ_e, γtotdot, κtotdot)
    κedot = κtotdot - κpdot(B, γ_e, κ_e, γtotdot, κtotdot)
    return κedot
end


# We can write the h²plasticity form too 
function γedot_h_squared(B, γ_e, κ_e, γtotdot, κtotdot)
    γedot = γtotdot - γpdot_h_squared(B, γ_e, κ_e, γtotdot, κtotdot)
    return γedot
end


function κedot_h_squared(B, γ_e, κ_e, γtotdot, κtotdot)
    κedot = κtotdot - κpdot_h_squared(B, γ_e, κ_e, γtotdot, κtotdot)
    return κedot
end


# We write some helper functions that ease the calculation of the incremental tensors
function A_ij(B, γ_e, κ_e)
    A_val = dτdγ(B)*dyΦdτ(B, γ_e, κ_e)
    return A_val
end


function B_ij(B, γ_e)
    B_val = dτdB(γ_e)*dyΦdEB(B)
    return B_val
end


function C_ij(B, γ_e, κ_e)
    C_val = dμdκ(B)*dyΦdμ(B, γ_e, κ_e)
    return C_val
end


function D_ij(B, κ_e)
    D_val = dμdB(κ_e)*dyΦdEB(B)
    return D_val
end


# Now we write the incremental tensors
function Eep(B, γ_e, κ_e)
    λ_γ_val = λγ(B, γ_e, κ_e)
    first_product = (A_ij(B, γ_e, κ_e) - B_ij(B, γ_e))*λ_γ_val'
    Eep_val = dτdγ(B) - first_product
    return Eep_val
end


function Fep(B, γ_e, κ_e)
    λ_κ_val = λκ(B, γ_e, κ_e)
    Fep_val = -(A_ij(B, γ_e, κ_e) - B_ij(B, γ_e))*λ_κ_val'
    return Fep_val
end


function Kep(B, γ_e, κ_e)
    λ_γ_val = λγ(B, γ_e, κ_e)
    Kep_val = -(C_ij(B, γ_e, κ_e) - D_ij(B, κ_e))*λ_γ_val'
    return Kep_val
end


function Mep(B, γ_e, κ_e)
    λ_κ_val = λκ(B, γ_e, κ_e)
    first_product = (C_ij(B, γ_e, κ_e) - D_ij(B, κ_e))*λ_κ_val'
    Mep_val = dμdκ(B) - first_product
    return Mep_val
end


# We can write them in in h²plasticity form too
function Eep_h_squared(B, γ_e, κ_e)
    λ_γ_val = λγ(B, γ_e, κ_e)
    first_product = ((1 + ymix(B, γ_e, κ_e))^ξ)*(A_ij(B, γ_e, κ_e) - B_ij(B, γ_e))*λ_γ_val'
    Eep_val = dτdγ(B) - first_product
    return Eep_val
end


function Fep_h_squared(B, γ_e, κ_e)
    λ_κ_val = λκ(B, γ_e, κ_e)
    Fep_val = -((1 + ymix(B, γ_e, κ_e))^ξ)*(A_ij(B, γ_e, κ_e) - B_ij(B, γ_e))*λ_κ_val'
    return Fep_val
end


function Kep_h_squared(B, γ_e, κ_e)
    λ_γ_val = λγ(B, γ_e, κ_e)
    Kep_val = -((1 + ymix(B, γ_e, κ_e))^ξ)*(C_ij(B, γ_e, κ_e) - D_ij(B, κ_e))*λ_γ_val'
    return Kep_val
end


function Mep_h_squared(B, γ_e, κ_e)
    λ_κ_val = λκ(B, γ_e, κ_e)
    first_product = ((1 + ymix(B, γ_e, κ_e))^ξ)*(C_ij(B, γ_e, κ_e) - D_ij(B, κ_e))*λ_κ_val'
    Mep_val = dμdκ(B) - first_product
    return Mep_val
end


# We write a function returning the rotational moment of inertia
function I_rotational(B)
    I_val = (π/60)*(1 - θ_I*B)*ρ*(x_r^5)
    return I_val
end


# Write a function to calculate the instantaneous state variable rates using a 4th order Runge-Kutta method
function RK4_rate(B, γ_e, κ_e, γ_dot, κ_dot, h_step)
    B_1 = Bdot_h_squared(B, γ_e, κ_e, γ_dot, κ_dot)
    γ_e_1 = γedot_h_squared(B, γ_e, κ_e, γ_dot, κ_dot)
    κ_e_1 = κedot_h_squared(B, γ_e, κ_e, γ_dot, κ_dot)
    B_2 = Bdot_h_squared(B + 0.5*h_step*B_1, γ_e + 0.5*h_step*γ_e_1, κ_e + 0.5*h_step*κ_e_1, γ_dot, κ_dot)
    γ_e_2 = γedot_h_squared(B + 0.5*h_step*B_1, γ_e + 0.5*h_step*γ_e_1, κ_e + 0.5*h_step*κ_e_1, γ_dot, κ_dot)
    κ_e_2 = κedot_h_squared(B + 0.5*h_step*B_1, γ_e + 0.5*h_step*γ_e_1, κ_e + 0.5*h_step*κ_e_1, γ_dot, κ_dot)
    B_3 = Bdot_h_squared(B + 0.5*h_step*B_2, γ_e + 0.5*h_step*γ_e_2, κ_e + 0.5*h_step*κ_e_2, γ_dot, κ_dot)
    γ_e_3 = γedot_h_squared(B + 0.5*h_step*B_2, γ_e + 0.5*h_step*γ_e_2, κ_e + 0.5*h_step*κ_e_2, γ_dot, κ_dot)
    κ_e_3 = κedot_h_squared(B + 0.5*h_step*B_2, γ_e + 0.5*h_step*γ_e_2, κ_e + 0.5*h_step*κ_e_2, γ_dot, κ_dot)
    B_4 = Bdot_h_squared(B + h_step*B_3, γ_e + h_step*γ_e_3, κ_e + h_step*κ_e_3, γ_dot, κ_dot)
    γ_e_4 = γedot_h_squared(B + h_step*B_3, γ_e + h_step*γ_e_3, κ_e + h_step*κ_e_3, γ_dot, κ_dot)
    κ_e_4 = κedot_h_squared(B + h_step*B_3, γ_e + h_step*γ_e_3, κ_e + h_step*κ_e_3, γ_dot, κ_dot)
    B_rate = (1/6)*(B_1 + 2*B_2 + 2*B_3 + B_4)
    γ_e_rate = (1/6)*(γ_e_1 + 2*γ_e_2 + 2*γ_e_3 + γ_e_4)
    κ_e_rate = (1/6)*(κ_e_1 + 2*κ_e_2 + 2*κ_e_3 + κ_e_4)
    return B_rate, γ_e_rate, κ_e_rate
end


# Write a 4th order explicit Runge-Kutta function to do the integration in time
function RK4_explicit(B, γ_e, κ_e, γ_dot, κ_dot, h_step)
    # Use the rate function to get the rates
    B_rate, γ_e_rate, κ_e_rate = RK4_rate(B, γ_e, κ_e, γ_dot, κ_dot, h_step)
    # Use the rates to get the increments
    B = B + h_step*B_rate
    γ_e = γ_e + h_step*γ_e_rate
    κ_e = κ_e + h_step*κ_e_rate
    return B, γ_e, κ_e
end


# Write a function to calculate the stress rate residual (for use in the predictor-corrector method)
function reduced_stress_rate_function(state_new_u, args)
    # Small reduction in complexity for sake of predictor-corrector method, we
    # assume a fully-elastic step
    ε_new, ε_current, σ_dot, stress_control_vector, B = args[1], args[2], args[3], args[4], args[5]
    ε_new[stress_control_vector] = state_new_u
    # Calculate the stress residual
    residual_full = ((σ(B, ε_new) - σ(B, ε_current))/h_step) - σ_dot
    residual_strains = residual_full[stress_control_vector]
    residual_squared = 0
    for i in 1:length(residual_strains)
        residual_squared = residual_squared + residual_strains[i]^2
    end
    residual = sqrt(residual_squared)
    return residual
end


# Write an elastic predictor function
function elastic_predictor(B, γ_e, κ_e, γ_t_dot, κ_t_dot, σ_dot, stress_control_vector)
    # Allocate all strain elastically
    γ_e_new = γ_e + h_step*γ_t_dot
    κ_e_new = κ_e + h_step*κ_t_dot
    ε_e = vcat(γ_e, κ_e)
    ε_e_new = vcat(γ_e_new, κ_e_new)

    # Check if stress-controlled entries
    if sum(stress_control_vector) >= 1
        # Specify the variable to be modified (dofs subject to stress control + φ)
        state_guess = ε_e_new[stress_control_vector]
        # Assemble the vector of arguments that the stress rate function requries
        args_vec = [ε_e_new, ε_e, σ_dot, stress_control_vector, B]
        # Solve the function such that the γ_e of the dofs under stress rate control respect that stress rate
        # Declare the optimisation object, the algorithm and tolerance
        opt = Opt(:LN_NELDERMEAD, sum(stress_control_vector))
        # Declare the function to optimise, using an anonymous function to turn it into a function of one vector (grad is dummy to match NLOpt syntax)
        opt.min_objective = (x, grad) -> reduced_stress_rate_function(x, args_vec)
        opt.ftol_abs = 1e-11
        # Optimise and get the result
        (minf, state_nl, ret) = optimize(opt, state_guess)
        ε_e_new[stress_control_vector] = state_nl
        γ_e_new = ε_e_new[1:9]
        κ_e_new = ε_e_new[10:18]
        γ_e = γ_e_new
        κ_e = κ_e_new
    else
        γ_e = γ_e_new
        κ_e = κ_e_new
    end
    results = vcat([B], γ_e, κ_e)
    return results
end


# Write a scalar predictor-corrector that only uses λ and the stress-controlled entries
function scalar_residual(solver_variables, args_vec)
    # Get the components of the arguments out
    trial_state, current_state, σ_dot, stress_control_vector, resi_vec, resi_control_vector = args_vec[1], args_vec[2], args_vec[3], args_vec[4], args_vec[5], args_vec[6]
    # Start with the initial guess forward
    guess_state = trial_state - (solver_variables[end]*h_step)*flow_rates(current_state)
    # Insert the stress controlled entries
    solver_variable_counter = 1
    for i in 1:18
        if stress_control_vector[i]
            guess_state[1 + i] = solver_variables[solver_variable_counter]
            solver_variable_counter = solver_variable_counter + 1
        end
    end
    # Now, construct the residual vector
    resi_vec[1:19] = trial_state - guess_state - (solver_variables[end]*h_step)*flow_rates(guess_state)
    # Append the yield surface to the residual vector
    resi_vec[20] = 10*ymix(guess_state[1], guess_state[2:10], guess_state[11:19])
    # Add this term in because somehow the stress control is lengthening???
    if length(stress_control_vector) > 18
        stress_control_vector = stress_control_vector[1:18]
    end
    if sum(stress_control_vector) > 0
        # Calculate the stress residual of the guess state, divided by K_bar to make it the same dimensionally
        σ_residual_full = ((σ(guess_state[1], guess_state[2:19]) - σ(current_state[1], current_state[2:19]))/h_step) - σ_dot
        σ_residual_controlled = σ_residual_full[stress_control_vector]
        # Replace the stress controlled entries of the residual vector with the stress residuals
        resi_vec[resi_control_vector] = σ_residual_controlled
    end
    residual_squared = 0
    for i in 1:length(resi_vec)
        residual_squared = residual_squared + resi_vec[i]^2
    end
    residual = sqrt(residual_squared)
    return residual
end


function scalar_plastic_corrector(trial_state, guess_state, current_state, corrected_state, σ_dot, stress_control_vector, resi_vec, resi_control_vector)
    # Create some useful extended control vectors
    # Assemble the args vector
    args_vec = [trial_state, current_state, σ_dot, stress_control_vector, resi_vec, resi_control_vector]
    # Create the bounds on the guess vector.
    left_bound = vcat(-Inf*ones(sum(stress_control_vector)), [0.0])
    right_bound = vcat(Inf*ones(sum(stress_control_vector)), [Inf])
    # Declare the optimisation object, the algorithm and its bounds and tolerance
    opt = Opt(:LN_NELDERMEAD, sum(stress_control_vector) + 1)
    opt.min_objective = (x, grad) -> scalar_residual(x, args_vec)
    opt.lower_bounds = left_bound
    opt.upper_bounds = right_bound
    opt.ftol_abs = 1e-11
    (minf, minimising_state, ret) = optimize(opt, guess_state)
    # Make the first correction step from the calculated minimiser
    corrected_state[1:19] = trial_state - (minimising_state[end]*h_step)*flow_rates(current_state)
    # Insert the stress-controlled entries (if they exist)
    if sum(stress_control_vector) > 0
        corrected_state[resi_control_vector] = minimising_state[1:end - 1]
    end
     corrected_state[20] = minimising_state[end]
    return corrected_state
end