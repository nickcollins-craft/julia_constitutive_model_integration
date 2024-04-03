#= This is a code to conduct the bifurcation analysis of a small-strain Cosserat material with all n fixed, n1 to 1, and n2 and n3 to 0. This represents localisation into a horizontal band. =#

# Import the necessary packages
using SymEngine
using PolynomialRoots
using NLopt
using FileIO
using Einsum
using ProgressBars


# Import the material parameters and the bifurcation matrix
include("material_parameters.jl")
include("bifurcation_matrix.jl")

# Import the results of the original integration
load_results = load(save_folder * "integration_results.jld2")
Eep_res = load_results["Eepres"]
Fep_res = load_results["Fepres"]
Kep_res = load_results["Kepres"]
Mep_res = load_results["Mepres"]
ρ_res = load_results["ρres"]
I_res = load_results["Ires"]

# We start by simplifying and numericising the bifurcation matrix, and applying the rules constraining the normal vector
b_m_s = [symbols("b_ms_s$i$j$t") for i in 1:6, j in 1:6, t in 1:n_timesteps + 1]
n_rules = [1, 0, 0]
coefficient_matrix_with_n_rules = coefficient_matrix
for i in 1:6
    for j in 1:6
        coefficient_matrix_with_n_rules[i, j] = subs(coefficient_matrix[i, j], n[1]=>n_rules[1], n[2]=>n_rules[2], n[3]=>n_rules[3])
    end
end
println("Constructing the numerical bifurcation matrix:")
for t in ProgressBar(1:n_timesteps + 1)
    b_m_s[:, :, t] = coefficient_matrix_with_n_rules
    for i in 1:3
        for j in 1:3
            for k in 1:3
                for l in 1:3
                    for bm_index1 in 1:6
                        for bm_index2 in 1:6
                            b_m_s[bm_index1, bm_index2, t] = subs(b_m_s[bm_index1, bm_index2, t], Eepsym[i, j, k, l]=>Eep_res[i, j, k, l, t], Fepsym[i, j, k, l]=>Fep_res[i, j, k, l, t], Kepsym[i, j, k, l]=>Kep_res[i, j, k, l, t], Mepsym[i, j, k, l]=>Mep_res[i, j, k, l, t], ρ=>ρ_res[t], Irot[i, j]=>I_res[i, j, t])
                        end
                    end
                end
            end
        end
    end
end
println("Finished constructing the numerical bifurcation matrix, calculating the characteristic polynomial:")


# Define a general Levi-Civita function
function generalised_LC(index_vector)::Int64
    value = 1
    for index1 in 1:(length(index_vector) - 1)
        for index2 in (index1 + 1):length(index_vector)
            value = value*sign(index_vector[index2] - index_vector[index1])
        end
    end
    return value
end


# Get the characteristic polynomial as a function of the time, orientation, wavelength and value of β_bar
num_mat = [symbols("nm_$t") for t in 1:n_timesteps + 1]
fully_subbed = [symbols("fs$i$j") for i in 1:6, j in 1:6]
L_mat = [symbols("lm$i$j") for i in 1:6, j in 1:6]
U_mat = [symbols("um$i$j") for i in 1:6, j in 1:6]
order_6_LC = [generalised_LC([i, j, k, l, m, o]) for i in 1:6, j in 1:6, k in 1:6, l in 1:6, m in 1:6, o in 1:6]
row_1 = [symbols("rw_$i") for i in 1:6]
row_2 = [symbols("rw_$i") for i in 1:6]
row_3 = [symbols("rw_$i") for i in 1:6]
row_4 = [symbols("rw_$i") for i in 1:6]
row_5 = [symbols("rw_$i") for i in 1:6]
row_6 = [symbols("rw_$i") for i in 1:6]
polycoeff = [symbols("pc$i") for i in 1:7]
function polyΛβ_bar(t)
    fully_subbed = b_m_s[:, :, t]
    # We use the Levi-Civita tensor formulation of the Leibniz formula to get the determinant, but we help it out first by creating some auxiliaries
    row_1 = fully_subbed[1, :]
    row_2 = fully_subbed[2, :]
    row_3 = fully_subbed[3, :]
    row_4 = fully_subbed[4, :]
    row_5 = fully_subbed[5, :]
    row_6 = fully_subbed[6, :]
    @einsum r1_product[j, k, l, m, o] := order_6_LC[i, j , k, l, m, o]*row_1[i]
    @einsum r2_product[k, l, m, o] := r1_product[j, k, l, m, o]*row_2[j]
    @einsum r3_product[l, m, o] := r2_product[k, l, m, o]*row_3[k]
    @einsum r4_product[m, o] := r3_product[l, m, o]*row_4[l]
    @einsum r5_product[o] := r4_product[m, o]*row_5[m]
    @einsum n_m_t := r5_product[o]*row_6[o]
    # Expand it so it's in the form of a + b*s + c*s^2 ...
    expanded_n_m_t = expand(n_m_t)
    return expanded_n_m_t
end


# Run a loop to get all the values of n_m_t
n_m_t = [symbols("nm_$t") for t in 1:n_timesteps + 1]
for t in ProgressBar(1:n_timesteps + 1)
    n_m_t[t] = polyΛβ_bar(t)
end
println("Finished calculating the characteristic polynomial, now finding the fastest growing wavelength and saving:")


# Now write the function that returns the roots of the polynomial
function srootfunc(t, Λres)
    # Get the polynomial in clean powers of β_bar
    subbed_poly = expand(subs(n_m_t[t], Λ=>Λres))
    # Get the polynomial coefficients, using an extremelly inelegant method
    for i in 1:7
        if i == 1
            # Get the s^0 entry
            polycoeff[i] = subs(subbed_poly, β_bar=>0)
        else
            # Start with all the powers of s
            polycoeff[i] = subbed_poly
            for j in 1:i
                # Get rid of the entries for the powers lower than what we are concerned with
                polycoeff[i] = expand(polycoeff[i] - polycoeff[j]*β_bar^(j - 1))
            end
            # Divide out the power that we are concerned with
            polycoeff[i] = expand(polycoeff[i]/(β_bar^(i - 1)))
            # Substitute in β_bar = 0 to kill the higher powers
            polycoeff[i] = subs(polycoeff[i], β_bar=>0)
        end
    end
    # Change the type of the polynomial coefficient from Basic to Complex so it can play nicely with the optimisations which require proper numeric types
    polycoeffN = lambdify(polycoeff)
    # Enforce a type check so that it doesn't try and get the roots of a symbolic variable
    if typeof(polycoeffN) == Vector{ComplexF64}
        root_vec = roots(polycoeffN)
    else
        # In this case, enforce a very negative value so that the optimisation doesn't get confused
        root_vec = -1E30*ones(ComplexF64, 7)
    end
    return root_vec
end


# Get the largest real part of the roots
function srootmax(t, Λres_vec)
    # Extract the wavelength, which has to be given as a vector by the optimisation solver
    Λres = Λres_vec[1]
    # NLopt only has minimisation methods, so we need to multiply the maximum value by -1 so that we find the minimum of -s
    max_root = -1.0*maximum(real(srootfunc(t, Λres)))
    return max_root
end


# Now find the value of Λ that maximises β_bar (we need to phrase as a minimisation problem because that's how NLopt works)
function smaxfunc(t, Λmax, Λguess)
    # Declare the optimisation object, the algorithm and tolerance
    opt = Opt(:LN_NELDERMEAD, 1)
    # Declare the function to optimise, using an anonymous function to turn it into a function of one vector (grad is dummy to match NLopt syntax)
    opt.min_objective = (Λres, grad) -> srootmax(t, Λres)
    opt.ftol_abs = 1e-11
    opt.xtol_abs = optimisation_x_tol
    opt.maxeval = optimisation_iterations
    opt.lower_bounds = [dmin]
    opt.upper_bounds = [Λmax]
    # Optimise and get the result (returning just the first entry in the minimiser, which is returned as a vector)
    (minf, minimiser, ret) = optimize(opt, [Λguess])
    return minimiser[1]
end


# Now get the values at each step
function maximisation_run(Λmax)
    Λmax_vec = zeros(Float64, (n_timesteps + 1))
    for t in ProgressBar(1:n_timesteps + 1)
        if t == 1
            Λmax_vec[t] = smaxfunc(t, Λmax, Λmax)
        else
            Λmax_vec[t] = smaxfunc(t, Λmax, Λmax_vec[t - 1])
        end
    end
    return Λmax_vec
end

Λvec = maximisation_run(Λmax_limit)

# Calculate whether we actually have a localisation
function localisation_truth_func(Λvec)
    truth_result = falses(n_timesteps + 1)
    for t in 1:n_timesteps + 1
        # Invert the sense because of how our optimisation works
        if srootmax(t, Λvec[t]) < 0.0
            truth_result[t] = true
        end
    end
    return truth_result
end

localisation_truth_vec = localisation_truth_func(Λvec)

# Save it
save_filename_Λ = save_folder * "localisation_results.jld2"
save(save_filename_Λ, Dict("Λres"=>Λvec, "loc_truth_res"=>localisation_truth_vec))
