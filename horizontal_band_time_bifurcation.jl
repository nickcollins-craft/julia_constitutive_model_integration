#= This is a code to conduct the bifurcation analysis with all n fixed, one to 1
and two to 0 =#

using SymEngine
using LinearAlgebra
using PolynomialRoots
using Optim
using TensorOperations
using FileIO
include("material_parameters.jl")
include("bifurcation_matrix.jl")
include("element_integration.jl")

#= We start by simplifying and numericising the bifurcation matrix, and applying
the rules constraining the normal vector=#
b_m_s = [symbols("b_ms_s$i$j$t") for i in 1:2*ndim, j in 1:2*ndim, t in 1:n_timesteps+1]
n_rules = [1, 0, 0]
for t in 1:n_timesteps+1
    b_m_s[:, :, t] = coefficient_matrix
    for i in 1:ndim
        for j in 1:ndim
            for k in 1:ndim
                for l in 1:ndim
                    for bm_index1 in 1:2*ndim
                        for bm_index2 in 1:2ndim
                            b_m_s[bm_index1, bm_index2, t] = subs(b_m_s[bm_index1, bm_index2, t], n[i]=>n_rules[i])
                            b_m_s[bm_index1, bm_index2, t] = subs(b_m_s[bm_index1, bm_index2, t], Eepsym[i, j, k, l]=>Eep_res[i, j, k, l, t], Fepsym[i, j, k, l]=>Fep_res[i, j, k, l, t], Kepsym[i, j, k, l]=>Kep_res[i, j, k, l, t], Mepsym[i, j, k, l]=>Mep_res[i, j, k, l, t])
                            b_m_s[bm_index1, bm_index2, t] = subs(b_m_s[bm_index1, bm_index2, t], Irot=>Ires[t])
                        end
                    end
                end
            end
        end
    end
end


# Define a general Levi-Civita function
function generalised_LC(index_vector)::Int64
    value = 1
    for index1 in 1:(length(index_vector)-1)
        for index2 in (index1+1):length(index_vector)
            value = value*sign(index_vector[index2] - index_vector[index1])
        end
    end
    return value
end


#= Get the characteristic polynomial as a function of the time, orientation,
wavelength and value of sq =#
num_mat = [symbols("nm_$t") for t in 1:n_timesteps+1]
fully_subbed = [symbols("fs$i$j") for i in 1:2*ndim, j in 1:2*ndim]
L_mat = [symbols("lm$i$j") for i in 1:2*ndim, j in 1:2*ndim]
U_mat = [symbols("um$i$j") for i in 1:2*ndim, j in 1:2*ndim]
order_6_LC = [generalised_LC([i,j,k,l,m,o]) for i in 1:2*ndim, j in 1:2*ndim, k in 1:2*ndim, l in 1:2*ndim, m in 1:2*ndim, o in 1:2*ndim]
#n_m_t = symbols(:n_m_t)
row_1 = [symbols("rw_$i") for i in 1:2*ndim]
row_2 = [symbols("rw_$i") for i in 1:2*ndim]
row_3 = [symbols("rw_$i") for i in 1:2*ndim]
row_4 = [symbols("rw_$i") for i in 1:2*ndim]
row_5 = [symbols("rw_$i") for i in 1:2*ndim]
row_6 = [symbols("rw_$i") for i in 1:2*ndim]
r1_product = [symbols("r1") for j in 1:2*ndim, k in 1:2*ndim, l in 1:2*ndim, m in 1:2*ndim, o in 1:2*ndim]
r2_product = [symbols("r2") for k in 1:2*ndim, l in 1:2*ndim, m in 1:2*ndim, o in 1:2*ndim]
r3_product = [symbols("r3") for l in 1:2*ndim, m in 1:2*ndim, o in 1:2*ndim]
r4_product = [symbols("r4") for m in 1:2*ndim, o in 1:2*ndim]
r5_product = [symbols("r5") for o in 1:2*ndim]
polycoeff = [symbols("pc$i") for i in 1:2*ndim+1]
function polyΛsq(t)
    fully_subbed = b_m_s[:, :, t]
    #= We use the Levi-Civita tensor formulation of the Leibniz formula to get the
    determinant, but we help it out first by creating some auxiliaries =#
    row_1 = fully_subbed[1, :]
    row_2 = fully_subbed[2, :]
    row_3 = fully_subbed[3, :]
    row_4 = fully_subbed[4, :]
    row_5 = fully_subbed[5, :]
    row_6 = fully_subbed[6, :]
    @tensor begin
        r1_product[j, k, l, m, o] = order_6_LC[i, j , k, l, m, o]*row_1[i]
        r2_product[k, l, m, o] = r1_product[j, k, l, m, o]*row_2[j]
        r3_product[l, m, o] = r2_product[k, l, m, o]*row_3[k]
        r4_product[m, o] = r3_product[l, m, o]*row_4[l]
        r5_product[o] = r4_product[m, o]*row_5[m]
    end
    #= Wrap it up with a loop because TensorOperations doesn't want to give me a
    scalar =#
    n_m_t = symbols(:n_m_t)
    for o = 1:2*ndim
        if o == 1
            n_m_t = r5_product[o]*row_6[o]
        else
            n_m_t = n_m_t + r5_product[o]*row_6[o]
        end
    end
    # Expand it so it's in the form of a + b*s + c*s^2 ...
    expanded_n_m_t = expand(n_m_t)
    return expanded_n_m_t
end

# Run a loop to get all the values of n_m_t
n_m_t = [symbols("nm_$t") for t in 1:n_timesteps+1]
for t in 1:n_timesteps+1
    n_m_t[t] = polyΛsq(t)
end

# Now write the function that returns the roots of the polynomial
function srootfunc(t, Λres)
    # Get the polynomial in clean powers of sq
    subbed_poly = expand(subs(n_m_t[t], Λ=>Λres))
    # Get the polynomial coefficients, using an extremelly inelegant method
    for i in 1:2*ndim+1
        if i == 1
            # Get the s^0 entry
            polycoeff[i] = subs(subbed_poly, sq=>0)
        else
            # Start with all the powers of s
            polycoeff[i] = subbed_poly
            for j in 1:i
                #= Get rid of the entries for the powers lower than what we are
                concerned with =#
                polycoeff[i] = expand(polycoeff[i] - polycoeff[j]*sq^(j-1))
            end
            # Divide out the power that we are concerned with
            polycoeff[i] = expand(polycoeff[i]/(sq^(i-1)))
            # Substitute in sq = 0 to kill the higher powers
            polycoeff[i] = subs(polycoeff[i], sq=>0)
        end
    end
    #= Change the type of the polynomial coefficient from Basic to Complex so it
    can play nicely with the optimisations which require proper numeric types =#
    polycoeffN = lambdify(polycoeff)
    root_vec = roots(polycoeffN)
    return root_vec
end

# Get the largest real part of the roots
function srootmax(t, Λres)
    #= Optim.jl only has minimisation methods, so we need to multiply the maximum
    value by -1 so that we find the minimum of -s, which I think is the same as
    maximising s =#
    max_root = -1.0*maximum(real(srootfunc(t, Λres)))
    return max_root
end


#= Now find the value of Λ that maximises sq (we need to phrase as a
minimisation problem because this is all that's in Optim) =#
function smaxfunc(t, Λmax)
    results = optimize(Λres -> srootmax(t, Λres), dmin, Λmax, Brent(); iterations=optimisation_iterations, abs_tol=optimisation_x_tol)
    minimiser = Optim.minimizer(results)
    return minimiser
end


# Now get the values at each step
function maximisation_run(Λmax)
    Λmax_vec = zeros(Float64, (n_timesteps+1))
    for t in 1:n_timesteps+1
        if t == 1
            Λmax_vec[t] = smaxfunc(t, Λmax)
        else
            Λmax_vec[t] = smaxfunc(t, Λmax)
        end
    end
    return Λmax_vec
end

Λvec = maximisation_run(Λmax_limit)

# Calculate whether we actually have a localisation
function localisation_truth(Λvec)
    truth_result = falses(n_timesteps+1)
    for t in 1:n_timesteps+1
        # Invert the sense because of how our optimisation works
        if srootmax(t, Λvec[:, t]) < 0.0
            truth_result[t] = true
        end
    end
    return truth_result
end

localisation_truth_vec = localisation_truth(nΛvec)

# Save it
save_filenameΛ = save_folder * "Λ_results.jld2"
save(save_filenameΛ, Dict("nΛres"=>nΛvec, "loc_truth_res"=>localisation_truth))
