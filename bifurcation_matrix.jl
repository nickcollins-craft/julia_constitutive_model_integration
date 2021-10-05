#This is a code to create the bifurcation matrix

using TensorOperations
using SymEngine
include("material_parameters.jl")

# Define the symbolic variabls for manipulation
x = [symbols("x_$i") for i in 1:ndim]
n = [symbols("n_$i") for i in 1:ndim]
t = symbols(:t)
s = symbols(:s)
Λ = symbols(:Λ)
fake_s = symbols(:fake_s)
fake_Λ = symbols(:fake_Λ)
Irot = symbols(:Irot)
U = [symbols("U_$i") for i in 1:ndim]
if ndim == 2
    Ω_c = symbols(Ω_c)
else
    Ω_c = [symbols("Ω_c_$i") for i in 1:ndim]
end
Eepsym = [symbols("Eepsym_$i$j$k$l") for i in 1:ndim, j in 1:ndim, k in 1:ndim, l in 1:ndim]
Fepsym = [symbols("Fepsym_$i$j$k$l") for i in 1:ndim, j in 1:ndim, k in 1:ndim, l in 1:ndim]
Kepsym = [symbols("Kepsym_$i$j$k$l") for i in 1:ndim, j in 1:ndim, k in 1:ndim, l in 1:ndim]
Mepsym = [symbols("Mepsym_$i$j$k$l") for i in 1:ndim, j in 1:ndim, k in 1:ndim, l in 1:ndim]

# Levi-Civita symbol
function LeviCivita(i, j, k)::Int
    if (i, j, k) == (1, 2, 3) || (i, j, k) == (2, 3, 1) || (i, j, k) == (3, 1, 2)
        return 1
    elseif (i, j, k) == (3, 2, 1) || (i, j, k) == (1, 3, 2) || (i, j, k) == (2, 1, 3)
        return -1
    else
        return 0
    end
end


# Solution rules with fake variables so we can cancel the exponentials
function u(x, t)
    u_sol = U*exp(s*t + (2*π*im/Λ)*sum(x.*n) + fake_s + fake_Λ)
    return u_sol
end


function ω_c(x, t)
    ω_c_sol = Ω_c*exp(s*t + (2*π*im/Λ)*sum(x.*n) + fake_s + fake_Λ)
    return ω_c_sol
end

#= Write some supplementary tensors to declare symbolic variables and to aid
TensorOperations not be overwhelmed by too much maths at once=#
LevCiv = [LeviCivita(i, j, k) for i in 1:ndim, j in 1:ndim, k in 1:ndim]
mombalance = [symbols("mom_$i") for i in 1:ndim]
angmombalance = [symbols("angmom_$i") for i in 1:ndim]
u_first_derivative = [diff(u(x, t)[i], x[j]) for i in 1:ndim, j in 1:ndim]
u_second_derivative = [diff(u(x, t)[i], x[j], x[k]) for i in 1:ndim, j in 1:ndim, k in 1:ndim]
raw_ω = [ω_c(x, t)[i] for i in 1:ndim]
ω_first_derivative = [diff(ω_c(x, t)[i], x[j]) for i in 1:ndim, j in 1:ndim]
ω_second_derivative = [diff(ω_c(x, t)[i], x[j], x[k]) for i in 1:ndim, j in 1:ndim, k in 1:ndim]
u_time_derivative = [diff(u(x, t)[i], t, t) for i in 1:ndim]
ω_time_derivative = [diff(ω_c(x, t)[i], t, t) for i in 1:ndim]
first_product = [symbols("fp_$i$j$k") for i in 1:ndim, j in 1:ndim, k in 1:ndim]
second_product = [symbols("sp_$i") for i in 1:ndim]
third_product = [symbols("fop_$i") for i in 1:ndim]
fourth_product = [symbols("tp_$i$j") for i in 1:ndim, j in 1:ndim]
fifth_product = [symbols("fip_$i$j") for i in 1:ndim, j in 1:ndim]
sixth_product = [symbols("sip_$i$j") for i in 1:ndim, j in 1:ndim]
first_sum = fifth_product + sixth_product
seventh_product = [symbols("sep_$i") for i in 1:ndim]
solved_sans_exps = [symbols("sse_$i") for i in 1:2*ndim]


# Write the momentum balance equations (just in 3D for the time being)
@tensor function linear_momentum_balance(x, t)
    first_product[k, l, j] = u_second_derivative[k, l, j] + LevCiv[k, l, m]*ω_first_derivative[m, j]
    second_product[i] = Eepsym[i, j, k, l]*first_product[k, l, j]
    mombalance = second_product - ρ*u_time_derivative
    return mombalance
end


@tensor function angular_momentum_balance(x, t)
    first_product[k, l, j] = u_second_derivative[k, l, j] + LevCiv[k, l, m]*ω_first_derivative[m, j]
    second_product[i] = Kepsym[i, j, k, l]*first_product[k, l, j]
    third_product[i] = Mepsym[i, j, k, l]*ω_second_derivative[k, l, j]
    fourth_product[l, m] = u_first_derivative[l, m] + LevCiv[l, m, o]*raw_ω[o]
    fifth_product[j, k] = Eepsym[j, k, l, m]*fourth_product[l, m]
    sixth_product[j, k] = Fepsym[j, k, l, m]*ω_first_derivative[l, m]
    first_sum = fifth_product + sixth_product
    seventh_product[i] = LevCiv[i, j, k]*first_sum[j, k]
    angmombalance = second_product + third_product - seventh_product - Irot*ω_time_derivative
    return angmombalance
end


# Get rid of the exponentials
function balance_equations(x, t)
    balanced = vcat(linear_momentum_balance(x, t), angular_momentum_balance(x, t))
    return balanced
end
for i in 1:2*ndim
    solved_sans_exps[i] = subs(balance_equations(x, t)[i], fake_s=>-1*s*t, fake_Λ=>(-2*π*im/Λ)*sum(x.*n))
end


# Get the coefficient matrix
coefficient_matrix = [symbols("res$i$j") for i in 1:2*ndim, j in 1:2*ndim]
sq = symbols(:sq)
in_terms_of = vcat(U, Ω_c)
for i in 1:2*ndim
    for j in 1:2*ndim
        #= Get the terms that give the U_jth item in the ith momentum equation
        and reduce the power of s=#
        coefficient_matrix[i, j] = subs(solved_sans_exps[i], in_terms_of[j]=>1, s^2=>sq)
        for k in 1:2*ndim
            if j != k
                # Set to zero all the other ones
                coefficient_matrix[i, j] = subs(coefficient_matrix[i, j], in_terms_of[k]=>0)
            end
        end
    end
end
