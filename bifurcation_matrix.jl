#= This is a code to calculate the "bifurcation matrix" of a small-strain Cosserat material. It uses the notation of equations 66, 74--84 of the paper "A Cosserat Breakage Mechanics model for brittle granular media",
 by N.A. Collins-Craft, I. Stefanou, J. Sulem & I. Einav, in the Journal of the Mechanics and Phyics of Solids (2020). 
 Publsihed version: https://www.sciencedirect.com/science/article/pii/S0022509620302106
 Open access version: https://hal.science/hal-03120686v1 
 However, this code retains a more general form of the equations, which can be used for any Cosserat material, without any of the simplifications that are made in the paper.=#

 # Import the necessary packages
 using Einsum
 using SymEngine
 
 
 # Define the symbolic variabls for manipulation
 x = [symbols("x_$i") for i in 1:3]
 n = [symbols("n_$i") for i in 1:3]
 t = symbols(:t)
 β = symbols(:β)
 Λ = symbols(:Λ)
 fake_β = symbols(:fake_β)
 fake_Λ = symbols(:fake_Λ)
 ρ = symbols(:ρ)
 Irot = [symbols("Irot_$i$j") for i in 1:3, j in 1:3]
 U = [symbols("U_$i") for i in 1:3]
 Ω_c = [symbols("Ω_c_$i") for i in 1:3]
 Eepsym = [symbols("Eepsym_$i$j$k$l") for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
 Fepsym = [symbols("Fepsym_$i$j$k$l") for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
 Kepsym = [symbols("Kepsym_$i$j$k$l") for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
 Mepsym = [symbols("Mepsym_$i$j$k$l") for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
 
 
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
     u_sol = U*exp(β*t + (2*π*im/Λ)*sum(x.*n) + fake_β + fake_Λ)
     return u_sol
 end
 
 
 function ω_c(x, t)
     ω_c_sol = Ω_c*exp(β*t + (2*π*im/Λ)*sum(x.*n) + fake_β + fake_Λ)
     return ω_c_sol
 end
 
 # Write some supplementary tensors to declare symbolic variables (the SymEngine.diff is called to avoid any conflicts the the diff function in base, which does something else)
 LevCiv = [LeviCivita(i, j, k) for i in 1:3, j in 1:3, k in 1:3]
 u_first_derivative = [SymEngine.diff(u(x, t)[i], x[j]) for i in 1:3, j in 1:3]
 u_second_derivative = [SymEngine.diff(u(x, t)[i], x[j], x[k]) for i in 1:3, j in 1:3, k in 1:3]
 raw_ω = [ω_c(x, t)[i] for i in 1:3]
 ω_first_derivative = [SymEngine.diff(ω_c(x, t)[i], x[j]) for i in 1:3, j in 1:3]
 ω_second_derivative = [SymEngine.diff(ω_c(x, t)[i], x[j], x[k]) for i in 1:3, j in 1:3, k in 1:3]
 u_time_derivative = [SymEngine.diff(u(x, t)[i], t, t) for i in 1:3]
 ω_time_derivative = [SymEngine.diff(ω_c(x, t)[i], t, t) for i in 1:3]
 solved_sans_exps = [symbols("sse_$i") for i in 1:6]
 
 # Now start building the products of the tensors for the linear momentum balance (these need to be done one-by-one to avoid the einsum function getting confused)
 @einsum LevCiv_times_ω_first_derivative[j, k, l] := LevCiv[j, k, o]*ω_first_derivative[o, l]
 strain_derivatives = u_second_derivative .+ LevCiv_times_ω_first_derivative
 @einsum Eepsym_times_strain_derivatives[i] := Eepsym[i, j, k, l]*strain_derivatives[k, l, j]
 @einsum Fepsym_times_ω_second_derivative[i] := Fepsym[i, j, k, l]*ω_second_derivative[k, l, j]
 # Now get the linear momentum balance
 mombalance = Eepsym_times_strain_derivatives .+ Fepsym_times_ω_second_derivative .- ρ*u_time_derivative
 # Now start building the products of the tensors for the angular momentum balance
 @einsum Kepsym_times_strain_derivatives[i] := Kepsym[i, j, k, l]*strain_derivatives[k, l, j]
 @einsum Mepsym_times_ω_second_derivative[i] := Mepsym[i, j, k, l]*ω_second_derivative[k, l, j]
 @einsum LevCiv_times_ω[l, o] := LevCiv[l, o, r]*raw_ω[r]
 strain = u_first_derivative .+ LevCiv_times_ω
 @einsum Eepsym_times_strain[j, k] := Eepsym[j, k, l, o]*strain[l, o]
 @einsum Fepsym_times_ω_first_derivative[j, k] := Fepsym[j, k, l, o]*ω_first_derivative[l, o]
 stress = Eepsym_times_strain .+ Fepsym_times_ω_first_derivative
 @einsum LevCiv_times_stress[i] := LevCiv[i, j, k]*stress[j, k]
 @einsum Irot_times_ω_time_derivative[i] := Irot[i, j]*ω_time_derivative[j]
 # Now get the angular momentum balance
 angmombalance = Kepsym_times_strain_derivatives .+ Mepsym_times_ω_second_derivative .- LevCiv_times_stress .- Irot_times_ω_time_derivative
 
 # Get all the balanced equations and then substitute the fake variables to cancel the exponentials
 balanced_equations = vcat(mombalance, angmombalance)
 for i in 1:6
     solved_sans_exps[i] = subs(balanced_equations[i], fake_β=>-1*β*t, fake_Λ=>(-2*π*im/Λ)*sum(x.*n))
 end
 
 # Get the coefficient matrix
 coefficient_matrix = [symbols("res$i$j") for i in 1:6, j in 1:6]
 β_bar = symbols(:β_bar)
 in_terms_of = vcat(U, Ω_c)
 for i in 1:6
     for j in 1:6
         # Get the terms that give the U_jth item in the ith momentum equation and reduce the power of s
         coefficient_matrix[i, j] = subs(solved_sans_exps[i], in_terms_of[j]=>1, β^2=>β_bar)
         for k in 1:6
             if j != k
                 # Set to zero all the other ones
                 coefficient_matrix[i, j] = subs(coefficient_matrix[i, j], in_terms_of[k]=>0)
             end
         end
     end
 end