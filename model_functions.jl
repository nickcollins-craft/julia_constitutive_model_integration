# A file to hold the model functions

using Base
using TensorOperations
using LinearAlgebra
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
function mac(x::Float64)::Float64
    if x >= 0
        y = x
    else
        y = 0
    end
    return y
end


# Fill out the elastic stiffness tensors
Celastic = zeros(Float64, (ndim, ndim, ndim, ndim))
Delastic = zeros(Float64, (ndim, ndim, ndim, ndim))
for i = 1:ndim
    for j = 1:ndim
        for k = 1:ndim
            for l = 1:ndim
                Celastic[i, j, k, l] = (K - 2*G/3)*KroneckerDelta(i, j)*KroneckerDelta(k, l) + (G + Gc)*KroneckerDelta(i, k)*KroneckerDelta(j, l) + (G - Gc)*KroneckerDelta(i, l)*KroneckerDelta(j, k)
                Delastic[i, j, k, l] = (L - 2*H/3)*KroneckerDelta(i, j)*KroneckerDelta(k, l) + (H + Hc)*KroneckerDelta(i, k)*KroneckerDelta(j, l) + (H - Hc)*KroneckerDelta(i, l)*KroneckerDelta(j, k)
            end
        end
    end
end


# τ function
@tensor function τ(B::Float64, γ_e::Array{Float64, 2})::Array{Float64, 2}
    τval = zeros(Float64, (ndim, ndim))
    τval[i, j] = (1-θ_γ*B)*Celastic[i, j, k, l]*γ_e[k, l]
    return τval
end


# μ function
@tensor function μ(B::Float64, κ_e::Array{Float64, 2})::Array{Float64, 2}
    μval = zeros(Float64, (ndim, ndim))
    μval[i, j] = (x_r^2)*(1-θ_κ*B)*Delastic[i, j, k, l]*κ_e[k, l]
    return μval
end


# Breakage energy
@tensor function EB(γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
        eb = (θ_γ/2)*Celastic[i, j, k, l]*γ_e[i, j]*γ_e[k, l] + (x_r^2)*(θ_κ/2)*Delastic[i, j, k, l]*κ_e[i, j]*κ_e[k, l]
    return eb
end


# Elastic length scale
function ell(B::Float64)::Float64
    # For the sake of convenience, check positive B here
    if B < 0.0
        B_use = 0.0
    else
        B_use = B
    end
    ell = x_r*sqrt(1-θ_κ*B_use)
    return ell
end


# Isotropic presure
@tensor function p(B::Float64, γ_e::Array{Float64, 2})::Float64
        p = (1/ndim)*τ(B, γ_e)[k, k]
    return p
end


# Isotropic pressure for μ
@tensor function momp(B::Float64, κ_e::Array{Float64, 2})::Float64
    momp = (1/ndim)*μ(B, κ_e)[k, k]
    return momp
end


# Deviator stress
function s_dev(B::Float64, γ_e::Array{Float64, 2})::Array{Float64, 2}
    pmat = p(B, γ_e)*Matrix(1.0I, ndim, ndim)
    s_dev = τ(B, γ_e) - pmat
    return s_dev
end


# μ deviator stress
function m_dev(B::Float64, κ_e::Array{Float64, 2})::Array{Float64, 2} # Fewer types so it doesn't get upset
    mompmat = momp(B, κ_e)*Matrix(1.0I, ndim, ndim)
    m_dev = μ(B, κ_e) - mompmat
    return m_dev
end


# Second deviatoric stress invariant
@tensor function q(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    qsquare = h1*s_dev(B, γ_e)[i, j]*s_dev(B, γ_e)[i, j] + h2*s_dev(B, γ_e)[i, j]*s_dev(B, γ_e)[j, i] + (1/(ell(B))^2)*(h3*m_dev(B, κ_e)[i, j]*m_dev(B, κ_e)[i, j] + h4*m_dev(B, κ_e)[i, j]*m_dev(B, κ_e)[j, i])
    q = sqrt(qsquare)
    return q
end


# Yield surface
function ymix(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    ymix = ((EB(γ_e, κ_e)*((1-B)^2))/Ec) + ((q(B, γ_e, κ_e)/(M*p(B, γ_e)))^2) - 1
    return ymix
end


# functions for the flow rules (minus λ)
@tensor function dyΦdτ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    sbit = s_dev(B, γ_e)
    sbit_T = transpose(sbit)
    deviatorpart = 2*(h1*sbit + h2*sbit_T)/((M*p(B, γ_e))^2)
    tracepart = ((2*EB(γ_e, κ_e)*((1-B)^2)*(sind(ω))^2)/(3*p(B, γ_e)*Ec))*Matrix(1.0I, ndim, ndim)
    dyΦdτ = tracepart + deviatorpart
    return dyΦdτ
end


@tensor function dyΦdμ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    dyΦdμ = zeros(Float64, (ndim, ndim))
    dyΦdμ[i, j] = (2*(h3*m_dev(B, κ_e)[i, j]+h4*m_dev(B, κ_e)[j, i]))/((ell(B)*M*p(B, γ_e))^2)
    return dyΦdμ
end


function dyΦdEB(B::Float64)::Float64
    dyΦdEB = (2*((1-B)^2)*(cosd(ω))^2)/Ec
    return dyΦdEB
end


# Now we function a series of helper functions on the way to λ
function dydτ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    sbit = s_dev(B, γ_e)
    sbit_T = transpose(sbit)
    deviatorpart = 2*(h1*sbit + h2*sbit_T)/((M*p(B, γ_e))^2)
    tracepart = ((-2*(q(B, γ_e, κ_e)^2))/(3*(M^2)*(p(B, γ_e)^3)))*Matrix(1.0I, ndim, ndim)
    dydτ = tracepart + deviatorpart
    return dydτ
end


@tensor function dydμ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    dydμ = zeros(Float64, (ndim, ndim))
    dydμ[i, j] = (2/((ell(B)*M*p(B, γ_e))^2))*(h3*m_dev(B, κ_e)[i, j] + h4*m_dev(B, κ_e)[j, i])
    return dydμ
end


function dydEB(B::Float64)::Float64
    dydEB = ((1-B)^2)/Ec
    return dydEB
end


function dydB(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    dydB = (-2*(1-B)*EB(γ_e, κ_e))/Ec
    return dydB
end


function dτdγ(B::Float64)::Array{Float64, 4} # No type because it gets upset
    dτdγ = (1-θ_γ*B)*Celastic
    return dτdγ
end


@tensor function dτdB(γ_e::Array{Float64, 2})::Array{Float64, 2}
    dτdB = zeros(Float64, (ndim, ndim))
    dτdB[i, j] = -θ_γ*Celastic[i, j, k, l]*γ_e[k, l]
    return dτdB
end


function dμdκ(B::Float64)::Array{Float64, 4}
    dμdκ = (x_r^2)*(1-θ_κ*B)*Delastic
    return dμdκ
end


@tensor function dμdB(κ_e::Array{Float64, 2})::Array{Float64, 2}
    dμdB = zeros(Float64, (ndim, ndim))
    dμdB[i, j] = -θ_κ*(x_r^2)*Delastic[i, j, k, l]*κ_e[k, l]
    return dμdB
end


@tensor function dEBdγ(γ_e::Array{Float64, 2})::Array{Float64, 2}
    dEBdγ = zeros(Float64, (ndim, ndim))
    dEBdγ[i, j] = θ_γ*Celastic[i, j, k, l]*γ_e[k, l]
    return dEBdγ
end


@tensor function dEBdκ(κ_e::Array{Float64, 2})::Array{Float64, 2}
    dEBdκ = zeros(Float64, (ndim, ndim))
    dEBdκ[i, j] = (x_r^2)*θ_κ*Delastic[i, j, k, l]*κ_e[k, l]
    return dEBdκ
end


@tensor function γstresspart(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    γstresspart = zeros(Float64, (ndim, ndim))
    γstresspart[k, l] = dydτ(B, γ_e, κ_e)[i, j]*dτdγ(B)[i, j, k, l]
    return γstresspart
end


@tensor function γenergypart(B::Float64, γ_e::Array{Float64, 2})::Array{Float64, 2}
    γenergypart = zeros(Float64, (ndim, ndim))
    γenergypart[k, l] = dydEB(B)*dEBdγ(γ_e)[k, l]
    return γenergypart
end


function γpart(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    γpart = γstresspart(B, γ_e, κ_e) + γenergypart(B, γ_e)
    return γpart
end


@tensor function κcouplepart(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    κcouplepart = zeros(Float64, (ndim, ndim))
    κcouplepart[k, l] = dydμ(B, γ_e, κ_e)[i, j]*dμdκ(B)[i, j, k, l]
    return κcouplepart
end


function κenergypart(B::Float64, κ_e::Array{Float64, 2})::Array{Float64, 2}
    κenergypart = dydEB(B)*dEBdκ(κ_e)
    return κenergypart
end


function κpart(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    κpart = κcouplepart(B, γ_e, κ_e) + κenergypart(B, κ_e)
    return κpart
end


@tensor function AIJ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    AIJ = zeros(Float64, (ndim, ndim))
    AIJ[i, j] = dτdγ(B)[i, j, k, l]*dyΦdτ(B, γ_e, κ_e)[k, l]
    return AIJ
end


function BIJ(B::Float64, γ_e::Array{Float64, 2})::Array{Float64, 2}
    BIJ = dτdB(γ_e)*dyΦdEB(B)
    return BIJ
end


@tensor function stresspart(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    stresspart = dydτ(B, γ_e, κ_e)[i, j]*(AIJ(B, γ_e, κ_e)[i, j] - BIJ(B, γ_e)[i, j])
    return stresspart
end


@tensor function CIJ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    CIJ = zeros(Float64, (ndim, ndim))
    CIJ[i, j] = dμdκ(B)[i, j, k, l]*dyΦdμ(B, γ_e, κ_e)[k, l]
    return CIJ
end


function DIJ(B::Float64, κ_e::Array{Float64, 2})::Array{Float64, 2}
    DIJ = dμdB(κ_e)*dyΦdEB(B)
    return DIJ
end


@tensor function couplepart(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    couplepart = dydμ(B, γ_e, κ_e)[i, j]*(CIJ(B, γ_e, κ_e)[i, j] - DIJ(B, κ_e)[i, j])
    return couplepart
end


@tensor function EIJ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    EIJ = dEBdγ(γ_e)[i, j]*dyΦdτ(B, γ_e, κ_e)[i, j]
    return EIJ
end


@tensor function FIJ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    FIJ = dEBdκ(κ_e)[i, j]*dyΦdμ(B, γ_e, κ_e)[i, j]
    return FIJ
end


function energypart(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    energypart = dydEB(B)*(EIJ(B, γ_e, κ_e)+FIJ(B, γ_e, κ_e))
    return energypart
end


function breakagepart(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    breakagepart = dydB(B, γ_e, κ_e)*dyΦdEB(B)
    return breakagepart
end


function λdenom(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Float64
    λdenom = stresspart(B, γ_e, κ_e) + couplepart(B, γ_e, κ_e) + energypart(B, γ_e, κ_e) - breakagepart(B, γ_e, κ_e)
    return λdenom
end


function λγ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    λγ= γpart(B, γ_e, κ_e)/λdenom(B, γ_e, κ_e)
    return λγ
end


function λκ(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 2}
    λκ = κpart(B, γ_e, κ_e)/λdenom(B, γ_e, κ_e)
    return λκ
end


# Finally we function the plastic multiplier λ
@tensor function λfunc(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2}, γtotdot::Array{Float64, 2}, κtotdot::Array{Float64, 2})::Float64
    λfunc = λγ(B, γ_e, κ_e)[i, j]*γtotdot[i, j] + λκ(B, γ_e, κ_e)[i, j]*κtotdot[i, j]
    return λfunc
end


# We can now write the flow rules explicitly
function Bdot(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2}, γtotdot::Array{Float64, 2}, κtotdot::Array{Float64, 2})::Float64
    bdot = mac(λfunc(B, γ_e, κ_e, γtotdot, κtotdot))*((1+ymix(B, γ_e, κ_e))^ξ)*dyΦdEB(B)
    return bdot
end


function γpdot(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2}, γtotdot::Array{Float64, 2}, κtotdot::Array{Float64, 2})::Array{Float64, 2}
    γpdot = mac(λfunc(B, γ_e, κ_e, γtotdot, κtotdot))*((1+ymix(B, γ_e, κ_e))^ξ)*dyΦdτ(B, γ_e, κ_e)
    return γpdot
end


function κpdot(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2}, γtotdot::Array{Float64, 2}, κtotdot::Array{Float64, 2})::Array{Float64, 2}
    κpdot = mac(λfunc(B, γ_e, κ_e, γtotdot, κtotdot))*((1+ymix(B, γ_e, κ_e))^ξ)*dyΦdμ(B, γ_e, κ_e)
    return κpdot
end


# Now we write the incremental tensors in h²plasticity form
@tensor function Eep(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 4}
    Eep = zeros(Float64, (ndim, ndim, ndim, ndim))
    Eep[i, j, k, l] = dτdγ(B)[i, j, k, l] - ((1+ymix(B, γ_e, κ_e))^ξ)*(AIJ(B, γ_e, κ_e)[i, j] - BIJ(B, γ_e)[i, j])*λγ(B, γ_e, κ_e)[k, l]
    return Eep
end


@tensor function Fep(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 4}
    Fep = zeros(Float64, (ndim, ndim, ndim, ndim))
    Fep[i, j, k, l] = -((1+ymix(B, γ_e, κ_e))^ξ)*(AIJ(B, γ_e, κ_e)[i, j] - BIJ(B, γ_e)[i, j])*λκ(B, γ_e, κ_e)[k, l]
    return Fep
end


@tensor function Kep(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 4}
    Kep = zeros(Float64, (ndim, ndim, ndim, ndim))
    Kep[i, j, k, l] = -((1+ymix(B, γ_e, κ_e))^ξ)*(CIJ(B, γ_e, κ_e)[i, j] - DIJ(B, κ_e)[i, j])*λγ(B, γ_e, κ_e)[k, l]
    return Kep
end


@tensor function Mep(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2})::Array{Float64, 4}
    Mep = zeros(Float64, (ndim, ndim, ndim, ndim))
    Mep[i, j, k, l] = dμdκ(B)[i, j, k, l] - ((1+ymix(B, γ_e, κ_e))^ξ)*(CIJ(B, γ_e, κ_e)[i, j] - DIJ(B, κ_e)[i, j])*λκ(B, γ_e, κ_e)[k, l]
    return Mep
end


# It is convenient to write the evolution laws for the state variables other than B
function γedot(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2}, γtotdot::Array{Float64, 2}, κtotdot::Array{Float64, 2})::Array{Float64, 2}
    γedot = γtotdot - γpdot(B, γ_e, κ_e, γtotdot, κtotdot)
    return γedot
end


function κedot(B::Float64, γ_e::Array{Float64, 2}, κ_e::Array{Float64, 2}, γtotdot::Array{Float64, 2}, κtotdot::Array{Float64, 2})::Array{Float64, 2}
    κedot = κtotdot - κpdot(B, γ_e, κ_e, γtotdot, κtotdot)
    return κedot
end


# We write a function returning the rotational moment of inertia
function I_rotational(B::Float64)::Float64
    I_val = (π/60)*(1-θ_I*B)*ρ*(x_r^5)
    return I_val
end


# Now we write a forward Runge-Kutta function
function RK4(B, γ_e, κ_e, γtotdot, κtotdot)
    k1γ = γedot(B, γ_e, κ_e, γtotdot, κtotdot)
    k1κ = κedot(B, γ_e, κ_e, γtotdot, κtotdot)
    k1B = Bdot(B, γ_e, κ_e, γtotdot, κtotdot)

    k2γ = γedot(B + (h/2)*k1B, γ_e + (h/2)*k1γ, κ_e + (h/2)*k1κ, γtotdot, κtotdot)
    k2κ = κedot(B + (h/2)*k1B, γ_e + (h/2)*k1γ, κ_e + (h/2)*k1κ, γtotdot, κtotdot)
    k2B = Bdot(B + (h/2)*k1B, γ_e + (h/2)*k1γ, κ_e + (h/2)*k1κ, γtotdot, κtotdot)

    k3γ = γedot(B + (h/2)*k2B, γ_e + (h/2)*k2γ, κ_e + (h/2)*k2κ, γtotdot, κtotdot)
    k3κ = κedot(B + (h/2)*k2B, γ_e + (h/2)*k2γ, κ_e + (h/2)*k2κ, γtotdot, κtotdot)
    k3B = Bdot(B + (h/2)*k2B, γ_e + (h/2)*k2γ, κ_e + (h/2)*k2κ, γtotdot, κtotdot)

    k4γ = γedot(B + h*k3B, γ_e + h*k3γ, κ_e + h*k3κ, γtotdot, κtotdot)
    k4κ = κedot(B + h*k3B, γ_e + h*k3γ, κ_e + h*k3κ, γtotdot, κtotdot)
    k4B = Bdot(B + h*k3B, γ_e + h*k3γ, κ_e + h*k3κ, γtotdot, κtotdot)

    γ_e_inc = (h/6)*(k1γ + 2*k2γ + 2*k3γ + k4γ)
    κ_e_inc = (h/6)*(k1κ + 2*k2κ + 2*k3κ + k4κ)
    B_inc = (h/6)*(k1B + 2*k2B + 2*k3B + k4B)
    return [B_inc, γ_e_inc, κ_e_inc]
end


# We write a function that is optimised for DifferentialEquations
function RK4_diff_eq!(du, u, params, t)
    B = u[1]
    γ_e = reshape(u[2:10], (ndim, ndim))
    κ_e = reshape(u[11:19], (ndim, ndim))
    γtotdot = reshape(params[1:9], (ndim, ndim))
    κtotdot = reshape(params[10:18], (ndim, ndim))

    k1γ = γedot(B, γ_e, κ_e, γtotdot, κtotdot)
    k1κ = κedot(B, γ_e, κ_e, γtotdot, κtotdot)
    k1B = Bdot(B, γ_e, κ_e, γtotdot, κtotdot)

    k2γ = γedot(B + (h/2)*k1B, γ_e + (h/2)*k1γ, κ_e + (h/2)*k1κ, γtotdot, κtotdot)
    k2κ = κedot(B + (h/2)*k1B, γ_e + (h/2)*k1γ, κ_e + (h/2)*k1κ, γtotdot, κtotdot)
    k2B = Bdot(B + (h/2)*k1B, γ_e + (h/2)*k1γ, κ_e + (h/2)*k1κ, γtotdot, κtotdot)

    k3γ = γedot(B + (h/2)*k2B, γ_e + (h/2)*k2γ, κ_e + (h/2)*k2κ, γtotdot, κtotdot)
    k3κ = κedot(B + (h/2)*k2B, γ_e + (h/2)*k2γ, κ_e + (h/2)*k2κ, γtotdot, κtotdot)
    k3B = Bdot(B + (h/2)*k2B, γ_e + (h/2)*k2γ, κ_e + (h/2)*k2κ, γtotdot, κtotdot)

    k4γ = γedot(B + h*k3B, γ_e + h*k3γ, κ_e + h*k3κ, γtotdot, κtotdot)
    k4κ = κedot(B + h*k3B, γ_e + h*k3γ, κ_e + h*k3κ, γtotdot, κtotdot)
    k4B = Bdot(B + h*k3B, γ_e + h*k3γ, κ_e + h*k3κ, γtotdot, κtotdot)

    du[1] = (h/6)*(k1B + 2*k2B + 2*k3B + k4B)
    du[2:10] = reshape((h/6)*(k1γ + 2*k2γ + 2*k3γ + k4γ), (ndim^2))
    du[11:19] = reshape((h/6)*(k1κ + 2*k2κ + 2*k3κ + k4κ), (ndim^2))
end


# We write a function optimised for DifferentialEquations that is just the model
function evolution_laws!(du, u, params, t)
    B = u[1]
    γ_e = reshape(u[2:10], (ndim, ndim))
    κ_e = reshape(u[11:19], (ndim, ndim))
    γtotdot = reshape(params[1:9], (ndim, ndim))
    κtotdot = reshape(params[10:18], (ndim, ndim))

    du[1] = Bdot(B, γ_e, κ_e, γtotdot, κtotdot)
    du[2:10] = γedot(B, γ_e, κ_e, γtotdot, κtotdot)
    du[11:19] = κedot(B, γ_e, κ_e, γtotdot, κtotdot)
end


# Write a function that includes the control parameters
function evolution_laws_control!(du, u, control, t)
    # Get the state variables
    B = u[1]
    γ_e = reshape(u[2:10], (ndim, ndim))
    κ_e = reshape(u[11:19], (ndim, ndim))

    # Get the initial values of the generalised stress and strain rates
    @views generalised_strain_rate = u[20:37]
    @views generalised_stress_rate = u[38:55]

    # Calculate the incremental stiffness as a matrix
    incremental_stiffness = [[reshape(Eep(B, γ_e, κ_e), (ndim^2, ndim^2)) reshape(Fep(B, γ_e, κ_e), (ndim^2, ndim^2))]; [reshape(Kep(B, γ_e, κ_e), (ndim^2, ndim^2)) reshape(Mep(B, γ_e, κ_e), (ndim^2, ndim^2))]]

    #= Get the stiffness relationship between the stress rates and the strain rates
    starting with unknown stress to known strain=#
    Euk = incremental_stiffness[.!control, .!control]
    # The unkown stress to unknown strain
    Euu = incremental_stiffness[.!control, control]
    # The known stress to known strain
    Ekk = incremental_stiffness[control, .!control]
    # The known stress to unknown strain
    Eku = incremental_stiffness[control, control]

    # Calculate the unknown strain rates
    epsilon_u = inv(Eku)*(generalised_stress_rate[control] - Ekk*generalised_strain_rate[.!control])

    # Replace unknown strain rate with calculated result
    j = 1
    for i = 1:size(generalised_strain_rate)[1]
        if control[i] == 1
            generalised_strain_rate[i] = epsilon_u[j]
            j = j + 1
        end
    end
    γtotdot = reshape(generalised_strain_rate[1:ndim^2], (ndim, ndim))
    κtotdot = reshape(generalised_strain_rate[(ndim^2 + 1):(2*ndim^2)], (ndim, ndim))

    du[1] = Bdot(B, γ_e, κ_e, γtotdot, κtotdot)
    du[2:10] = γedot(B, γ_e, κ_e, γtotdot, κtotdot)
    du[11:19] = κedot(B, γ_e, κ_e, γtotdot, κtotdot)
    du[20:37] = generalised_strain_rate - u[20:37]
    du[38:55] = generalised_stress_rate - u[38:55]
end
