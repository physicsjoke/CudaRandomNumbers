using Random
using CUDA
using CUDA: i32

# Algorithm from:
#
#   J.H. Ahrens, U. Dieter (1982)
#   "Computer Generation of Poisson Deviates from Modified Normal Distributions"
#   ACM Transactions on Mathematical Software, 8(2):163-179
#
#   For μ sufficiently large, (i.e. >= 10.0f0)
#
# Adapted from Distributions.jl at https://github.com/JuliaStats/Distributions.jl

const _fact_table_up_to_9 = Vector{FloatType}(undef, 9)
_fact_table_up_to_9[1] = FloatType(1.0)
_fact_table_up_to_9[2] = FloatType(2.0)
_fact_table_up_to_9[3] = FloatType(6.0)
_fact_table_up_to_9[4] = FloatType(24.0)
_fact_table_up_to_9[5] = FloatType(120.0)
_fact_table_up_to_9[6] = FloatType(720.0)
_fact_table_up_to_9[7] = FloatType(5040.0)
_fact_table_up_to_9[8] = FloatType(40320.0)
_fact_table_up_to_9[9] = FloatType(362880.0)

function factorial_lookup_up_to_9(n::FloatType, table)
    n == zero(FloatType) && return one(FloatType)
    @inbounds factorial = table[trunc(Int32, n)]
    return factorial
end

struct PoissonADSampler{T<:Real}
    μ::T
    s::T
    d::T
    L::FloatType
end

function PoissonADSampler(μ::Real)
    s = sqrt(μ)
    d = 6 * μ^2
    L = round( μ - 1.1484f0, RoundDown)

    PoissonADSampler(promote(μ, s, d)..., L)
end

function rand(rng::AbstractRNG, sampler::PoissonADSampler)
    μ = sampler.μ
    s = sampler.s
    d = sampler.d
    L = sampler.L
    μType = typeof(μ)

    # Step N
    G = μ + s * randn(rng, μType)

    if G >= zero(G)
        K = round(G, RoundDown)
        # Step I
        if K >= L
            return K
        end

        # Step S
        U = rand(rng, μType)
        if d * U >= (μ - K)^3
            return K
        end


        ω  = FloatType(0.3989422804014327)   / s
        b1 = FloatType(0.041666666666666664) / μ
        b2 = FloatType(0.3) * b1 * b1
        c3 = FloatType(0.14285714285714285) * b1 * b2
        c2 = b2 - 15 * c3
        c1 = b1 - 6 * b2 + 45 * c3
        c0 = 1 - b1 + 3 * b2 - 15 * c3

        if K < FloatType(10.0)
            px = -μ
            py =  μ^K/factorial_lookup_up_to_9(K, _fact_table_up_to_9)
        else
            δ  = FloatType(0.08333333333333333) /K
            δ -= FloatType(4.8)*δ^3
            V  = (μ-K) / K
            px = K*log1pmx(V) - δ # avoids need for table
            py = FloatType(0.3989422804014327) / sqrt(K)
        end

        X  = (K-μ+FloatType(0.5f0)) / s
        X2 =  X^2
        fx = -X2 / 2 # missing negation in pseudo-algorithm, but appears in fortran code.
        fy =  ω*(((c3*X2+c2)*X2+c1)*X2+c0)

        # Step Q
        if fy * (1 - U) <= py * exp(px - fx)
            return K
        end
    end

    while true
        # Step E
        E = randexp(rng, μType)
        U = 2 * rand(rng, μType) - one(μType)
        T = FloatType(1.8) + copysign(E, U)
        if T <= FloatType(-0.6744)
            continue
        end

        K = round( μ + s * T, RoundDown)
        #px, py, fx, fy = procf(μ, K, s)

        ω  = FloatType(0.3989422804014327)   / s
        b1 = FloatType(0.041666666666666664) / μ
        b2 = FloatType(0.3) * b1 * b1
        c3 = FloatType(0.14285714285714285) * b1 * b2
        c2 = b2 - 15 * c3
        c1 = b1 - 6 * b2 + 45 * c3
        c0 = 1 - b1 + 3 * b2 - 15 * c3

        if K < FloatType(10.0)
            px = -μ
            py =  μ^K/factorial_lookup_up_to_9(K, _fact_table_up_to_9)
        else
            δ  = FloatType(0.08333333333333333) /K
            δ -= FloatType(4.8)*δ^3
            V  = (μ-K) / K
            px = K*log1pmx(V) - δ # avoids need for table
            py = FloatType(0.3989422804014327) / sqrt(K)
        end

        X  = (K-μ+FloatType(0.5f0)) / s
        X2 =  X^2
        fx = -X2 / 2 # missing negation in pseudo-algorithm, but appears in fortran code.
        fy =  ω*(((c3*X2+c2)*X2+c1)*X2+c0)
        
        c = FloatType(0.1069) / μ

        # Step H
        if c*abs(U) <= py*exp(px + E) - fy*exp(fx + E)
            return K
        end
    end
end


rand_poisson(rng::AbstractRNG, μ::Real) = rand(rng, PoissonADSampler(μ))
