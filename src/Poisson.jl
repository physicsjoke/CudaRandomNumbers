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

# const _fact_table_up_to_9 = Vector{FloatType}(undef, 9)
# _fact_table_up_to_9[1] = FloatType(1.0)
# _fact_table_up_to_9[2] = FloatType(2.0)
# _fact_table_up_to_9[3] = FloatType(6.0)
# _fact_table_up_to_9[4] = FloatType(24.0)
# _fact_table_up_to_9[5] = FloatType(120.0)
# _fact_table_up_to_9[6] = FloatType(720.0)
# _fact_table_up_to_9[7] = FloatType(5040.0)
# _fact_table_up_to_9[8] = FloatType(40320.0)
# _fact_table_up_to_9[9] = FloatType(362880.0)

# _fact_table_up_to_9 = Array{Float32}(undef, 9)
# _fact_table_up_to_9[1] = Float32(1.0)
# _fact_table_up_to_9[2] = Float32(2.0)
# _fact_table_up_to_9[3] = Float32(6.0)
# _fact_table_up_to_9[4] = Float32(24.0)
# _fact_table_up_to_9[5] = Float32(120.0)
# _fact_table_up_to_9[6] = Float32(720.0)
# _fact_table_up_to_9[7] = Float32(5040.0)
# _fact_table_up_to_9[8] = Float32(40320.0)
# _fact_table_up_to_9[9] = Float32(362880.0)
# cu_fact_table_up_to_9 = CuArray(_fact_table_up_to_9)

function factorial_lookup_up_to_9(n::FloatType)
    n == zero(FloatType) && return one(FloatType)
    n ==  one(FloatType) && return one(FloatType)
    n == FloatType(2.0)  && return FloatType(2.0)
    n == FloatType(3.0)  && return FloatType(6.0)
    n == FloatType(4.0)  && return FloatType(24.0)
    n == FloatType(5.0)  && return FloatType(120.0)
    n == FloatType(6.0)  && return FloatType(720.0)
    n == FloatType(7.0)  && return FloatType(5040.0)
    n == FloatType(8.0)  && return FloatType(40320.0)
    n == FloatType(9.0)  && return FloatType(362880.0)
    n >= FloatType(10.0) && return NaN
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
        U = Random.rand(rng, μType)
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
            py =  μ^K/factorial_lookup_up_to_9(K)
        else
            δ  = FloatType(0.08333333333333333) / K
            δ -= FloatType(4.8)*δ^3
            V  = (μ-K) / K
            px = K*log1pmx(V) - δ # avoids need for table
            py = FloatType(0.3989422804014327) / sqrt(K)
        end

        X  = (K-μ+FloatType(0.5)) / s
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
        U = 2 * Random.rand(rng, μType) - one(μType)
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
            py =  μ^K/factorial_lookup_up_to_9(K)
        else
            δ  = FloatType(0.08333333333333333) /K
            δ -= FloatType(4.8)*δ^3
            V  = (μ-K) / K
            px = K*log1pmx(V) - δ # avoids need for table
            py = FloatType(0.3989422804014327) / sqrt(K)
        end

        X  = (K-μ+FloatType(0.5)) / s
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
