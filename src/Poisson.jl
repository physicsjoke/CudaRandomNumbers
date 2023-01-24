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




function factorial_lookup_up_to_9(n::FloatType)
    n == zero(FloatType) && return one(FloatType)
    n ==  one(FloatType) && return one(FloatType)
    n == 2.0f0  && return 2.0f0
    n == 3.0f0  && return 6.0f0
    n == 4.0f0  && return 24.0f0
    n == 5.0f0  && return 120.0f0
    n == 6.0f0  && return 720.0f0
    n == 7.0f0  && return 5040.0f0
    n == 8.0f0  && return 40320.0f0
    n == 9.0f0  && return 362880.0f0
    n >= 10.0f0 && return NaN
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


        ω  = 0.3989422804014327f0   / s
        b1 = 0.041666666666666664f0 / μ
        b2 = 0.3f0 * b1 * b1
        c3 = 0.14285714285714285f0 * b1 * b2
        c2 = b2 - 15 * c3
        c1 = b1 - 6 * b2 + 45 * c3
        c0 = 1 - b1 + 3 * b2 - 15 * c3

        if K < 10.0f0
            px = -μ
            py =  μ^K/factorial_lookup_up_to_9(K)
        else
            δ  = 0.08333333333333333f0 / K
            δ -= 4.8f0 * δ^3
            V  = (μ-K) / K
            px = K*log1pmx(V) - δ # avoids need for table
            py = 0.3989422804014327f0 / sqrt(K)
        end

        X  = ( K- μ + 0.5f0) / s
        X2 =   X^2
        fx =  -X2 / 2 # missing negation in pseudo-algorithm, but appears in fortran code.
        fy =   ω* (((c3*X2 + c2)*X2 + c1)*X2 + c0)

        # Step Q
        if fy * (1 - U) <= py * exp(px - fx)
            return K
        end
    end

    while true
        # Step E
        E = randexp(rng, μType)
        U = 2 * Random.rand(rng, μType) - one(μType)
        T = 1.8f0 + copysign(E, U)
        if T <= -0.6744f0
            continue
        end

        K = round( μ + s * T, RoundDown)
        #px, py, fx, fy = procf(μ, K, s)

        ω  = 0.3989422804014327f0   / s
        b1 = 0.041666666666666664f0 / μ
        b2 = 0.3f0 * b1 * b1
        c3 = 0.14285714285714285f0 * b1 * b2
        c2 = b2 - 15 * c3
        c1 = b1 - 6 * b2 + 45 * c3
        c0 = 1 - b1 + 3 * b2 - 15 * c3

        if K < 10.0f0
            px = -μ
            py =  μ^K/factorial_lookup_up_to_9(K)
        else
            δ  = 0.08333333333333333f0 /K
            δ -= 4.8f0 * δ^3
            V  = (μ-K) / K
            px = K*log1pmx(V) - δ # avoids need for table
            py = 0.3989422804014327f0 / sqrt(K)
        end

        X  = (K - μ + 0.5f0) / s
        X2 =  X^2
        fx = -X2 / 2 # missing negation in pseudo-algorithm, but appears in fortran code.
        fy =  ω*(((c3*X2+c2)*X2+c1)*X2+c0)

        c = 0.1069f0 / μ

        # Step H
        if c*abs(U) <= py*exp(px + E) - fy*exp(fx + E)
            return K
        end
    end
end


rand_poisson(rng::AbstractRNG, μ::Real) = rand(rng, PoissonADSampler(μ))


