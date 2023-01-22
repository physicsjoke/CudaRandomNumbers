using Random
using CUDA
using CUDA: i32


struct GammaMTSampler{T<:Real}
    d::T
    c::T
    κ::T
    r::T
end

function GammaMTSampler(shape::Real, scale::Real)
    # Setup (Step 1)
    d = shape - 1//3
    c = inv(3 * sqrt(d))

    # Pre-compute scaling factor
    κ = d * scale

    # We also pre-compute the factor in the squeeze function
    return GammaMTSampler(promote(d, c, κ, 331//10_000)...)
end

function rand(rng::AbstractRNG, s::GammaMTSampler{T}) where {T<:Real}
    d = s.d
    c = s.c
    κ = s.κ
    r = s.r
    z = zero(T)
    while true
        # Generate v (Step 2)
        x = randn(rng, T)
        cbrt_v = 1 + c * x
        while cbrt_v <= z # requires x <= -sqrt(9 * shape - 3)
            x = randn(rng, T)
            cbrt_v = 1 + c * x
        end
        v = cbrt_v^3

        # Generate uniform u (Step 3)
        u = Random.rand(rng, T)

        # Check acceptance (Step 4 and 5)
        xsq = x^2
        if u < 1 - r * xsq^2 || log(u) < xsq / 2 + d * logmxp1(v)
            return v * κ
        end
    end
end

rand_gamma(rng::AbstractRNG, shape::Real, scale::Real) = rand(rng, GammaMTSampler(shape, scale))