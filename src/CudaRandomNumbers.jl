module CudaRandomNumbers

export rand_gamma, rand_poisson

using StatsFuns
const FloatType=Float32

    
include("Poisson.jl")
include("Gamma.jl")

end # module