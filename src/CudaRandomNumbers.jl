module CudaRandomNumbers

export rand_gamma, rand_poisson

using StatsFuns
using ChangePrecision
const FloatType=Float32

@changeprecision FloatType begin
    
include("Poisson.jl")
include("Gamma.jl")


end # changeprecision FloatType
end # module