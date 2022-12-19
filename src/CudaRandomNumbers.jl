module CudaRandomNumbers

export gamma_rho_dep!, rand_poisson

using StatsFuns
const FloatType=Float32

include("Poisson.jl")
include("Gamma.jl")




end # module