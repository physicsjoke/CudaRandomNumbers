module CudaRandomNumbers

export gamma_rho_dep!, poisson_rho_dep!

const FloatType=Float32

include("Poisson.jl")
include("Gamma.jl")




end # module