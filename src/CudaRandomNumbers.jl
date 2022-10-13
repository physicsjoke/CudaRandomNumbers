module CudaRandomNumbers


const FloatType=Float32

include("Poisson.jl")
include("Gamma.jl")

export gamma_rho_dep!, poisson_rho_dep!


end # module