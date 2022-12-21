using CudaRandomNumbers
using CUDA
using CUDA: i32


λ = 5.0f0
n = 2^20
rng = CUDA.RNG()

CUDA.CUDA.@profile begin

A = CUDA.ones(Float32, N) .* λ
_fact_table_up_to_9 = Array{Float32}(undef, 9)
_fact_table_up_to_9[1] = Float32(1.0)
_fact_table_up_to_9[2] = Float32(2.0)
_fact_table_up_to_9[3] = Float32(6.0)
_fact_table_up_to_9[4] = Float32(24.0)
_fact_table_up_to_9[5] = Float32(120.0)
_fact_table_up_to_9[6] = Float32(720.0)
_fact_table_up_to_9[7] = Float32(5040.0)
_fact_table_up_to_9[8] = Float32(40320.0)
_fact_table_up_to_9[9] = Float32(362880.0)
cu_fact_table_up_to_9 = CuArray(_fact_table_up_to_9)


function test_kernel!(rho::CuDeviceArray{Float32}, seed::UInt32, counter::UInt32, table)
    device_rng = Random.default_rng()
    @inbounds Random.seed!(device_rng, seed, counter)

    # grid-stride loop
    tid    = Int32(threadIdx().x)
    window = Int32((blockDim().x)        *  gridDim().x)
    offset = Int32((blockIdx().x - 1i32) * blockDim().x)

    while offset < length(rho)
        i = Int32(tid + offset)
        if i <=  length(rho)
            if  rho[i] >= zero(Float32)
                rho[i]  = rand_poisson(device_rng, rho[i], table)
            end
        end
        offset += window
    end
    return nothing
end

kernel = @cuda launch=false test_kernel!(A, rng.seed, rng.counter, cu_fact_table_up_to_9)
config = CUDA.launch_configuration(kernel.fun; max_threads=64)
threads = max(32,min(length(A),config.threads))
blocks = min(config.blocks, cld(length(A), threads))
CUDA.@sync kernel(A, rng.seed, rng.counter; threads=threads, blocks=blocks)

CUDA.@sync B = Array(A)
B
end # CUDA.@profile