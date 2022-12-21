using CudaRandomNumbers
using CUDA
using CUDA: i32
using Random

λ = 5.0f0
N = 2^20
rng = CUDA.RNG()


A = CUDA.ones(Float32, N) .* λ

CUDA.@profile begin
function test_kernel!(rho::CuDeviceArray{Float32}, seed::UInt32, counter::UInt32)
    device_rng = Random.default_rng()
    @inbounds Random.seed!(device_rng, seed, counter)

    # grid-stride loop
    tid    = threadIdx().x
    window = (blockDim().x)        *  gridDim().x
    offset = (blockIdx().x - 1i32) * blockDim().x

    while offset < length(rho)
        i = tid + offset
        if i <=  length(rho)
            if  rho[i] >= zero(Float32)
                rho[i]  = rand_poisson(device_rng, rho[i])
            end
        end
        offset += window
    end
    return nothing
end

kernel = @cuda launch=false test_kernel!(A, rng.seed, rng.counter)
config = CUDA.launch_configuration(kernel.fun; max_threads=64)
threads = max(32,min(length(A),config.threads))
blocks = min(config.blocks, cld(length(A), threads))
CUDA.@sync kernel(A, rng.seed, rng.counter; threads=threads, blocks=blocks)

CUDA.@sync B = Array(A)
B
end # CUDA.@profile