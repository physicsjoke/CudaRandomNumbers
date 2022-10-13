using Random
using CUDA
using CUDA: i32


function gamma_rho_dep!(rng::CUDA.RNG, rho)

    # The GPU kernel
    function kernel!(rho, seed::UInt32, counter::UInt32)
        device_rng = Random.default_rng()
        # initialize the state
        @inbounds Random.seed!(device_rng, seed, counter)


        # grid-stride loop
        tid    = threadIdx().x
        window = (blockDim().x) * gridDim().x
        offset = (blockIdx().x - 1i32) * blockDim().x

        while offset < length(rho)
            i = tid + offset
            if i <=  length(rho)
                if rho[i]<FloatType(typemax(UInt32))
                    @inbounds k = round(UInt32,rho[i])
                    @inbounds rho[i] = 0.0f0
                    for j=1:k
                    
                        @inbounds rho[i] -= CUDA.log(Random.rand(device_rng, FloatType))
                    end
                end
            end
            offset += window
        end
        return nothing
    end

    kernel = @cuda launch=false kernel!(rho, rng.seed, rng.counter)
    config = CUDA.launch_configuration(kernel.fun; max_threads=64)
    threads = max(32,min(length(rho),config.threads))
    blocks = min(config.blocks, cld(length(rho), threads))
    CUDA.@sync kernel(rho, rng.seed, rng.counter; threads=threads, blocks=blocks)

    new_counter = Int64(rng.counter) + length(rho)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow     
    rng.counter = remainder
    rho
end