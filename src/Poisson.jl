using Random
using CUDA
using CUDA: i32


function poisson_rho_dep!(rng::CUDA.RNG, rho::AnyCuArray, λ::FloatType)

    # The GPU kernel
    function kernel!(rho, λ::FloatType, seed::UInt32, counter::UInt32)
        device_rng = Random.default_rng()
        # initialize the stateS
        @inbounds Random.seed!(device_rng, seed, counter)

        # grid-stride loop
        tid    = threadIdx().x
        window = (blockDim().x) * gridDim().x
        offset = (blockIdx().x - 1i32) * blockDim().x

        while offset < length(rho)
            
            i = tid + offset
            if i <= length(rho)
                if isnan(rho[i]) @cuprintln("NaN beginning") end
                if rho[i] > 0
                    k = 0.0f0
                    p = 1.0f0
                    @inbounds L = CUDA.exp(-λ*rho[i])
                    while true
                        k += 1.0f0
                        p *= Random.rand(device_rng, FloatType)
                        p <= L && break
                    end
                    rho[i] = k-1.0f0
                end
            end
            offset += window
            #if isnan(rho[i]) @cuprintln("NaN end") end
        end
        
        return nothing
    end

    kernel = @cuda launch=false kernel!(rho, λ, rng.seed, rng.counter)
    config = CUDA.launch_configuration(kernel.fun; max_threads=64)
    threads = max(32, min(length(rho),config.threads))
    blocks = cld(length(rho), threads)
    CUDA.@sync kernel(rho, λ, rng.seed, rng.counter; threads, blocks)

    new_counter = Int64(rng.counter) + length(rho)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow     
    rng.counter = remainder
    rho
end

# CPU version
function poisson_rho_dep!(rng::Random.AbstractRNG, rho, λ::FloatType)

    for i in eachindex(rho)
        k = 0.0f0
        p = 1.0f0
        @inbounds L = exp(-λ*rho[i])
        while true
            k += 1.0f0
            p *= Random.rand(rng, FloatType)
            p <= L && break
        end
        rho[i] = k-1.0f0
    end
    return nothing
end

# RNG-less interface
poisson_rho_dep!(rho, λ::FloatType) = poisson_rho_dep!(Random.Xoshiro(), rho, λ)