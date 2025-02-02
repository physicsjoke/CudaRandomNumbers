using CudaRandomNumbers
using CUDA
using CUDA: i32
using Distributions
using Statistics
using StatsBase
using BenchmarkTools
using Test
using Random


@testset "CudaRandomNumbers.jl" begin
    rng = CUDA.RNG()
    @testset "Poisson" begin
        @testset "Parameter regimes" begin
            @testset "Uniform input array" begin
                function test_mean_variance(A, λ, N)
                    @test mean(A) ≈ var(A) rtol=sqrt(10000/N) atol=1.0f-6
                    @test mean(A) ≈ λ rtol=sqrt(10000/N) atol=1.0f-6
                end

                function test_higher_cumulants(A, λ, N, n_cumulants)
                    measured_cumulants = cumulant(A, 3:n_cumulants)
                    for i in eachindex(measured_cumulants)
                        @test measured_cumulants[i] ≈ λ rtol=sqrt(10000/N) atol=1.0f-6
                    end
                end
                @testset "CPU: λ = $λ, N = $N" for λ in [1.0f0, 5.0f0, 30.0f0, 200f0, 2000f0], N in [2^10, 2^16, 2^19]
                    B = ones(Float32, N) .* λ
                    for i in eachindex(B)
                        B[i] = rand_poisson(Random.default_rng(), B[i])
                    end
                    test_mean_variance(B, λ, N)
                    #test_higher_cumulants(B, λ, N, 5)
                end
                @testset "GPU: λ = $λ, N = $N" for λ in [1.0f0, 5.0f0, 30.0f0, 200f0, 2000f0], N in [2^10, 2^16, 2^19, 2^22, 2^26]
                    A = CUDA.ones(Float32, N) .* λ

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
                    test_mean_variance(B, λ, N)
                    test_higher_cumulants(B, λ, N, 5)
                end
            end
        end   
    end


    @testset "Gamma" begin
        @testset "Parameter regimes" begin
            @testset "Uniform input array" begin


                function test_cumulants(A, k, N)
                    measured_cumulants = cumulant(A, 1:4)
                    theoretical_cumulants = [k, k, 2*k, 6*k]
                    for i in eachindex(measured_cumulants)
                        @test measured_cumulants[i] ≈ theoretical_cumulants[i] rtol=sqrt(10000/N) atol=1.0f-6
                    end
                end

                @testset "GPU: k = $k, N = $N" for k in [1.0f0, 5.0f0, 10.0f0, 15.0f0, 30.0f0], N in [2^7, 2^10, 2^13, 2^16, 2^19, 2^22]
                    A = CUDA.ones(Float32, N) .* k

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
                                    rho[i]  = rand_gamma(device_rng, rho[i], 1.0f0)
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
                    test_cumulants(B, k, N)
                end

                @testset "CPU: k = $k, N = $N" for k in [1.0f0, 5.0f0, 10.0f0, 15.0f0, 30.0f0], N in [2^7, 2^10, 2^13, 2^16, 2^19]
                    B = ones(Float32, N) .* k
                    for i in eachindex(B)
                        B[i] = rand_gamma(Random.default_rng(), B[i], 1)
                    end
                    test_cumulants(B, k, N)
                end
            end
        end
    end
end