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
                    for i in 3:n_cumulants
                        @test cumulant(A, i) ≈ λ rtol=sqrt(10000/N) atol=1.0f-6
                    end
                end
                @testset "CPU: λ = $λ, N = $N" for λ in [1.0f0, 5.0f0, 30.0f0], N in [2^10, 2^16, 2^19, 2^22]
                    B = ones(Float32, N) .* λ
                    for i in eachindex(B)
                        B[i] = rand_poisson(Random.default_rng(), B[i])
                    end
                    test_mean_variance(B, λ, N)
                    #test_higher_cumulants(B, λ, N, 5)
                end
                @testset "GPU: λ = $λ, N = $N" for λ in [1.0f0, 5.0f0, 30.0f0], N in [2^19]
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
                    test_mean_variance(B, λ, N)
                    #test_higher_cumulants(B, λ, N, 5)
                end
            end
        end   
    end


    # @testset "Gamma" begin
    #     @testset "Parameter regimes" begin
    #         @testset "Uniform input array" begin
    #             function test_mean_variance(A, k, N)
    #                 @test mean(A) ≈ k rtol=sqrt(10000/N) atol=1.0f-7
    #                 @test var(A) ≈ k rtol=sqrt(10000/N) atol=1.0f-7
    #             end
    #             @testset "GPU: k = $k, N = $N" for k in [1.0f0, 5.0f0, 10.0f0, 15.0f0, 30.0f0], N in [2^7, 2^10, 2^13, 2^16, 2^19, 2^22]
    #                 A = k*CUDA.ones(Float32, N)
    #                 CudaRandomNumbers.gamma_rho_dep!(rng, A)
    #                 test_mean_variance(A, k, N)
    #             end
    #             @testset "CPU: k = $k, N = $N" for k in [1.0f0, 5.0f0, 10.0f0, 15.0f0, 30.0f0], N in [2^7, 2^10, 2^13, 2^16, 2^19, 2^22]
    #                 A = k*ones(Float32, N)
    #                 CudaRandomNumbers.gamma_rho_dep!(A)
    #                 test_mean_variance(A, k, N)
    #             end
    #         end
    #     end
    # end
end