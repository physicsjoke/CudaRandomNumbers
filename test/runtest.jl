using CudaRandomNumbers
using CUDA
using Distributions
using Statistics
using StatsBase
using BenchmarkTools
using Test


@testset "CudaRandomNumbers.jl" begin
    rng = CUDA.RNG()
    @testset "FloatType = $FloatType" for FloatType in [Float32, Float64]
        @testset "Poisson" begin
            @testset "Distributions" begin
            
            end
            @testset "Parameter regimes" begin
                @testset "Uniform input array" begin
                    function test_mean_variance(A, λ, N)
                        @test mean(A) ≈ var(A) rtol=sqrt(10/N) atol=1.0f-7
                        @test mean(A) ≈ λ rtol=sqrt(10/N) atol=1.0f-7
                    end

                    function test_higher_cumulants(A, λ, N, n_cumulants)
                        for i in 3:n_cumulants
                            @test cumulant(A, i) ≈ λ rtol=sqrt(10/N) atol=1.0f-7
                        end
                    end
                    @testset "λ = $λ, N = $N" for λ in [1.0f0, 5.0f0, 10.0f0, 15.0f0, 30.0f0], N in [2^7, 2^10, 2^13, 2^16, 2^19, 2^22]
                        A = CUDA.ones(FloatType, N)
                        CudaRandomNumbers.poisson_rho_dep!(rng, A, λ)
                        A_copy = Array(A)
                        test_mean_variance(A_copy, λ, N)
                        test_higher_cumulants(A_copy, λ, N, 10)
                    end
                end

            end
        
        end
        @testset "Gamma" begin
            @testset "Parameter regimes" begin
                @testset "Uniform input array" begin
                    function test_mean_variance(A, k, N)
                        @test mean(A) ≈ k rtol=sqrt(10/N) atol=1.0f-7
                        @test var(A) ≈ k rtol=sqrt(10/N) atol=1.0f-7
                    end

                    @testset "k = $k, N = $N" for k in [1.0f0, 5.0f0, 10.0f0, 15.0f0, 30.0f0], N in [2^7, 2^10, 2^13, 2^16, 2^19, 2^22]
                        A = k*CUDA.ones(FloatType, N)
                        CudaRandomNumbers.gamma_rho_dep!(rng, A)
                        test_mean_variance(A, k, N)
                    end
                end

            end
        end
    end
end