using Test
using LinearAlgebra
using ProximalCore
using ProximalCore: prox, convex_conjugate#, moreau_envelope
using ProximalCore: Zero, IndZero
import ProximalCore: prox!, is_convex, is_generalized_quadratic

@testset "Package sanity checks" begin
    @test isempty(detect_unbound_args(ProximalCore))
end

struct UnitInfNormBall end

is_convex(::Type{UnitInfNormBall}) = true

function (::UnitInfNormBall)(x)
    R = eltype(y)
    return norm(x, Inf) <= 1 ? R(0) : R(Inf)
end

function prox!(y, ::UnitInfNormBall, x, gamma)
    R = eltype(y)
    y .= max.(R(-1), min.(R(1), x))
    return R(0)
end

@testset "Basic types" begin

    @testset "Zero" begin

        @test is_convex(Zero())
        @test is_generalized_quadratic(Zero())
        
    end

    @testset "IndZero" begin

        @test is_convex(IndZero())
        @test is_generalized_quadratic(IndZero())

    end
    
    @testset "Conjugation" begin

        @test convex_conjugate(Zero()) isa IndZero
        @test convex_conjugate(IndZero()) isa Zero

    end

end

@testset "Custom types" begin

    @testset "UnitInfNormBall" begin
        f = UnitInfNormBall()
        x = Float64[-2, -1, 0, 1, 2]
        y, v = @inferred prox(f, x)

        @test is_convex(f)
        @test !is_generalized_quadratic(f)
        
        @test typeof(v) == real(eltype(x))
        @test y == Float64[-1, -1, 0, 1, 1]
        @test v == 0

        f_conjugate = convex_conjugate(f)
        y, v = @inferred prox(f_conjugate, x)

        @test is_convex(f_conjugate)
        @test !is_generalized_quadratic(f_conjugate)

        @test typeof(v) == real(eltype(x))
        @test y == Float64[-1, 0, 0, 0, 1]
        @test v == norm(y, 1)

        y, v = @inferred prox(f_conjugate, x, 0.5)

        @test typeof(v) == real(eltype(x))
        @test y == Float64[-1.5, -0.5, 0, 0.5, 1.5]
        @test v == norm(y, 1)

        f_biconjugate = convex_conjugate(f_conjugate)
        y, v = @inferred prox(f_biconjugate, x, 0.5)

        @test is_convex(f_biconjugate)
        @test !is_generalized_quadratic(f_biconjugate)

        @test typeof(v) == real(eltype(x))
        @test y == Float64[-1, -1, 0, 1, 1]
        @test v == 0
    end

end
