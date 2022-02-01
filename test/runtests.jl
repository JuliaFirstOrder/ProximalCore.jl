using Test
using LinearAlgebra
using ProximalCore
using ProximalCore: prox, convex_conjugate#, moreau_envelope
using ProximalCore: Zero, IndZero
import ProximalCore: prox!, convexity, regularity, quadraticness

@testset "Package sanity checks" begin
    @test isempty(detect_unbound_args(ProximalCore))
end

struct UnitInfNormBall end

convexity(::Type{UnitInfNormBall}) = ProximalCore.IsJustConvex()

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

        @test convexity(Zero) == ProximalCore.IsJustConvex()
        @test regularity(Zero) == ProximalCore.IsSmooth()
        @test quadraticness(Zero) == ProximalCore.IsGeneralizedQuadratic()

    end

    @testset "IndZero" begin

        @test convexity(IndZero) == ProximalCore.IsStronglyConvex()
        @test regularity(IndZero) == ProximalCore.UnknownRegularity()
        @test quadraticness(IndZero) == ProximalCore.IsGeneralizedQuadratic()

    end
    
    @testset "Conjugation" begin

        @test typeof(convex_conjugate(Zero())) == IndZero
        @test typeof(convex_conjugate(IndZero())) == Zero

    end

end

@testset "Custom types" begin

    @testset "UnitInfNormBall" begin
        f = UnitInfNormBall()
        x = Float64[-2, -1, 0, 1, 2]
        y, v = @inferred prox(f, x)

        @test convexity(typeof(f)) == ProximalCore.IsJustConvex()
        @test quadraticness(typeof(f)) == ProximalCore.UnknownQuadraticness()
        
        @test typeof(v) == real(eltype(x))
        @test y == Float64[-1, -1, 0, 1, 1]
        @test v == 0

        f_conjugate = convex_conjugate(f)
        y, v = @inferred prox(f_conjugate, x)

        @test convexity(typeof(f_conjugate)) == ProximalCore.IsJustConvex()
        @test quadraticness(typeof(f_conjugate)) == ProximalCore.UnknownQuadraticness()

        @test typeof(v) == real(eltype(x))
        @test y == Float64[-1, 0, 0, 0, 1]
        @test v == norm(y, 1)

        y, v = @inferred prox(f_conjugate, x, 0.5)

        @test typeof(v) == real(eltype(x))
        @test y == Float64[-1.5, -0.5, 0, 0.5, 1.5]
        @test v == norm(y, 1)

        f_biconjugate = convex_conjugate(f_conjugate)
        y, v = @inferred prox(f_biconjugate, x, 0.5)

        @test convexity(typeof(f_biconjugate)) == ProximalCore.IsJustConvex()
        @test quadraticness(typeof(f_biconjugate)) == ProximalCore.UnknownQuadraticness()

        @test typeof(v) == real(eltype(x))
        @test y == Float64[-1, -1, 0, 1, 1]
        @test v == 0
    end

end
