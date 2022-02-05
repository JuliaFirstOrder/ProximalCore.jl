module ProximalCore

using LinearAlgebra

is_convex(::Type) = false
is_convex(::T) where T = is_convex(T)

is_generalized_quadratic(::Type) = false
is_generalized_quadratic(::T) where T = is_generalized_quadratic(T)

"""
    gradient!(y, f, x)

In-place gradient (and value) of `f` at `x`.

The gradient is written to the (pre-allocated) array `y`, which should have the same shape/size as `x`.

Returns the value `f` at `x`.

See also: [`gradient`](@ref).
"""
gradient!(y, f, x) = error("`gradient!` is not defined for $(typeof(f))")

"""
    gradient(f, x)

Gradient (and value) of `f` at `x`.

Return a tuple `(y, fx)` consisting of
- `y`: the gradient of `f` at `x`
- `fx`: the value of `f` at `x`

See also: [`gradient!`](@ref).
"""
function gradient(f, x)
    y = similar(x)
    fx = gradient!(y, f, x)
    return y, fx
end

"""
    prox!(y, f, x, gamma=1)

In-place proximal mapping for `f`, evaluated at `x`, with stepsize `gamma`.

The proximal mapping is defined as
```math
\\mathrm{prox}_{\\gamma f}(x) = \\arg\\min_z \\left\\{ f(z) + \\tfrac{1}{2\\gamma}\\|z-x\\|^2 \\right\\}.
```
The result is written to the (pre-allocated) array `y`, which should have the same shape/size as `x`.

Returns the value of `f` at `y`.

See also: [`prox`](@ref).
"""
prox!(y, f, x, gamma) = error("`prox!` is not defined for $(typeof(f))")
prox!(y, f, x) = prox!(y, f, x, 1)

"""
    prox(f, x, gamma=1)

Proximal mapping for `f`, evaluated at `x`, with stepsize `gamma`.

    The proximal mapping is defined as
```math
\\mathrm{prox}_{\\gamma f}(x) = \\arg\\min_z \\left\\{ f(z) + \\tfrac{1}{2\\gamma}\\|z-x\\|^2 \\right\\}.
```

Returns a tuple `(y, fy)` consisting of
- `y`: the output of the proximal mapping of `f` at `x` with stepsize `gamma`
- `fy`: the value of `f` at `y`

See also: [`prox!`](@ref).
"""
function prox(f, x, gamma=1)
    y = similar(x)
    fy = prox!(y, f, x, gamma)
    return y, fy
end

struct Zero end

(f::Zero)(x) = real(eltype(x))(0)

function prox!(y, ::Zero, x, gamma)
    y .= x
    return real(eltype(y))(0)
end

is_convex(::Type{Zero}) = true
is_generalized_quadratic(::Type{Zero}) = true

struct IndZero end

function (f::IndZero)(x)
    R = real(eltype(x))
    if iszero(x)
        return R(Inf)
    end
    return R(0)
end

is_convex(::Type{IndZero}) = true
is_generalized_quadratic(::Type{IndZero}) = true

function prox!(y, ::IndZero, x, gamma)
    R = real(eltype(x))
    y .= R(0)
    return R(0)
end

struct ConvexConjugate{T}
    f::T
end

is_convex(::Type{<:ConvexConjugate}) = true
is_generalized_quadratic(::Type{ConvexConjugate{T}}) where T = is_generalized_quadratic(T)

function prox_conjugate!(y, u, f, x, gamma)
    u .= x ./ gamma
    v = prox!(y, f, u, 1 / gamma)
    v = real(dot(x, y)) - gamma * real(dot(y, y)) - v
    y .= x .- gamma .* y
    return v
end

prox_conjugate!(y, f, x, gamma) = prox_conjugate!(y, similar(x), f, x, gamma)

prox!(y, g::ConvexConjugate, x, gamma) = prox_conjugate!(y, g.f, x, gamma)

convex_conjugate(f) = ConvexConjugate(f)
convex_conjugate(::Zero) = IndZero()
convex_conjugate(::IndZero) = Zero()

end # module
