"""
    is_convex(T::Type)

Returns `true` if the type `T` represents a convex function.
A function f(x) is convex if its domain is a convex set and for all x, y in the domain and for all λ in [0, 1], we have f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y).
"""
is_convex(::Type) = false
is_convex(::T) where T = is_convex(T)

"""
    is_generalized_quadratic(T::Type)

Returns `true` if the type `T` represents a generalized quadratic function, i.e. a quadratic function over a subspace and +∞ outside of it.
A quadratic function has the form f(x)=(1/2)*<Ax, x> + <b, x> + c where A is a symmetric matrix, b is a vector, and c a real number.
"""
is_generalized_quadratic(::Type) = false
is_generalized_quadratic(::T) where T = is_generalized_quadratic(T)

"""
    is_proximable(T::Type)

Returns `true` if the type `T` has a proximal operator that can be expressed in a closed formula.
    (i.e. `prox!` function is defined for the type `T`).
"""
is_proximable(::Type) = true
is_proximable(::T) where T = is_proximable(T)

"""
    is_separable(T::Type)

Returns `true` if the type `T` represents a separable function.
A function f(x) is separable if it can applied to each element of x independently.
"""
is_separable(::Type) = false
is_separable(::T) where T = is_separable(T)

"""
    is_singleton_indicator(T::Type)

Returns `true` if the type `T` represents a singleton indicator function.
An indicator function f(x) is a singleton if it is 0 for a single value and ∞ otherwise.
"""
is_singleton_indicator(::Type) = false
is_singleton_indicator(::T) where T = is_singleton_indicator(T)

"""
    is_cone_indicator(T::Type)

Returns `true` if the type `T` represents the indicator function of a cone.
A cone is a set C such that if x ∈ C, then λx ∈ C for any λ ≥ 0. In other words, is a set that is closed under non-negative scaling.
``
"""
is_cone_indicator(::Type) = false
is_cone_indicator(::T) where T = is_cone_indicator(T)

"""
    is_affine_indicator(T::Type)

Returns `true` if the type `T` represents the indicator of an affine set.
An affine set is a set that can be represented as the solution set of a system of linear equations.
In other words, f(x) = 0 if Ax = b, for a given matrix  A and vector b, and ∞ otherwise.
"""
is_affine_indicator(T::Type) = is_singleton_indicator(T)
is_affine_indicator(::T) where T = is_affine_indicator(T)

"""
    is_set_indicator(T::Type)

Returns `true` if the type `T` represents an indicator function of a set.
The indicator of a set S is a function associating 0 to points in S, and ∞ otherwise.
"""
is_set_indicator(T::Type) = is_cone_indicator(T) || is_affine_indicator(T)
is_set_indicator(::T) where T = is_set_indicator(T)

"""
    is_positively_homogeneous(T::Type)

Returns `true` if the type `T` represents a positively homogeneous function.
A function f(x) is positively homogeneous if f(λx) = λf(x) for all λ ≥ 0.
"""
is_positively_homogeneous(T::Type) = is_cone_indicator(T)
is_positively_homogeneous(::T) where T = is_positively_homogeneous(T)

"""
    is_support(T::Type)

Returns `true` if the type `T` represents a support function of a set.
A function f(x) is a support function of a set C if f(x) = sup{⟨x, c⟩ : c ∈ C}.
"""
is_support(T::Type) = is_convex(T) && is_positively_homogeneous(T)
is_support(::T) where T = is_support(T)

"""
    is_locally_smooth(T::Type)

Returns `true` if the type `T` represents a locally smooth function.
A function f(x) is locally smooth if it is smooth on every open set within its domain.
If f is locally smooth, then `gradient!(y, f, x)` is expected to be defined, and it should return the value of f at x and store the gradient in y.
"""
is_locally_smooth(T::Type) = is_smooth(T)
is_locally_smooth(::T) where T = is_locally_smooth(T)

"""
    is_smooth(T::Type)

Returns `true` if the type `T` represents a smooth function.
A function f(x) is smooth if it is continuously differentiable and its gradient is Lipschitz.
If f is smooth, then `gradient!(y, f, x)` is expected to be defined, and it should return the value of f at x and store the gradient in y.
"""
is_smooth(::Type) = false
is_smooth(::T) where T = is_smooth(T)

"""
    is_quadratic(T::Type)

Returns `true` if the type `T` represents a quadratic function.
A function f(x) is quadratic if it is smooth and its Hessian is constant.
"""
is_quadratic(T::Type) = is_generalized_quadratic(T) && is_smooth(T)
is_quadratic(::T) where T = is_quadratic(T)

"""
    is_strongly_convex(T::Type)

Returns `true` if the type `T` represents a strongly convex function.
A function f(x) is strongly convex if it is convex and there exists a positive constant μ such that f(x) - μ/2 * ||x||^2 is convex.
"""
is_strongly_convex(::Type) = false
is_strongly_convex(::T) where T = is_strongly_convex(T)
