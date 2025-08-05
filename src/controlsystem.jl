"""
Specification of continuous time control system.

# Fields
- X :: LazySet - state space.
- U :: LazySet - set of control inputs.
- W :: LazySet - set of disturbances.
- f :: Function - vector field, ``f:X\\times U\\times W\\rightarrow \\mathbb{R}^n`` where ``n`` is the dimension of ``X``.
"""
struct ContinuousSystem
    X::LazySet
    U::LazySet
    W::LazySet
    f::Function
end

"""
    ContinuousSystem(X::Hyperrectangle, U::LazySet, f::Function; W::LazySet = Hyperrectangle([0.0], [0.0]))	

Constructor for continuous system.

# Args
- X :: Hyperrectangle - State space.
- U :: LazySet - set of control inputs.
- f :: Function - transition function ``f:X\\times U\\times W\\rightarrow \\mathbb{R}^{dim}``.
- W :: LazySet [Optional Named] - set of disturbances.
"""
function ContinuousSystem(X::Hyperrectangle, U::LazySet, f::Function)
    W = Hyperrectangle([0.0], [0.0])
    ContinuousSystem(X, U, W, f)
end

"""
    ContinuousSystem(dim::Integer, U::LazySet, f::Function; W::LazySet = Hyperrectangle([0.0], [0.0]))	

Constructor for continuous system.

# Args
- dim :: Integer - dimension of state space.
- U :: LazySet - set of control inputs.
- f :: Function - transition function ``f:X\\times U\\times W\\rightarrow \\mathbb{R}^{dim}``.
- W :: LazySet [Optional Named] - set of disturbances.
"""
function ContinuousSystem(dim::Integer, U::LazySet, f::Function; W::LazySet = Hyperrectangle([0.0], [0.0]))
    X = Hyperrectangle(zeros(dim), Inf*ones(dim))
    ContinuousSystem(X, U, W, f)
end

"""
    (cs::ContinuousSystem)(x::Vector{<:Real}, u::Vector{<:Real}, dt::Real; w::Vector{<:Real} = LazySets.sample(cs.W))

Computes the next state after a very small time elapse given a constant control input and disturbance input.

# Args
- x :: Vector{<:Real} - current state.
- u :: Vector{<:Real} - control input.
- dt :: Real - time step size.
- w :: Vector{<:Real} [Optional Named] - disturbance.
"""
function (cs::ContinuousSystem)(x::Vector{<:Real}, u::Vector{<:Real}, dt::Real; w::Vector{<:Real} = LazySets.sample(cs.W))
    x + cs.f(x, u, w)*dt
end

"""
    (cs::ContinuousSystem)(x::Vector{<:Real}, umat::Matrix{<:Real}, dt::Real;
    wmat::Matrix{<:Real} = reduce(hcat, LazySets.sample(cs.W, size(umat, 2)))) :: Matrix{<:Real}

Computes the state trajectory of a control system given a sequence of control inputs and disturbances with a constant sampling period between successive inputs.

# Args
- x :: Vector{<:Real} - initial state.
- umat :: Matrix{<:Real} - sequence of control inputs as columns of umat.
- dt :: Real - sampling period between successive inputs.
- wmat :: Matrix{<:Real} [Optional Named] - sequence of disturbances.
"""
function (cs::ContinuousSystem)(x::Vector{<:Real}, umat::Matrix{<:Real}, dt::Real;
wmat::Matrix{<:Real} = reduce(hcat, LazySets.sample(cs.W, size(umat, 2)))) :: Matrix{<:Real}
    xmat = reshape(x, size(x, 1), 1)   
    for i in 1:size(umat, 2)
        new_state = cs(xmat[:, end], umat[:, i], dt; w=wmat[:,i])
        xmat = hcat(xmat, new_state)
    end
    xmat
end

"""
    (cs::ContinuousSystem)(x::Vector{<:Real}, v::Vector{<:Real}, κ::Function, τ::Real;
    dt::Real = 1f-3, 
    γ::Function = (x::Vector{<:Real}, v::Vector{<:Real}, t::Real) -> LazySets.sample(cs.W)
    )

Computes state trajectory and sequence of inputs for a given reference input, feedback function and time span.

# Args
- x :: Vector{<:Real} - initial state.
- v :: Vector{<:Real} - reference input.
- κ :: Function - feedback function of the form ``\\kappa:X\\times V\\times \\mathbb{R}_{\\geq 0} \\rightarrow U``, where ``X`` is the state space, ``V`` is the set of reference inputs, the non-negative real line ``\\mathbb{R}_{\\geq 0}`` represents the set of times and ``U`` is the set of control inputs.
- τ :: Real - time span of trajectory
- dt :: Real [Optional Named] - integration time step.
- γ :: Function [Optional Named] - disturbance function of the form ``\\gamma:X\\times V\\times \\mathbb{R}_{\\geq 0} \\rightarrow W``, where ``X`` is the state space, ``V`` is the set of reference inputs, the non-negative real line ``\\mathbb{R}_{\\geq 0}`` represents the set of times and ``W`` is the set of disturbances.
"""
function (cs::ContinuousSystem)(x::Vector{<:Real}, v::Vector{<:Real}, κ::Function, τ::Real;
dt::Real = 1f-3, 
γ::Function = (x::Vector{<:Real}, v::Vector{<:Real}, t::Real) -> LazySets.sample(cs.W)
)
    numsteps = Int(floor(τ/dt))
    xmat = x
    umat = Matrix{Real}(undef, size(cs.U.center, 1), 0)
    for i in 1:numsteps
        u = κ(xmat[:, end], v, i*dt)
        w = γ(x, v, i*dt)
        new_state = cs(xmat[:, end], u, dt; w=w)
        xmat = hcat(xmat, new_state)
        umat = hcat(umat, u)
    end
    xmat, umat
end

"""
Specification of discrete time control system with a possibly randomized transition function.

# Fields
- X :: LazySet - state space.
- U :: LazySet - set of control inputs.
- f :: Random function - transition function, ``f:X\\times U\\rightarrow X``.
"""
struct DiscreteRandomSystem
    X::LazySet
    U::LazySet
    f::Function
end

"""
    DiscreteRandomSystem(cs::ContinuousSystem, V::LazySet, κ::Function, τ::Real;
    dt::Real=1f-3, 
    γ::Function = (x::Vector{<:Real}, v::Vector{<:Real}, t::Real) -> LazySets.sample(cs.W)
    )

Constructor for randomized discrete system.  Time discretization of continuous system to a discrete time system with a randomized transition function.

# Args
- cs :: ContinuousSystem. 
- V :: LazySet - set of sampled control inputs.
- κ :: Function - feedback function applied during discretization.
- τ :: Real - size of time step of discretization.
- dt :: Real [Optional Named] - integration time step size.
- γ :: Function [Optional Named] - disturbance function of the form ``\\gamma:X\\times V\\times \\mathbb{R}_{\\geq 0} \\rightarrow W``, where ``X`` is the state space, ``V`` is the set of reference inputs, the non-negative real line ``\\mathbb{R}_{\\geq 0}`` represents the set of times and ``W`` is the set of disturbances of the continuous system.
"""
function DiscreteRandomSystem(cs::ContinuousSystem, V::LazySet, κ::Function, τ::Real;
dt::Real=1f-3, 
γ::Function = (x::Vector{<:Real}, v::Vector{<:Real}, t::Real) -> LazySets.sample(cs.W)
)
    f = (x::Vector{<:Real}, u::Vector{<:Real}) -> cs(x, u, κ, τ, dt = dt, γ=γ)[1][:, end]
    DiscreteRandomSystem(cs.X, V, f)
end

"""
    (ds::DiscreteRandomSystem)(x::Vector{<:Real}, u::Vector{<:Real})
    ds.transition(x, u)

Computes the next state of a discrete system with random dynamics, give the current state and control input.

# Args
- x :: Vector{<:Real} - current state of the system.
- u :: Vector{<:Real} - control input applied to the system.
"""
function (ds::DiscreteRandomSystem)(x::Vector{<:Real}, u::Vector{<:Real})
    ds.f(x, u)
end

"""
    (ds::DiscreteRandomSystem)(x::Vector{<:Real}, umat::Matrix{<:Real})

Computes the state trajectory of a discrete time system with randomized dynamics, given an initial state and a sequence of control inputs.

# Args
- x :: Vector{<:Real} - initial state.
- umat :: Matrix{<:Real} - sequence of control inputs as columns of a matrix.
"""
function (ds::DiscreteRandomSystem)(x::Vector{<:Real}, umat::Matrix{<:Real})
    xmat = x
    for u in eachcol(umat)
        new_state = ds(xmat[:, end], Vector(u))
        xmat = hcat(xmat, new_state)
    end
    xmat
end