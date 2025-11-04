using Flux
using ..GANModels: Gan, generator_forward

_colmat(x::AbstractVector) = reshape(x, :, 1)
_colmat(x::AbstractMatrix) = x

function _latent_matrix(gan::Gan, latent, batch::Integer)
    if latent === nothing
        return zeros(Float32, gan.latent_dim, batch)
    else
        mat = _colmat(latent)
        @assert size(mat, 1) == gan.latent_dim "latent dimension mismatch"
        @assert size(mat, 2) == batch "latent batch mismatch"
        return Float32.(mat)
    end
end

function _time_features(dim::Integer, step::Integer, total::Integer, batch::Integer, T::Type)
    @assert total > 0 "total_time must be positive"
    @assert 1 <= step <= total "time_step must be within [1, total_time]"
    features = zeros(T, dim, batch)
    norm_step = T(step) / T(total)
    if dim >= 1
        features[1, :] .= norm_step
    end
    if dim >= 2
        features[2, :] .= T(step)
    end
    if dim >= 3
        features[3, :] .= T(total)
    end
    if dim >= 4
        features[4, :] .= T(total - step)
    end
    return features
end

struct GANControlNet{TG,IG,CG}
    terminal::TG
    intermediate::IG
    controller::CG
    state_dim::Int
    goal_dim::Int
    control_dim::Int
    time_feature_dim::Int
end

function GANControlNet(state_dim::Integer, goal_dim::Integer, control_dim::Integer;
                       terminal_latent::Integer=8,
                       intermediate_latent::Integer=8,
                       control_latent::Integer=8,
                       time_feature_dim::Integer=2,
                       terminal_kwargs::NamedTuple=NamedTuple(),
                       intermediate_kwargs::NamedTuple=NamedTuple(),
                       control_kwargs::NamedTuple=NamedTuple())
    time_feature_dim >= 0 || throw(ArgumentError("time_feature_dim must be non-negative"))

    term_context_dim = state_dim + goal_dim
    intermediate_context_dim = 2 * state_dim + time_feature_dim
    control_context_dim = 2 * state_dim

    terminal_gan = Gan(terminal_latent, term_context_dim, state_dim; terminal_kwargs...)
    intermediate_gan = Gan(intermediate_latent, intermediate_context_dim, state_dim; intermediate_kwargs...)
    controller_gan = Gan(control_latent, control_context_dim, control_dim; control_kwargs...)

    return GANControlNet(terminal_gan, intermediate_gan, controller_gan,
                         state_dim, goal_dim, control_dim, time_feature_dim)
end

function predict_terminal_state(net::GANControlNet,
                                current_state::AbstractVecOrMat,
                                goal::AbstractVecOrMat;
                                latent=nothing)
    state = _colmat(current_state)
    goal_mat = _colmat(goal)
    @assert size(state, 1) == net.state_dim "current_state dimension mismatch"
    @assert size(goal_mat, 1) == net.goal_dim "goal dimension mismatch"
    @assert size(state, 2) == size(goal_mat, 2) "state/goal batch mismatch"

    batch = size(state, 2)
    context = vcat(Float32.(state), Float32.(goal_mat))
    latent_mat = _latent_matrix(net.terminal, latent, batch)
    return generator_forward(net.terminal, context, latent_mat)
end

function predict_state_at(net::GANControlNet,
                          current_state::AbstractVecOrMat,
                          terminal_state::AbstractVecOrMat,
                          time_step::Integer,
                          total_time::Integer;
                          latent=nothing)
    state = _colmat(current_state)
    terminal = _colmat(terminal_state)
    @assert size(state, 1) == net.state_dim "current_state dimension mismatch"
    @assert size(terminal, 1) == net.state_dim "terminal_state dimension mismatch"
    @assert size(state, 2) == size(terminal, 2) "state/terminal batch mismatch"

    batch = size(state, 2)
    T = Float32
    time_mat = net.time_feature_dim == 0 ?
        zeros(T, 0, batch) :
        _time_features(net.time_feature_dim, time_step, total_time, batch, T)

    state_f = Float32.(state)
    terminal_f = Float32.(terminal)
    context = vcat(state_f, terminal_f, time_mat)
    latent_mat = _latent_matrix(net.intermediate, latent, batch)
    rate = generator_forward(net.intermediate, context, latent_mat)
    factor = time_step <= 0 ? 1f0 : Float32(time_step)
    return state_f .+ rate .* factor
end

function predict_control_input(net::GANControlNet,
                               current_state::AbstractVecOrMat,
                               next_state::AbstractVecOrMat;
                               latent=nothing)
    state = _colmat(current_state)
    next = _colmat(next_state)
    @assert size(state, 1) == net.state_dim "current_state dimension mismatch"
    @assert size(next, 1) == net.state_dim "next_state dimension mismatch"
    @assert size(state, 2) == size(next, 2) "state/next batch mismatch"

    batch = size(state, 2)
    context = vcat(Float32.(state), Float32.(next))
    latent_mat = _latent_matrix(net.controller, latent, batch)
    return generator_forward(net.controller, context, latent_mat)
end

function predict_control(net::GANControlNet,
                         current_state::AbstractVecOrMat,
                         goal::AbstractVecOrMat,
                         time_step::Integer,
                         total_time::Integer;
                         latent_terminal=nothing,
                         latent_intermediate=nothing,
                         latent_control=nothing)
    terminal_state = predict_terminal_state(net, current_state, goal; latent=latent_terminal)
    intermediate_state = predict_state_at(net, current_state, terminal_state,
                                          time_step, total_time;
                                          latent=latent_intermediate)
    control = predict_control_input(net, current_state, intermediate_state;
                                    latent=latent_control)
    return (terminal_state=terminal_state,
            intermediate_state=intermediate_state,
            control=control)
end
