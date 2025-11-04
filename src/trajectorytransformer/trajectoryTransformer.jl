using Flux
using Flux: Dense, MultiHeadAttention

_colmat(x::AbstractVector) = reshape(Float32.(x), :, 1)
_colmat(x::AbstractMatrix) = Float32.(x)

_to3d(x::AbstractMatrix) = reshape(x, size(x, 1), size(x, 2), 1)
_from3d(x) = reshape(x, size(x, 1), size(x, 2))

"""
    TrajectoryTransformer{LP,CP,MH,FF1,FF2,SH,LH}

Container for the learnable blocks that convert latent sequences into state trajectories conditioned on the current state and goal.

# Fields
- `latent_proj::LP`: projection applied to each latent token to obtain the model embedding.
- `context_proj::CP`: projection of the concatenated current-state/goal context token.
- `attention::MH`: multi-head attention block operating on latent and context tokens.
- `ff1::FF1`, `ff2::FF2`: position-wise feed-forward layers applied after attention.
- `state_head::SH`: decoder mapping embeddings to state vectors.
- `latent_head::LH`: decoder generating next-step latent tokens for recurrence.
- `state_dim::Int`, `goal_dim::Int`: dimensionalities of state and goal inputs.
- `latent_dim::Int`: dimensionality of the latent tokens supplied per step.
- `embed_dim::Int`: internal embedding size.
- `heads::Int`: number of attention heads in the transformer block.
"""
struct TrajectoryTransformer{LP,CP,MH,FF1,FF2,SH,LH}
    latent_proj::LP
    context_proj::CP
    attention::MH
    ff1::FF1
    ff2::FF2
    state_head::SH
    latent_head::LH
    state_dim::Int
    goal_dim::Int
    latent_dim::Int
    embed_dim::Int
    heads::Int
end

Flux.@layer TrajectoryTransformer

"""
    TrajectoryTransformer(state_dim, goal_dim, latent_dim; kwargs...)

Construct a learnable transformer that maps latent sequences to state trajectories conditioned on the current state and goal.

# Arguments
- `state_dim::Integer`: dimensionality of the current state.
- `goal_dim::Integer`: dimensionality of the goal signal.
- `latent_dim::Integer`: dimensionality of the latent tokens provided as input.

# Keyword Arguments
- `embed_dim::Integer=64`: embedding size of the transformer tokens.
- `heads::Integer=4`: number of attention heads.
- `ff_hidden::Integer=128`: hidden width of the feed-forward block.
- `activation`: activation function used in projection and feed-forward layers (default `Flux.relu`).

# Returns
`TrajectoryTransformer` whose fields expose the internal projection, attention, and decoding layers.
"""
function TrajectoryTransformer(state_dim::Integer,
                               goal_dim::Integer,
                               latent_dim::Integer;
                               embed_dim::Integer=64,
                               heads::Integer=4,
                               ff_hidden::Integer=128,
                               activation=Flux.relu)
    latent_proj = Dense(latent_dim, embed_dim, activation)
    context_proj = Dense(state_dim + goal_dim, embed_dim, activation)
    attention = MultiHeadAttention(embed_dim => embed_dim => embed_dim,
                                   nheads=heads, dropout_prob=0f0, bias=false)
    ff1 = Dense(embed_dim, ff_hidden, activation)
    ff2 = Dense(ff_hidden, embed_dim, identity)
    state_head = Dense(embed_dim, state_dim, identity)
    latent_head = Dense(embed_dim, latent_dim, identity)
    return TrajectoryTransformer(latent_proj, context_proj, attention,
                                 ff1, ff2, state_head, latent_head,
                                 state_dim, goal_dim, latent_dim,
                                 embed_dim, heads)
end

function _transform_once(net::TrajectoryTransformer,
                         current_state,
                         goal,
                         latents::AbstractMatrix)::Tuple{Matrix{Float32},Matrix{Float32}}
    latent_seq = Float32.(latents)
    latent_embed = net.latent_proj(latent_seq)

    state_vec = vec(_colmat(current_state))
    goal_vec = vec(_colmat(goal))
    context_vec = vcat(state_vec, goal_vec)
    context_token = net.context_proj(_colmat(context_vec))

    combined = hcat(latent_embed, context_token)

    latent_embed_3d = reshape(latent_embed, size(latent_embed, 1), size(latent_embed, 2), 1)
    combined_3d = reshape(combined, size(combined, 1), size(combined, 2), 1)

    attn_out, _ = net.attention(latent_embed_3d, combined_3d, combined_3d)
    attn_flat = reshape(attn_out, size(attn_out, 1), size(attn_out, 2))

    hidden = net.ff2(net.ff1(attn_flat))
    state_seq = net.state_head(hidden)
    latent_seq_next = net.latent_head(hidden)

    return state_seq, latent_seq_next
end

"""
    transform_sequence(net, current_state, goal, latents; steps=1)

Propagate a latent sequence through the transformer for a number of recurrent refinements.

# Arguments
- `net::TrajectoryTransformer`: trained or initialised network.
- `current_state`: vector or matrix representing the current system state; columns correspond to batch entries.
- `goal`: vector or matrix describing the target; must share the batch size with `current_state`.
- `latents`: matrix `(latent_dim, sequence_length)` (or vector) supplying the initial latent trajectory.

# Keyword Arguments
- `steps::Integer=1`: number of recurrent decoding passes to perform.
- `return_history::Bool=false`: when `true`, retain and return intermediate state/latent sequences for each recurrence step.

# Returns
NamedTuple containing:
- `states::Matrix{Float32}`: final decoded state sequence `(state_dim, sequence_length)`.
- `latents::Matrix{Float32}`: latent sequence after the last recurrence `(latent_dim, sequence_length)`.
- `state_history`: either an empty `Vector{Matrix{Float32}}` when `return_history=false`, or the per-step state sequences when enabled.
- `latent_history`: analogous to `state_history`, holding latent sequences when requested.
"""
function transform_sequence(net::TrajectoryTransformer,
                            current_state,
                            goal,
                            latents;
                            steps::Integer=1,
                            return_history::Bool=false)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    latent_mat = _colmat(latents)
    states_history = return_history ? Vector{Matrix{Float32}}(undef, steps) : Matrix{Float32}[]
    latent_history = return_history ? Vector{Matrix{Float32}}(undef, steps) : Matrix{Float32}[]
    current_latent = latent_mat
    state_seq = Matrix{Float32}(undef, net.state_dim, size(latent_mat, 2))
    next_latent = latent_mat
    for step in 1:steps
        state_seq, next_latent = _transform_once(net, current_state, goal, current_latent)
        if return_history
            states_history[step] = state_seq
            latent_history[step] = next_latent
        end
        current_latent = next_latent
    end
    final_states = return_history ? states_history[end] : state_seq
    return (states=final_states,
            latents=current_latent,
            state_history=states_history,
            latent_history=latent_history)
end

"""
    predict_state_sequence(net, current_state, goal, latents; steps=1)

Return only the decoded state sequence after optionally multiple recurrent passes.

# Arguments / Keywords
Same as [`transform_sequence`](@ref).

# Returns
`Matrix{Float32}` containing the state sequence `(state_dim, sequence_length)`.
"""
function predict_state_sequence(net::TrajectoryTransformer,
                                current_state,
                                goal,
                                latents;
                                steps::Integer=1)
    result = transform_sequence(net, current_state, goal, latents;
                                steps=steps, return_history=false)
    return result.states
end

"""
    predict_latent_sequence(net, current_state, goal, latents; steps=1)

Return the latent sequence emitted after recurrent refinement.

# Arguments / Keywords
Same as [`transform_sequence`](@ref).

# Returns
`Matrix{Float32}` containing the updated latent sequence `(latent_dim, sequence_length)`.
"""
function predict_latent_sequence(net::TrajectoryTransformer,
                                 current_state,
                                 goal,
                                 latents;
                                 steps::Integer=1)
    result = transform_sequence(net, current_state, goal, latents;
                                steps=steps, return_history=false)
    return result.latents
end
