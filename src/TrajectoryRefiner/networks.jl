using Flux
using ..SequenceTransform: SequenceTransformation

"""
    CorrectionNetwork

A network that computes intermediate and terminal corrections for trajectory refinement.

# Fields
- `inter_net`: Network for intermediate correction (dynamics).
- `term_net`: Network for terminal correction (constraints).
"""
struct CorrectionNetwork{I, T}
    inter_net::I
    term_net::T
end

"""
    CorrectionNetwork(state_dim::Int, input_dim::Int, hidden_dim::Int, out_dim::Int, depth::Int, context_dim::Int=0, activation=relu)

Constructs a `CorrectionNetwork`.

# Arguments
- `state_dim`: Dimension of the state vector.
- `input_dim`: Dimension of the input vector.
- `hidden_dim`: Hidden dimension for the internal layers.
- `out_dim`: Output dimension (usually state_dim + input_dim).
- `depth`: Depth of the `SequenceTransformation` chains.
- `context_dim`: Dimension of the context vector (x_0).
- `activation`: Activation function.
"""
function CorrectionNetwork(state_dim::Int, input_dim::Int, hidden_dim::Int, out_dim::Int, depth::Int, context_dim::Int=0, activation=relu)
    # Inputs to the networks are concatenated: x_res, x_guess, u_guess
    # Dimensions: state_dim + state_dim + input_dim
    net_in_dim = 2 * state_dim + input_dim
    
    # Intermediate network
    inter_net = SequenceTransformation(net_in_dim, hidden_dim, out_dim, depth, context_dim, activation)
    
    # Terminal network
    term_net = SequenceTransformation(net_in_dim, hidden_dim, out_dim, depth, context_dim, activation)
    
    return CorrectionNetwork(inter_net, term_net)
end

Flux.@layer CorrectionNetwork

"""
    (m::CorrectionNetwork)(x_res, x_guess, u_guess, x_0)

Forward pass of the correction network.

# Arguments
- `x_res`: Residual state trajectory (from multiple shooting). Shape: (state_dim, seq_len, batch)
- `x_guess`: Guess state trajectory. Shape: (state_dim, seq_len, batch)
- `u_guess`: Guess input trajectory. Shape: (input_dim, seq_len, batch)
- `x_0`: Initial state context. Shape: (state_dim, batch)

# Returns
- `delta_inter`: Intermediate correction (unscaled).
- `out_term`: Terminal correction output (to be scaled by violation).
"""
function (m::CorrectionNetwork)(x_res::AbstractArray, x_guess::AbstractArray, u_guess::AbstractArray, x_0::AbstractArray)
    # Concatenate inputs along feature dimension
    # x_res: (S, T, B)
    # x_guess: (S, T, B)
    # u_guess: (U, T, B)
    # net_input: (2S+U, T, B)
    net_input = cat(x_res, x_guess, u_guess, dims=1)
    
    # Compute intermediate correction terms
    # Term 1: Network(x_res, x_guess, u_guess)
    out_inter_res = m.inter_net(net_input, x_0)
    
    # Term 2: Network(x_guess, x_guess, u_guess)
    # We replace x_res with x_guess in the input
    net_input_guess = cat(x_guess, x_guess, u_guess, dims=1)
    out_inter_guess = m.inter_net(net_input_guess, x_0)
    
    delta_inter = out_inter_res - out_inter_guess
    
    # Compute terminal correction output
    out_term = m.term_net(net_input, x_0)
    
    return delta_inter, out_term
end
