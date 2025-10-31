using Test
using ReachabilityCascade
using ReachabilityCascade: terminal_flow_gradient, intermediate_flow_gradient,
    control_flow_gradient
using Flux: destructure

@testset "Recurrent Control Network" begin
    state_dim, goal_dim, control_dim = 3, 2, 1
    net = RecurrentControlNet(state_dim, goal_dim, control_dim;
                              terminal_steps=2,
                              intermediate_steps=2,
                              control_steps=2,
                              time_embed_dim=2,
                              terminal_kwargs=(n_blocks=1,),
                              intermediate_kwargs=(n_blocks=1,),
                              control_kwargs=(n_blocks=1,))

    state = randn(Float32, state_dim)
    goal = randn(Float32, goal_dim)
    result = predict_control(net, state, goal, 1, 2)

    @test size(result.terminal_state, 1) == state_dim
    @test size(result.intermediate_state, 1) == state_dim
    @test size(result.control, 1) == control_dim

    batch_state = randn(Float32, state_dim, 4)
    batch_goal = randn(Float32, goal_dim, 4)
    terminal_latent = zeros(Float32, state_dim, 4)
    intermediate_latent = zeros(Float32, state_dim, 4)
    control_latent = zeros(Float32, control_dim, 4)

    batch_result = predict_control(net, batch_state, batch_goal, 1, 2;
                                   latent_terminal=terminal_latent,
                                   latent_intermediate=intermediate_latent,
                                   latent_control=control_latent)
    @test size(batch_result.terminal_state) == (state_dim, 4)
    @test size(batch_result.intermediate_state) == (state_dim, 4)
    @test size(batch_result.control) == (control_dim, 4)

    term_samples = randn(Float32, state_dim, 4)
    term_grad = terminal_flow_gradient(net, term_samples, batch_state, batch_goal; num_lowest=2)
    @test term_grad.loss isa Real
    term_grad_vec, _ = destructure(term_grad.grads)
    @test !any(isnan, term_grad_vec)
    if all(iszero, term_grad_vec)
        @warn "Terminal flow gradient is zero for the current batch."
    end
    @test size(term_grad.sorted_terminal_samples, 2) == 2
    @test size(term_grad.sorted_current_states, 2) == 2
    @test size(term_grad.sorted_goals, 2) == 2

    time_steps = fill(1, 4)
    total_times = fill(2, 4)
    inter_samples = randn(Float32, state_dim, 4)
    inter_grad = intermediate_flow_gradient(net, inter_samples, batch_state, term_samples,
                                            time_steps, total_times; num_lowest=2)
    @test inter_grad.loss isa Real
    inter_grad_vec, _ = destructure(inter_grad.grads)
    @test !any(isnan, inter_grad_vec)
    if all(iszero, inter_grad_vec)
        @warn "Intermediate flow gradient is zero for the current batch."
    end
    @test size(inter_grad.sorted_intermediate_samples, 2) == 2
    @test size(inter_grad.sorted_current_states, 2) == 2
    @test size(inter_grad.sorted_terminal_states, 2) == 2
    @test length(inter_grad.sorted_time_steps) == 2
    @test length(inter_grad.sorted_total_times) == 2

    control_samples = randn(Float32, control_dim, 4)
    ctrl_grad = control_flow_gradient(net, control_samples, batch_state, inter_samples; num_lowest=2)
    @test ctrl_grad.loss isa Real
    ctrl_grad_vec, _ = destructure(ctrl_grad.grads)
    @test !any(isnan, ctrl_grad_vec)
    if all(iszero, ctrl_grad_vec)
        @warn "Control flow gradient is zero for the current batch."
    end
    @test size(ctrl_grad.sorted_control_samples, 2) == 2
    @test size(ctrl_grad.sorted_current_states, 2) == 2
    @test size(ctrl_grad.sorted_next_states, 2) == 2
end
