using Test
using ReachabilityCascade
using ReachabilityCascade: terminal_flow_gradient, intermediate_flow_gradient,
    control_flow_gradient, train_recurrent_control!, TerminalGradientDatum,
    IntermediateGradientDatum, ControlGradientDatum
using Flux: destructure, Adam, Descent, state
using JLD2: load

@testset "Recurrent Control Network" begin
    state_dim, goal_dim, control_dim = 3, 2, 1
    net = RecurrentControlNet(state_dim, goal_dim, control_dim;
                              recurrence_steps=2,
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
    term_hard = term_grad.hard_examples
    @test term_hard !== nothing
    @test size(term_hard.samples, 2) == 2
    @test size(term_hard.context, 1) == state_dim + goal_dim
    @test size(term_hard.context, 2) == 2

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
    inter_hard = inter_grad.hard_examples
    @test inter_hard !== nothing
    @test size(inter_hard.samples, 2) == 2
    @test size(inter_hard.context, 1) == 2 * state_dim + net.time_embed_dim
    @test size(inter_hard.context, 2) == 2

    control_samples = randn(Float32, control_dim, 4)
    ctrl_grad = control_flow_gradient(net, control_samples, batch_state, inter_samples; num_lowest=2)
    @test ctrl_grad.loss isa Real
    ctrl_grad_vec, _ = destructure(ctrl_grad.grads)
    @test !any(isnan, ctrl_grad_vec)
    if all(iszero, ctrl_grad_vec)
        @warn "Control flow gradient is zero for the current batch."
    end
    ctrl_hard = ctrl_grad.hard_examples
    @test ctrl_hard !== nothing
    @test size(ctrl_hard.samples, 2) == 2
    @test size(ctrl_hard.context, 1) == 2 * state_dim
    @test size(ctrl_hard.context, 2) == 2
end

@testset "Recurrent Control Trainer" begin
    state_dim, goal_dim, control_dim = 2, 1, 1
    net = RecurrentControlNet(state_dim, goal_dim, control_dim;
                              recurrence_steps=2,
                              time_embed_dim=2,
                              terminal_kwargs=(n_blocks=1,),
                              intermediate_kwargs=(n_blocks=1,),
                              control_kwargs=(n_blocks=1,))

    batch = 1
    term_samples_1 = randn(Float32, state_dim, batch)
    term_samples_2 = randn(Float32, state_dim, batch)
    current_states_1 = randn(Float32, state_dim, batch)
    current_states_2 = randn(Float32, state_dim, batch)
    goals_1 = randn(Float32, goal_dim, batch)
    goals_2 = randn(Float32, goal_dim, batch)
    terminal_data = [
        (TerminalGradientDatum(term_samples_1, current_states_1, goals_1), (; num_lowest=1)),
        TerminalGradientDatum(term_samples_2, current_states_2, goals_2)
    ]

    intermediate_samples_1 = randn(Float32, state_dim, batch)
    intermediate_samples_2 = randn(Float32, state_dim, batch)
    times = fill(1, batch)
    totals = fill(2, batch)
    intermediate_data = [
        (IntermediateGradientDatum(intermediate_samples_1, current_states_1, term_samples_1, times, totals), (; num_lowest=1)),
        IntermediateGradientDatum(intermediate_samples_2, current_states_2, term_samples_2, times, totals)
    ]

    control_samples_1 = randn(Float32, control_dim, batch)
    control_samples_2 = randn(Float32, control_dim, batch)
    control_data = [
        (ControlGradientDatum(control_samples_1, current_states_1, intermediate_samples_1), (; num_lowest=1)),
        ControlGradientDatum(control_samples_2, current_states_2, intermediate_samples_2)
    ]

    results = Dict(:terminal => Any[], :intermediate => Any[], :control => Any[])
    callback = function(component, result, _epoch)
        push!(results[component], result)
    end
    save_path = tempname() * ".jld2"

    trained_net = train_recurrent_control!(net,
                                          terminal_data,
                                          intermediate_data,
                                          control_data,
                                          Adam(1e-3);
                                          epochs=1,
                                          callback=callback,
                                          save_path=save_path,
                                          save_interval=0.0)

    @test length(results[:terminal]) == 2
    @test length(results[:intermediate]) == 2
    @test length(results[:control]) == 2

    term_second = results[:terminal][2]
    @test size(term_second.transitions[1].input, 2) == 2
    inter_second = results[:intermediate][2]
    @test size(inter_second.transitions[1].input, 2) == 2
    ctrl_second = results[:control][2]
    @test size(ctrl_second.transitions[1].input, 2) == 2

    @test trained_net === net
    @test isfile(save_path)

    saved_payload = load(save_path)
    saved_state = saved_payload["model_state"]

    reload_net = RecurrentControlNet(state_dim, goal_dim, control_dim;
                                     recurrence_steps=2,
                                     time_embed_dim=2,
                                     terminal_kwargs=(n_blocks=1,),
                                     intermediate_kwargs=(n_blocks=1,),
                                     control_kwargs=(n_blocks=1,))

    train_recurrent_control!(reload_net,
                             terminal_data,
                             intermediate_data,
                             control_data,
                             Descent(0.0);
                             epochs=1,
                             save_path="",
                             load_path=save_path)

    @test state(reload_net) == saved_state

    rm(save_path; force=true)
end

@testset "Data-driven Constructor" begin
    state_dim, goal_dim, control_dim = 2, 1, 1
    batch = 2
    terminal_samples = randn(Float32, state_dim, batch)
    current_states = randn(Float32, state_dim, batch)
    goals = randn(Float32, goal_dim, batch)
    terminal_data = [
        (TerminalGradientDatum(terminal_samples, current_states, goals), (; num_lowest=1))
    ]

    intermediate_samples = randn(Float32, state_dim, batch)
    times = fill(1, batch)
    totals = fill(2, batch)
    intermediate_data = [
        (IntermediateGradientDatum(intermediate_samples, current_states, terminal_samples, times, totals), (; num_lowest=1))
    ]

    control_samples = randn(Float32, control_dim, batch)
    control_data = [
        (ControlGradientDatum(control_samples, current_states, intermediate_samples), (; num_lowest=1))
    ]

    save_path = tempname() * ".jld2"
    net = RecurrentControlNet(terminal_data,
                              intermediate_data,
                              control_data,
                              Adam(1e-3);
                              epochs=1,
                              recurrence_steps=2,
                              save_path=save_path,
                              load_path=save_path,
                              save_interval=0.0)
    @test net.state_dim == state_dim
    @test net.goal_dim == goal_dim
    @test net.control_dim == control_dim
    @test isfile(save_path)

    payload = load(save_path)
    stored_state = payload["model_state"]

    net_reload = RecurrentControlNet(terminal_data,
                                     intermediate_data,
                                     control_data,
                                     Descent(0.0);
                                     epochs=1,
                                     recurrence_steps=2,
                                     save_path="",
                                     load_path=save_path)

    @test state(net_reload) == stored_state
    rm(save_path; force=true)
end
