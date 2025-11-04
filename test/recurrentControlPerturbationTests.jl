using Test
using Random
using ReachabilityCascade
using ReachabilityCascade: RecurrentControlNet, train_recurrent_control_perturb!,
    TerminalGradientDatum, IntermediateGradientDatum, ControlGradientDatum,
    load_recurrent_control
using Flux: state

@testset "Recurrent Control Perturbation Trainer" begin
    data_rng = MersenneTwister(42)
    perturb_rng = MersenneTwister(43)
    state_dim, goal_dim, control_dim = 2, 1, 1
    net = RecurrentControlNet(state_dim, goal_dim, control_dim;
                              recurrence_steps=2,
                              time_embed_dim=2,
                              terminal_kwargs=(n_blocks=1,),
                              intermediate_kwargs=(n_blocks=1,),
                              control_kwargs=(n_blocks=1,))

    batch = 2
    term_samples_1 = randn(data_rng, Float32, state_dim, batch)
    term_samples_2 = randn(data_rng, Float32, state_dim, batch)
    current_states_1 = randn(data_rng, Float32, state_dim, batch)
    current_states_2 = randn(data_rng, Float32, state_dim, batch)
    goals_1 = randn(data_rng, Float32, goal_dim, batch)
    goals_2 = randn(data_rng, Float32, goal_dim, batch)
    terminal_data = [
        TerminalGradientDatum(term_samples_1, current_states_1, goals_1),
        TerminalGradientDatum(term_samples_2, current_states_2, goals_2)
    ]

    intermediate_samples_1 = randn(data_rng, Float32, state_dim, batch)
    intermediate_samples_2 = randn(data_rng, Float32, state_dim, batch)
    times = fill(1, batch)
    totals = fill(2, batch)
    intermediate_data = [
        IntermediateGradientDatum(intermediate_samples_1, current_states_1, term_samples_1, times, totals),
        IntermediateGradientDatum(intermediate_samples_2, current_states_2, term_samples_2, times, totals)
    ]

    control_samples_1 = randn(data_rng, Float32, control_dim, batch)
    control_samples_2 = randn(data_rng, Float32, control_dim, batch)
    control_data = [
        ControlGradientDatum(control_samples_1, current_states_1, intermediate_samples_1),
        ControlGradientDatum(control_samples_2, current_states_2, intermediate_samples_2)
    ]

    results = Dict(:terminal => Any[], :intermediate => Any[], :control => Any[])
    callback = function(component, result, epoch)
        push!(results[component], (epoch=epoch, result=result))
    end

    returned_net = train_recurrent_control_perturb!(net,
                                                    terminal_data,
                                                    intermediate_data,
                                                    control_data;
                                                    epochs=1,
                                                    perturb_scale=1e-3,
                                                    rng=perturb_rng,
                                                    carryover_limit=1,
                                                    callback=callback)

    @test returned_net === net

    for key in (:terminal, :intermediate, :control)
        @test length(results[key]) == 2
        for entry in results[key]
            res = entry.result
            @test res.accepted isa Bool
            @test res.previous_loglikelihood isa Real
            @test res.new_loglikelihood isa Real
            @test res.new_loglikelihood >= res.previous_loglikelihood
            @test res.perturb_norm >= 0
        end
    end

    save_path = tempname() * ".jld2"
    net_ckpt = RecurrentControlNet(state_dim, goal_dim, control_dim;
                                   recurrence_steps=2,
                                   time_embed_dim=2,
                                   terminal_kwargs=(n_blocks=1,),
                                   intermediate_kwargs=(n_blocks=1,),
                                   control_kwargs=(n_blocks=1,))
    constructor_kwargs = (recurrence_steps=2,
                          time_embed_dim=2,
                          terminal_kwargs=(n_blocks=1,),
                          intermediate_kwargs=(n_blocks=1,),
                          control_kwargs=(n_blocks=1,))
    constructor_info = Dict("args" => (state_dim, goal_dim, control_dim),
                            "kwargs" => constructor_kwargs)

    train_recurrent_control_perturb!(net_ckpt,
                                     terminal_data,
                                     intermediate_data,
                                     control_data;
                                     epochs=1,
                                     perturb_scale=1e-3,
                                     rng=MersenneTwister(99),
                                     carryover_limit=1,
                                     save_path=save_path,
                                     load_path=save_path,
                                     constructor_info=constructor_info,
                                     save_interval=Inf)

    @test isfile(save_path)
    reloaded = load_recurrent_control(save_path)
    @test state(reloaded) == state(net_ckpt)
    rm(save_path; force=true)
end
