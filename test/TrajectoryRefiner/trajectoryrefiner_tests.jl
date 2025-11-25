using Test
using Flux
using ReachabilityCascade.TrajectoryRefiner
using ReachabilityCascade.SequenceTransform

@testset "TrajectoryRefiner Tests" begin
    
    @testset "CorrectionNetwork" begin
        state_dim = 2
        input_dim = 1
        hidden_dim = 10
        out_dim = state_dim + input_dim
        depth = 2
        context_dim = state_dim # x_0 is context
        
        net = CorrectionNetwork(state_dim, input_dim, hidden_dim, out_dim, depth, context_dim)
        
        batch_size = 3
        seq_len = 5
        
        x_res = rand(Float32, state_dim, seq_len, batch_size)
        x_guess = rand(Float32, state_dim, seq_len, batch_size)
        u_guess = rand(Float32, input_dim, seq_len, batch_size)
        x_0 = rand(Float32, state_dim, batch_size)
        
        delta_inter, out_term = net(x_res, x_guess, u_guess, x_0)
        
        @test size(delta_inter) == (out_dim, seq_len, batch_size)
        @test size(out_term) == (out_dim, seq_len, batch_size)
        
        # Test zero correction property
        # If x_res == x_guess, delta_inter should be zero
        delta_inter_zero, _ = net(x_guess, x_guess, u_guess, x_0)
        @test all(isapprox.(delta_inter_zero, 0, atol=1e-5))
    end
    
    @testset "RefinerSolver" begin
        state_dim = 2
        input_dim = 1
        hidden_dim = 10
        out_dim = state_dim + input_dim
        depth = 2
        context_dim = state_dim
        
        net = CorrectionNetwork(state_dim, input_dim, hidden_dim, out_dim, depth, context_dim)
        
        # Mock transition network: linear dynamics x_next = x + u
        # Input: (x_prev_seq, u_seq) -> x_res_seq
        # Dimensions: (state_dim, seq_len, batch)
        struct MockTransition
        end
        
        (m::MockTransition)(x, u) = x + vcat(u, u) # broadcast u to state dim
        
        transition = MockTransition()
        
        # Mock constraint function: x[1] > 0.5
        constraint_fn(x) = x[1:1, :] .- 0.5f0
        
        solver = RefinerSolver(net, transition, constraint_fn)
        
        batch_size = 2
        seq_len = 5
        
        x_guess = rand(Float32, state_dim, seq_len, batch_size)
        u_guess = rand(Float32, input_dim, seq_len, batch_size)
        x_0 = rand(Float32, state_dim, batch_size)
        
        x_new, u_new = step_refiner(solver, x_guess, u_guess, x_0)
        
        @test size(x_new) == (state_dim, seq_len, batch_size)
        @test size(u_new) == (input_dim, seq_len, batch_size)
        
        # Test that if dynamics are satisfied and constraint is satisfied, correction is near zero
        # Construct a valid trajectory
        x_valid = zeros(Float32, state_dim, seq_len, batch_size)
        u_valid = zeros(Float32, input_dim, seq_len, batch_size) # u=0
        
        # x_0 = [1, 1] (satisfies constraint > 0.5)
        x_0_valid = ones(Float32, state_dim, batch_size)
        
        # Simulate
        curr_x = x_0_valid
        for t in 1:seq_len
            curr_x = transition(curr_x, u_valid[:, t, :])
            x_valid[:, t, :] = curr_x
        end
        
        # x_valid should satisfy dynamics by construction
        # x_valid end state is [1, 1], constraint val = 0.5 > 0. Satisfied.
        
        # Initialize network with small weights to ensure small noise doesn't blow up?
        # Actually, delta_inter should be exactly zero because x_res will equal x_guess.
        # delta_term should be zero because violation is zero.
        
        x_new_valid, u_new_valid = step_refiner(solver, x_valid, u_valid, x_0_valid)
        
        @test x_new_valid ≈ x_valid
        @test u_new_valid ≈ u_valid
        
        # Test Flux.trainable
        params = Flux.trainable(solver)
        @test length(params) == 1
        @test params[1] === net
    end
end
