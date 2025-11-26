using Test
using Flux
using Random
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
    
    @testset "refine" begin
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
        
        # Mock terminal cost: negative when x[1] > 0.5 (satisfied), positive otherwise.
        term_cost_fn(x) = 0.5f0 .- x[1:1, :]
        

        
        batch_size = 2
        seq_len = 5
        
        x_guess = rand(Float32, state_dim, seq_len, batch_size)
        u_guess = rand(Float32, input_dim, seq_len, batch_size)
        x_0 = rand(Float32, state_dim, batch_size)
        
        x_new, u_new = refine(net, transition, term_cost_fn, x_guess, u_guess, x_0)
        
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
        # x_valid end state is [1, 1], term_cost = -0.5 < 0. Satisfied.
        
        # Initialize network with small weights to ensure small noise doesn't blow up?
        # Actually, delta_inter should be exactly zero because x_res will equal x_guess.
        # delta_term should be zero because violation is zero.
        
        x_new_valid, u_new_valid = refine(net, transition, term_cost_fn, x_valid, u_valid, x_0_valid)
        
        @test x_new_valid ≈ x_valid
        @test u_new_valid ≈ u_valid
        
    end
    
    @testset "refine recursion" begin
        Random.seed!(42)
        state_dim = 2
        input_dim = 1
        hidden_dim = 10
        out_dim = state_dim + input_dim
        depth = 2
        context_dim = state_dim
        
        net = CorrectionNetwork(state_dim, input_dim, hidden_dim, out_dim, depth, context_dim)
        
        struct MockTransition
        end
        (m::MockTransition)(x, u) = x + vcat(u, u)
        transition = MockTransition()
        
        term_cost_fn(x) = 0.5f0 .- x[1:1, :]
        
        batch_size = 2
        seq_len = 4
        x_guess = rand(Float32, state_dim, seq_len, batch_size)
        u_guess = rand(Float32, input_dim, seq_len, batch_size)
        x_0 = rand(Float32, state_dim, batch_size)
        
        # Manual two-step refinement
        x1, u1 = refine(net, transition, term_cost_fn, x_guess, u_guess, x_0)
        x2, u2 = refine(net, transition, term_cost_fn, x1, u1, x_0)
        
        # Recursive call should match manual chaining
        x_rec, u_rec = refine(net, transition, term_cost_fn, x_guess, u_guess, x_0, 2)
        
        @test x_rec ≈ x2
        @test u_rec ≈ u2
    end
    
    @testset "refinement_loss" begin
        # Use a zero network so refinement leaves trajectories unchanged.
        struct ZeroNetwork
            out_dim::Int
        end
        (m::ZeroNetwork)(x_res, x_guess, u_guess, x_0) = (zeros(Float32, m.out_dim, size(x_guess, 2), size(x_guess, 3)),
                                                           zeros(Float32, m.out_dim, size(x_guess, 2), size(x_guess, 3)))
        
        state_dim = 1
        input_dim = 1
        out_dim = state_dim + input_dim
        net = ZeroNetwork(out_dim)
        
        # Simple transition: x_next = x_prev + u
        transition_fn(x_prev, u) = x_prev .+ u
        
        # Terminal cost: sum of terminal state
        term_cost_fn(x_term) = sum(x_term)
        
        # Mismatch: squared error sum between residual and guess
        mismatch_fn(x_res, x_guess) = sum(abs2.(x_res .- x_guess))
        
        # Deterministic inputs
        seq_len = 2
        batch = 1
        x_guess_init = fill(1f0, state_dim, seq_len, batch)
        u_guess_init = fill(2f0, input_dim, seq_len, batch)
        x_0 = fill(0f0, state_dim, batch)
        
        # Expected residual with zero-network (no correction): x_prev = [0, 1], u = [2, 2] => x_res = [2, 3]
        # Mismatch = (2-1)^2 + (3-1)^2 = 1 + 4 = 5
        # Terminal cost = sum of terminal x_guess (1) = 1
        # Total = 6
        loss_val = refinement_loss(net, transition_fn, term_cost_fn, mismatch_fn, x_guess_init, u_guess_init, x_0, 1)
        @test loss_val ≈ 6f0
    end
end
