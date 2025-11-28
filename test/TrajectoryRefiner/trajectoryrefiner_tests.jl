using Test
using Flux
using Random
using ReachabilityCascade.TrajectoryRefiner
using ReachabilityCascade.SequenceTransform

@testset "TrajectoryRefiner Tests" begin
    
    @testset "RefinementModel" begin
        state_dim = 2
        input_dim = 1
        hidden_dim = 10
        out_dim = state_dim + input_dim
        depth = 2
        
        net = RefinementModel(state_dim, input_dim, hidden_dim, out_dim, depth)
        
        batch_size = 3
        seq_len = 5
        
        # Build a bundle where the residual equals the guess so refinement is identity.
        x0 = rand(Float32, state_dim, batch_size)
        x_body = rand(Float32, state_dim, seq_len, batch_size)
        u_guess = rand(Float32, input_dim, seq_len, batch_size)
        x_full = cat(reshape(x0, state_dim, 1, batch_size), x_body; dims=2)
        bundle = ShootingBundle(x_full, u_guess)

        # Transition returns the provided guess body so x_res == x_body.
        struct IdentityTransition end
        (m::IdentityTransition)(x, u) = x  # ignore u to mirror x_prev_seq
        transition = IdentityTransition()

        # Trajectory cost is zero so no correction comes from it.
        traj_cost_fn(x) = zero(eltype(x))

        refined = net(bundle, transition, traj_cost_fn)

        @test size(refined.x_guess) == (state_dim, seq_len + 1, batch_size)
        @test size(refined.u_guess) == (input_dim, seq_len, batch_size)
        @test refined.x_guess ≈ x_full
        @test refined.u_guess ≈ u_guess
    end
    
    @testset "refine" begin
        state_dim = 2
        input_dim = 1
        hidden_dim = 10
        out_dim = state_dim + input_dim
        depth = 2
        
        net = RefinementModel(state_dim, input_dim, hidden_dim, out_dim, depth)
        
        # Mock transition network: linear dynamics x_next = x + u
        # Input: (x_prev_seq, u_seq) -> x_res_seq
        # Dimensions: (state_dim, seq_len, batch)
        struct MockTransition
        end
        
        (m::MockTransition)(x, u) = x + vcat(u, u) # broadcast u to state dim
        
        transition = MockTransition()
        
        # Mock terminal cost: negative when x[1] > 0.5 (satisfied), positive otherwise.
        traj_cost_fn(x_res) = sum(0.5f0 .- x_res[1, :, :])
        

        
        batch_size = 2
        seq_len = 5
        
        x0 = rand(Float32, state_dim, batch_size)
        x_guess_body = rand(Float32, state_dim, seq_len, batch_size)
        u_guess = rand(Float32, input_dim, seq_len, batch_size)
        x_full = cat(reshape(x0, state_dim, 1, batch_size), x_guess_body; dims=2)
        bundle = ShootingBundle(x_full, u_guess)
        
        refined = net(bundle, transition, traj_cost_fn)
        
        @test size(refined.x_guess) == (state_dim, seq_len + 1, batch_size)
        @test size(refined.u_guess) == (input_dim, seq_len, batch_size)
        
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
        
        valid_bundle = ShootingBundle(cat(reshape(x_0_valid, state_dim, 1, batch_size), x_valid; dims=2), u_valid)
        refined_valid = net(valid_bundle, transition, traj_cost_fn)
        
        @test selectdim(refined_valid.x_guess, 2, 2:size(refined_valid.x_guess, 2)) ≈ x_valid
        @test refined_valid.u_guess ≈ u_valid atol=1e-5
        
    end
    
    @testset "refine recursion" begin
        Random.seed!(42)
        state_dim = 2
        input_dim = 1
        hidden_dim = 10
        out_dim = state_dim + input_dim
        depth = 2
        
        net = RefinementModel(state_dim, input_dim, hidden_dim, out_dim, depth)
        
        struct MockTransitionRec
        end
        (m::MockTransitionRec)(x, u) = x + vcat(u, u)
        transition = MockTransitionRec()
        
        traj_cost_fn(x_res) = sum(0.5f0 .- x_res[1, :, :])
        
        batch_size = 2
        seq_len = 4
        x_guess_body = rand(Float32, state_dim, seq_len, batch_size)
        u_guess = rand(Float32, input_dim, seq_len, batch_size)
        x0 = rand(Float32, state_dim, batch_size)
        x_full = cat(reshape(x0, state_dim, 1, batch_size), x_guess_body; dims=2)
        bundle = ShootingBundle(x_full, u_guess)
        
        # Manual two-step refinement
        b1 = net(bundle, transition, traj_cost_fn)
        b2 = net(b1, transition, traj_cost_fn)
        
        # Recursive call should match manual chaining
        b_rec = net(bundle, transition, traj_cost_fn, 2)
        
        @test b_rec.x_guess ≈ b2.x_guess
        @test b_rec.u_guess ≈ b2.u_guess
    end
    
    @testset "refinement_loss" begin
        # Use a zero refinement model so refinement leaves trajectories unchanged.
        struct ZeroBlock
            out_dim::Int
        end
        (z::ZeroBlock)(x, ctx) = zeros(Float32, z.out_dim, size(x, 2), size(x, 3))
        net = RefinementModel(ZeroBlock(2), ZeroBlock(2))
        
        state_dim = 1
        input_dim = 1
        # Simple transition: x_next = x_prev + u
        transition_fn(x_prev, u) = x_prev .+ u
        
        # Terminal cost: sum of terminal state
        traj_cost_fn(x_res) = sum(x_res)
        
        # Mismatch: squared error sum between residual and guess
        mismatch_fn(x_res, x_guess) = sum(abs2.(x_res .- x_guess))
        
        # Deterministic inputs
        seq_len = 2
        batch = 1
        x_guess_init = fill(1f0, state_dim, seq_len, batch)
        u_guess_init = fill(2f0, input_dim, seq_len, batch)
        x_0 = fill(0f0, state_dim, batch)
        bundle = ShootingBundle(cat(reshape(x_0, state_dim, 1, batch), x_guess_init; dims=2), u_guess_init)
        
        # Expected residual with zero-network (no correction): x_prev = [0, 1], u = [2, 2] => x_res = [2, 3]
        # Mismatch = (2-1)^2 + (3-1)^2 = 1 + 4 = 5
        # Trajectory cost = sum(x_res) = 5
        # Total = 10
        loss_val = refinement_loss(net, transition_fn, traj_cost_fn, mismatch_fn, bundle, 1)
        @test loss_val ≈ 10f0
    end

    @testset "train!" begin
        Random.seed!(7)
        state_dim = 2
        input_dim = 1
        hidden_dim = 6
        out_dim = state_dim + input_dim
        depth = 1
        
        net = RefinementModel(state_dim, input_dim, hidden_dim, out_dim, depth)

        transition_fn(x_prev, u) = x_prev .+ vcat(u, u)
        traj_cost_fn(x_res) = sum(x_res)
        traj_mismatch_fn(x_res, x_guess) = sum(abs2.(x_res .- x_guess))

        seq_len = 3

        # Sample with x_traj carrying the initial state.
        x_traj = zeros(Float32, state_dim, seq_len + 1, 1)
        x_traj[1, :, :] .= range(-0.3f0, stop=0.3f0, length=seq_len + 1)
        x_traj[2, :, :] .= 0.1f0
        u_traj = fill(0.05f0, input_dim, seq_len, 1)
        sample1 = ShootingBundle(x_traj, u_traj)

        # Sample with batch expansion for u_guess and target.
        x_guess2 = fill(0.2f0, state_dim, seq_len, 2)
        u_guess2 = fill(0.1f0, input_dim, seq_len, 1)
        x0_2 = fill(0.0f0, state_dim, 2)
        x_full2 = cat(reshape(x0_2, state_dim, 1, 2), x_guess2; dims=2)
        target2 = fill(0.25f0, state_dim, seq_len, 2)
        sample2 = ShootingBundle(x_full2, u_guess2; x_target=target2)

        data = [sample1, sample2]

        # Flatten parameters to a single vector (avoids deprecated Flux.params).
        flat_before, _ = Flux.destructure(net)

        _, losses = train!(net, data, 2, 1;
                           opt=Flux.Adam(5e-3),
                           transition_fn=transition_fn,
                           traj_cost_fn=traj_cost_fn,
                           traj_mismatch_fn=traj_mismatch_fn,
                           imitation_weight=0.3)

        @test length(losses) == 2
        @test all(isfinite.(losses))

        flat_after, _ = Flux.destructure(net)
        @test !isapprox(flat_before, flat_after; atol=1e-8)
    end
end
