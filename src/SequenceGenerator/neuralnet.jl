struct SequenceGenerator{N}
	sample_dim::Integer 
	context_dim::Integer 
	latent_dim::Integer 
	net::N
end

Flux.@layer SequenceGenerator

# constructor 
function SequenceGenerator(; sample_dim::Integer=2, context_dim=2, latent_dim=2, widths::Vector{<:Integer}=[128, 128], act::Function=Flux.leakyrelu, gate_act=Flux.σ, bias::Bool=true)
	@assert latent_dim >= sample_dim "the dimension of sample should not be greater than latent"
	
	layers = []

	# push dense layer 
	push!(
		layers,
		Flux.Dense(
			(context_dim+latent_dim)=>widths[1],
			act 
		)
	)

	# push gated layer 
	push!(
		layers,
		GLU(
			widths[1] => widths[1],
			act=gate_act
		)
	)

	for i in 1:(length(widths)-1)
		# push dense layer
		push!(layers, Flux.Dense(widths[i]=>widths[i+1], act))

		# push gated layer
		push!(layers, GLU(widths[i+1]=>widths[i+1], act=gate_act))
	end

	# push final dense layer
	push!(layers, Flux.Dense(widths[end]=>latent_dim)) 

	net = Flux.Chain(layers...)

	return SequenceGenerator(sample_dim, context_dim, latent_dim, net)
end

function generator_velocity(sg::SequenceGenerator, context::AbstractVecOrMat{<:Real}, latent::AbstractVecOrMat{<:Real})
	@assert size(context, 2)==size(latent, 2) "batch size of context and latent should be same"
	
	# concatenate and convert to Float32
	in_arr = Float32.(vcat(context, latent))

	return sg.net(in_arr)
end

function generator_velocity(sg::SequenceGenerator, sample::AbstractVecOrMat{<:Real}, context::AbstractVecOrMat{<:Real}, latent::AbstractVecOrMat{<:Real}, tseq::Union{Vector{<:Real}, Real})
	@assert sg.latent_dim >= sg.sample_dim "the dimension of sample should not be greater than latent"
	@assert size(sample, 1) == sg.sample_dim "dimension of sample should be same as specified in the generator"
	@assert size(context, 1) == sg.context_dim "dimension of context should be same as specified in the generator"
	@assert size(latent, 1) == sg.latent_dim "dimension of latent should be same as specified in the generator"
	@assert size(context, 2)==size(sample, 2) "batch size of sample and context should be same"	
	@assert size(context, 2)==size(latent, 2) "batch size of context and latent should be same"	

	# pad samples
	padding = latent[(sg.sample_dim + 1):end, :]
	sample_padded = vcat(sample, padding)

	# number of samples times
	n = length(tseq)

	# extend context
	ctx_ext = repeat(context, 1, n)
	# extedn sample_padded 
	sample_padded = repeat(sample_padded, 1, n)
	# extend latent
	latent_ext = repeat(latent, 1, n)
	# extend times
	τ = reduce(vcat, fill(t, size(latent, 2)) for t in tseq)

	# calculate intermediate latents 
	inter_latent = latent_ext + τ'.*(sample_padded - latent_ext)

	return sg.net(Float32.(vcat(ctx_ext, inter_latent))), sample_padded - latent_ext
end

function (sg::SequenceGenerator)(context::AbstractVecOrMat{<:Real}, latent::AbstractVecOrMat{<:Real}, n::Integer=5)
	new_latent = Float32.(copy(latent))

	for _ in 1:n
		new_latent += generator_velocity(sg, context, new_latent)/n 
	end

	return new_latent[1:sg.sample_dim, :]
end
