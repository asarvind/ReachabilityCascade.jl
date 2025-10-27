"""
    Gan(latent_dim, context_dim, data_dim; kwargs...)

Convenience container that wires conditional generator, discriminator, and encoder
blocks built from `glu_mlp`. All networks accept a context vector alongside their
primary input, and the encoder outputs lie inside the unit hyperrectangle via a
saturating activation (default `tanh`).
"""
struct Gan{G,D,E}
    generator::G
    discriminator::D
    encoder::E
    latent_dim::Int
    context_dim::Int
    data_dim::Int
end

Flux.@layer Gan

"""
    _as_matrix(x)

Normalize vectors into `(features, batch)` matrices.
"""
_as_matrix(x::AbstractVector) = reshape(x, :, 1)
_as_matrix(x::AbstractMatrix) = x

function _stack_inputs(args...)
    mats = map(_as_matrix, args)
    ncols = size(mats[1], 2)
    @assert all(size(mat, 2) == ncols for mat in mats) "Conditional GAN input batch mismatch"
    return Flux.f32(reduce(vcat, mats))
end

function Gan(latent_dim::Integer, context_dim::Integer, data_dim::Integer;
             gen_hidden::Integer=128,
             disc_hidden::Integer=128,
             enc_hidden::Integer=128,
             n_glu_gen::Integer=2,
             n_glu_disc::Integer=2,
             n_glu_enc::Integer=2,
             gen_gate=Flux.σ,
             disc_gate=Flux.σ,
             enc_gate=Flux.σ,
             generator_out::Union{Function,Nothing}=Flux.tanh,
             discriminator_out::Union{Function,Nothing}=Flux.σ,
             encoder_saturation::Function=tanh,
             generator_zero_init::Bool=true,
             discriminator_zero_init::Bool=false,
             encoder_zero_init::Bool=false)
    gen_input_dim = latent_dim + context_dim
    gen_core = glu_mlp(gen_input_dim, gen_hidden, data_dim;
                       n_glu=n_glu_gen, act=gen_gate, zero_init=generator_zero_init)
    generator = generator_out === nothing ? gen_core : Chain(gen_core, generator_out)

    disc_input_dim = data_dim + context_dim
    disc_core = glu_mlp(disc_input_dim, disc_hidden, 1;
                        n_glu=n_glu_disc, act=disc_gate, zero_init=discriminator_zero_init)
    discriminator = discriminator_out === nothing ? disc_core : Chain(disc_core, discriminator_out)

    enc_input_dim = data_dim + context_dim
    enc_core = glu_mlp(enc_input_dim, enc_hidden, latent_dim;
                       n_glu=n_glu_enc, act=enc_gate, zero_init=encoder_zero_init)
    encoder_tail = x -> encoder_saturation.(x)
    encoder = Chain(enc_core, encoder_tail)

    return Gan(generator, discriminator, encoder, latent_dim, context_dim, data_dim)
end

(gan::Gan)(context, latent) = generator_forward(gan, context, latent)

"""
    generator_forward(gan::Gan, context, latent)

Evaluate the generator on a context/latent pair. Supports vectors and batched matrices.
"""
function generator_forward(gan::Gan, context, latent)
    input = _stack_inputs(context, latent)
    @assert size(input, 1) == gan.context_dim + gan.latent_dim "Generator input/context mismatch"
    return gan.generator(input)
end

"""
    discriminator_forward(gan::Gan, context, samples)

Evaluate the discriminator on context-conditioned samples. Supports batched input.
"""
function discriminator_forward(gan::Gan, context, samples)
    input = _stack_inputs(context, samples)
    @assert size(input, 1) == gan.context_dim + gan.data_dim "Discriminator input/context mismatch"
    return gan.discriminator(input)
end

"""
    encoder_forward(gan::Gan, context, samples)

Map a context-conditioned sample back into the latent space. Outputs are guaranteed
to lie inside the unit hyperrectangle by the saturating tail activation.
"""
function encoder_forward(gan::Gan, context, samples)
    input = _stack_inputs(context, samples)
    @assert size(input, 1) == gan.context_dim + gan.data_dim "Encoder input/context mismatch"
    return gan.encoder(input)
end
