using ReachabilityCascade: ConditionalFlowExamples

let examples = ConditionalFlowExamples
    rt = examples.roundtrip()
    @assert rt.max_reconstruction_error < 1f-4
    println("Roundtrip example latent size: ", size(rt.latent))

    scaled = examples.roundtrip_scaled()
    @assert any(abs.(scaled.logdet) .> 0f0)
    println("Scaled roundtrip logdet sample: ", scaled.logdet)

    ss = examples.single_sample()
    @assert length(ss.logdet) == 1
    println("Single-sample example logdet: ", ss.logdet)

    recur = examples.recurrent_roundtrip()
    @assert recur.max_reconstruction_error < 1f-4
    println("Recurrent example total logdet: ", recur.logdet)
    println("Recurrent transitions recorded: ", length(recur.per_step_latents))
end
