using Test
using ReachabilityCascade

@testset "Kolmogorov Layers" begin
    @testset "SprecherLayer shapes and batching" begin
        in_dim, out_dim = 3, 4
        layer = ReachabilityCascade.KolmogorovLayers.SprecherLayer(in_dim, out_dim; act=identity)

        x = rand(Float32, in_dim)
        y = layer(x)
        @test size(y) == (out_dim,)

        xb = rand(Float32, in_dim, 5)
        yb = layer(xb)
        @test size(yb) == (out_dim, 5)
    end

    @testset "SprecherNetwork forward" begin
        in_dim = 3
        hidden = [4, 5]
        out_dim = 2
        net = ReachabilityCascade.KolmogorovRepresentation.SprecherNetwork(in_dim, hidden, out_dim; act=relu)

        x = rand(Float32, in_dim)
        y = net(x)
        @test size(y) == (out_dim,)

        xb = rand(Float32, in_dim, 6)
        yb = net(xb)
        @test size(yb) == (out_dim, 6)
    end
end
