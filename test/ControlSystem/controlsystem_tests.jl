using Test
using ReachabilityCascade
using LazySets

@testset "DiscreteRandomSystem trajectory periods" begin
    X = Hyperrectangle(zeros(1), ones(1))
    U = Hyperrectangle(zeros(1), ones(1))
    ds = ReachabilityCascade.DiscreteRandomSystem(X, U, (x, u) -> x .+ u)

    x0 = [0.0]
    umat = reshape([1.0, 2.0], 1, :)

    trj = ds(x0, umat, 2)
    @test size(trj, 2) == 1 + 2 * size(umat, 2)

    trj2 = ds(x0, umat, [1, 3])
    @test size(trj2, 2) == 1 + 1 + 3

    @test_throws DimensionMismatch ds(x0, umat, [1])
end
