using Test
using POMDPModels
using EntropyRegularizedPBVI


@testset "EntropyRegularizedPBVI.jl" begin
    pomdp = TigerPOMDP()

    solver = ERPBVISolver(max_iterations=10, ϵ=0.1, λ=0.1, verbose=true)
    policy = solve(solver, pomdp)
end
