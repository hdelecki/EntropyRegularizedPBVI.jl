module EntropyRegularizedPBVI

using POMDPs
using POMDPTools
using LinearAlgebra
using Distributions
using FiniteHorizonPOMDPs
using JuMP, GLPK

import POMDPs: Solver, solve, action, value, updater
import Base: ==, hash, convert
import FiniteHorizonPOMDPs: InStageDistribution, FixedHorizonPOMDPWrapper
import StatsFuns: logsumexp, softmax


export
    ERPBVISolver,
    solve

include("erpbvi.jl")

export
    EntropyRegularizedPolicy,
    action

include("policy.jl")


end
