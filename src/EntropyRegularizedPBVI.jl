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

# Write your package code here.

export say_hello

say_hello() = println("Hello!")

# export
#     ERPBVISolver,
#     solve

# include("erpbvi.jl")

export
    QPBVISolver,
    solve

include("qpbvi.jl")

export
    EntropyRegularizedPolicy,
    action

include("policy.jl")


end
