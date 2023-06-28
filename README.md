# Entropy Regularized Point-based Value Iteration

[![Build Status](https://github.com/hdelecki/EntropyRegularizedPBVI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/hdelecki/EntropyRegularizedPBVI.jl/actions/workflows/CI.yml?query=branch%3Amain)

Entropy regularized point-based value iteration solver for the  [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework.

## Installation
From the julia REPL,
```
]add git@github.com:hdelecki/EntropyRegularizedPBVI.jl.git
```

## Usage

```julia
using EntropyRegularizedPBVI
using POMDPModels

pomdp = TigerPOMDP()

solver = ERPBVISolver(max_iterations=10, ϵ=0.1, λ=0.1, verbose=true)

policy = solve(solver, pomdp)
```
