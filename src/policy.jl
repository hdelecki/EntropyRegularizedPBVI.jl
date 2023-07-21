mutable struct EntropyRegularizedPolicy{P<:POMDP, A} <: Policy
    pomdp::P
    qs::Vector{Vector{Vector{Float64}}}
    action_map::Vector{A}
    temperature::Float64
    stochastic::Bool
end

# function EntropyRegularizedPolicy(pomdp::POMDP, qs::Vector{Matrix{Float64}}, action_map::Vector, temperature::Float64)
#     return EntropyRegularizedPolicy(pomdp, qs, action_map, temperature)
# end

POMDPs.updater(p::EntropyRegularizedPolicy) = DiscreteUpdater(p.pomdp)

function POMDPs.action(p::EntropyRegularizedPolicy, b)
    λ = p.temperature
    Qba = [_argmax(α -> α⋅b.b, Γi) for Γi in p.qs]
    
    if p.stochastic
        # return random action from softmax sample
        W = softmax([dot(α, b.b)/λ for α in Qba])

        # Sample from softmax weights W
        aidx = rand(Categorical(W./sum(W)))
    else
        # return argmax action
        aidx = argmax([dot(α, b.b)/λ for α in Qba])
    end

    return p.action_map[aidx]
end