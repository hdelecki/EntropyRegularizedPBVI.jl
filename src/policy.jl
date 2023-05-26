struct EntropyRegularizedPolicy{P<:POMDP, A} <: Policy
    pomdp::P
    qs::Vector{Matrix{Float64}}
    action_map::Vector{A}
    temperature::Float64
end

# function EntropyRegularizedPolicy(pomdp::POMDP, qs::Vector{Matrix{Float64}}, action_map::Vector, temperature::Float64)
#     return EntropyRegularizedPolicy(pomdp, qs, action_map, temperature)
# end

POMDPs.updater(p::EntropyRegularizedPolicy) = DiscreteUpdater(p.pomdp)

function POMDPs.action(p::EntropyRegularizedPolicy, b)
    # Find dominating q-function
    λ = p.temperature


    Qb = [λ*logsumexp((A'*b.b)/λ) for A in p.qs]#λ*logsumexp([dot(α, b)/λ for α in eachcol(A)])
    Qmax = p.qs[argmax(Qb)]

    # return random action from softmax sample
    W = softmax([dot(α, b.b)/λ for α in eachcol(Qmax)])

    # Sample from softmax weights W
    aidx = rand(Categorical(W./sum(W)))

    return p.action_map[aidx]
end