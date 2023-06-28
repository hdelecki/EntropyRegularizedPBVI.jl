"""
    QPBVISolver <: Solver

Options dictionary for Point-Based Value Iteration for POMDPs.

# Fields
- `max_iterations::Int64` the maximal number of iterations the solver runs. Default: 10
- `ϵ::Float64` the maximal gap between alpha vector improve steps. Default = 0.01
- `verbose::Bool` switch for solver text output. Default: false
"""
struct QPBVISolver <: Solver
    max_iterations::Int64
    ϵ::Float64
    λ::Float64
    optimizer_factory
    verbose::Bool
end

function QPBVISolver(;max_iterations::Int64=10, ϵ::Float64=0.01, λ::Float64=1.0, optimizer_factory=GLPK.Optimizer, verbose::Bool=false)
    return QPBVISolver(max_iterations, ϵ, λ, optimizer_factory, verbose)
end
"""
    AlphaVec

Pair of alpha vector and corresponding action.

# Fields
- `alpha` α vector
- `action` action corresponding to α vector
"""
struct AlphaVec
    alpha::Vector{Float64}
    action::Any
end

==(a::AlphaVec, b::AlphaVec) = (a.alpha,a.action) == (b.alpha, b.action)
Base.hash(a::AlphaVec, h::UInt) = hash(a.alpha, hash(a.action, h))

function _argmax(f, X)
    return X[argmax(map(f, X))]
end

function dominate(α::Array{Float64,1}, A::Set{Array{Float64,1}}, optimizer_factory)
    ns = length(α)
    αset = Set{Array{Float64,1}}()
    push!(αset, α)
    Adiff = setdiff(A,αset)
    L = Model(optimizer_factory)
    @variable(L, x[1:ns])
    @variable(L, δ)
    @objective(L, Max, δ)
    @constraint(L, sum(x) == 1)
    @constraint(L, x[1:ns] .<= 1)
    @constraint(L, x[1:ns] .>= 0)
    for ap in Adiff
        @constraint(L, dot(x, α) >= δ + dot(x, ap))
    end
    JuMP.optimize!(L)
    sol_status = JuMP.termination_status(L)
    if sol_status == :Infeasible
        return :Perp
    else
        xval = JuMP.value.(x)
        dval = JuMP.value.(δ)
        if dval > 0
            return xval
        else
            return :Perp
        end
    end
end

"""
    filtervec(F)

The set of vectors in `F` that contribute to the value function.
"""
function filtervec(F::Set{Array{Float64,1}}, optimizer_factory)
    ns = length(sum(F))
    W = Set{Array{Float64,1}}()
    for i = 1: ns
        if !isempty(F)
            # println("i: $i  ")
            w = Array{Float64,1}()
            fsmax = -Inf
            for f in F
                # println("f: $f")
                if f[i] > fsmax
                    fsmax = f[i]
                    w = f
                    end
            end
            wset = Set{Array{Float64,1}}()
            push!(wset, w)
            # println("w: $w")
            push!(W,w)
            setdiff!(F,wset)
        end
    end
    while !isempty(F)
        ϕ = pop!(F)
        x = dominate(ϕ, W, optimizer_factory)
        if x != :Perp
            push!(F, ϕ)
            w = Array{Float64,1}()
            fsmax = -Inf
            for f in F
                if dot(x, f) > fsmax
                    fsmax = dot(x, f)
                    w = f
                    end
            end
            wset = Set{Array{Float64,1}}()
            push!(wset, w)
            push!(W,w)
            setdiff!(F,wset)
        end
    end
    temp = [Float64[]]
    setdiff!(W,temp)
    W
end

# adds probabilities of terminals in b to b′ and normalizes b′
function belief_norm(pomdp, b, b′, terminals, not_terminals)
    if sum(b′[not_terminals]) != 0.
        if !isempty(terminals)
            b′[not_terminals] = b′[not_terminals] / (sum(b′[not_terminals]) / (1. - sum(b[terminals]) - sum(b′[terminals])))
            b′[terminals] += b[terminals]
        else
            b′[not_terminals] /= sum(b′[not_terminals])
        end
    else
        b′[terminals] += b[terminals]
        b′[terminals] /= sum(b′[terminals])
    end
    return b′
end


function ∇Ulse(solver, b, Γ)
    λ = solver.λ
    W = softmax([dot(α, b)/λ for α in Γ])
    ∇U = hcat(Γ...)*W
    return ∇U
end


function Ulse(solver, b, Γ)
    λ = solver.λ
    Qba = [_argmax(α -> α⋅b, Γi) for Γi in Γ]
    return λ*logsumexp([dot(α, b)/λ for α in Qba])
end


function U(solver, b, A)
    return maximum([dot(α, b) for α in A])
end

# Backups belief with α vector maximizing dot product of itself with belief b
function backup_belief(pomdp::POMDP, Γ, b, solver)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    Γa = Vector{Vector{Float64}}(undef, length(A))

    not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
    terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
    for a in A
        Γao = Vector{Vector{Float64}}(undef, length(O))
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        if !isempty(terminals) trans_probs[terminals] .+= b.b[terminals] end

        for o in O
            # update beliefs
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs

            if sum(b′) > 0.
                b′ = DiscreteBelief(pomdp, b.state_list, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
            else
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
            end
            
            # extract optimal alpha vector in each q-function
            Qba = [_argmax(α -> α ⋅ b′.b, Γi) for Γi in Γ]
            
            # extract optimal alpha vector at resulting belief
            Γao[obsindex(pomdp, o)] = ∇Ulse(solver, b′.b, Qba) #_argmax(α -> α ⋅ b′.b, Qba)
        end

        # construct new alpha vectors
        Γa[actionindex(pomdp, a)] = [r(s, a) + (!isterminal(pomdp, s) ? (γ * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                                        for (j, sp) in enumerate(S), (i, o) in enumerate(O))) : 0.)
                                        for s in S]
    end

    return [AlphaVec(Γa[actionindex(pomdp, a)], A[actionindex(pomdp, a)]) for a in A]
end

# Iteratively improves α vectors until the gap between steps is lesser than ϵ
function improve(pomdp, B, Γ, solver)
    alphavecs = nothing
    A = ordered_actions(pomdp)
    while true
        Γold = Γ
        new_alphas = [backup_belief(pomdp, Γold, b, solver) for b in B]
        Γ = [[alist[actionindex(pomdp, a)].alpha for alist in new_alphas] for a in A]
        for (i,Γi) in enumerate(Γ)
            append!(Γi, Γold[i])
        end
        prec = max([sum(abs.(U(solver, b.b, A1) .- U(solver, b.b, A2))) for (A1, A2, b) in zip(Γold, Γ, B)]...)
        if solver.verbose println("    Improving alphas, maximum gap between old and new α vector: $(prec)") end
        prec > solver.ϵ || break
    end

    return Γ#, alphavecs
end

function prune(solver, Γ)
    Γpruned = [[filtervec(Set(Γi), solver.optimizer_factory)...] for Γi in Γ]
    return Γpruned
end

# Returns all possible, not yet visited successors of current belief b
function successors(pomdp, b, Bs)
    S = ordered_states(pomdp)
    not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
    terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
    succs = []

    for a in actions(pomdp)
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        if !isempty(terminals) trans_probs[terminals] .+= b[terminals] end

        for o in observations(pomdp)
            #update belief
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs


            if sum(b′) > 0.
                b′ = belief_norm(pomdp, b, b′, terminals, not_terminals)

                if !in(b′, Bs)
                    push!(succs, b′)
                end
            end
        end
    end

    return succs
end

# Computes distance of successor to the belief vectors in belief space
function succ_dist(pomdp, bp, B)
    dist = [norm(bp - b.b, 1) for b in B]
    return max(dist...)
end

# Expands the belief space with the most distant belief vector
# Returns new belief space, set of belifs and early termination flag
function expand(pomdp, B, Bs)
    B_new = copy(B)
    for b in B
        succs = successors(pomdp, b.b, Bs)
        if length(succs) > 0
            b′ = succs[argmax([succ_dist(pomdp, bp, B) for bp in succs])]
            push!(B_new, DiscreteBelief(pomdp, b′))
            push!(Bs, b′)
        end
    end

    return B_new, Bs, length(B) == length(B_new)
end

# 1: B ← {b0}
# 2: while V has not converged to V∗ do
# 3:    Improve(V, B)
# 4:    B ← Expand(B)
function solve(solver::QPBVISolver, pomdp::POMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    # best action worst state lower bound
    α_init = 1 / (1 - γ) * maximum(minimum(r(s, a) for s in S) for a in A)
    Γ = [[fill(α_init, length(S))] for a in A]

    #init belief, if given distribution, convert to vector
    init = initialize_belief(DiscreteUpdater(pomdp), initialstate(pomdp))
    B = [init]
    Bs = Set([init.b])

    if solver.verbose println("Running PBVI solver on $(typeof(pomdp)) problem with following settings:\n    max_iterations = $(solver.max_iterations), ϵ = $(solver.ϵ), verbose = $(solver.verbose)\n+----------------------------------------------------------+") end

    # original code should run until V converges to V*, this yet needs to be implemented
    # for example as: while max(@. abs(newV - oldV)...) > solver.ϵ
    # However this probably would not work, as newV and oldV have different number of elements (arrays of alphas)
    alphavecs = nothing
    for i in 1:solver.max_iterations
        Γ = improve(pomdp, B, Γ, solver)
        Γ = prune(solver, Γ)
        B, Bs, early_term = expand(pomdp, B, Bs)
        if solver.verbose println("Iteration $(i) executed, belief set contains $(length(Bs)) belief vectors.") end
        if early_term
            if solver.verbose println("Belief space did not expand. \nTerminating early.") end
            break
        end
    end

    if solver.verbose println("+----------------------------------------------------------+") end

    policy = QSoftmaxPolicy(pomdp, Γ, A, solver.λ)

    return policy
end