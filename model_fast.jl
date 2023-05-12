cd(@__DIR__)
using Pkg
Pkg.activate(".")

using StochasticAD, Random, Distributions, Optimisers, Turing, LinearAlgebra

struct OccupancyModel{LocCov, StudyCov, pType}
    n::Int
    K::Vector{Int}
    Y::Vector{Vector{Bool}}
    Z::Vector{Bool}
    W::Vector{Vector{StudyCov}}
    X::Vector{LocCov}
    z_known::Vector{Bool}
    prior::pType
end

function OccupancyModel(Y,Z,W,X,prior)
    n = length(Y)
    K = [length(y) for y in Y]
    z_known = [!all(Y[i] .== 0) for i in 1:n]
    OccupancyModel(n, K, Y, Z, W, X, z_known, prior)
end

Random.seed!(123)

# probabilistc model
@inline ψ(x,β) = 0.00001 + 0.99998/(1+exp(-dot(β,x))) # occupation probability
@inline d(w,α) = 0.00001 + 0.99998/(1+exp(-dot(α,w))) # observation probability
@inline p_bern(ϕ)  =  1/2*(1+0.999*tanh(ϕ)) # 


function likelihood(y,z,αβ,z_known,W,X)
    p = 1.0
    @views α, β = αβ[1:2], αβ[3:4]
    k = 1
    for i in eachindex(X)
        zi = z_known[i] ? 1 : z[k]
        k += z_known[i] ? 0 : 1
        p *= ψ(X[i],β)^zi*(1-ψ(X[i], β))^(1-zi)
        for j in eachindex(W[i])
            p *= (zi*d(W[i][j], α))^y[i][j]*(1-zi*d(W[i][j], α))^(1-y[i][j])
        end

    end
    p 
end
likelihood(z, αβ, om::OccupancyModel) = likelihood(om.Y, z, αβ, om.z_known, om.W, om.X)

function logconditional(y,z,αβ,W,X;z_known = [])
    logp = 0.0
    @views α, β = αβ[1:2], αβ[3:4]
    if isempty(z_known)
        for i in eachindex(X)
            zi = z[i]
            for j in eachindex(W[i])
                dij = d(W[i][j], α)
                logp += log((zi*dij)^y[i][j]) + log((1-zi*dij)^(1-y[i][j]))
            end
        end
    else
        k = 1
        for i in eachindex(X)
            zi = z_known[i] ? 1 : z[k]
            k += z_known[i] ? 0 : 1
            for j in eachindex(W[i])
                dij = d(W[i][j], α)
                logp += log((zi*dij)^y[i][j]) + log((1-zi*dij)^(1-y[i][j]))
            end
        end
    end
    logp 
end
function logconditional_full(y,z,αβ,W,X)
    logp = 0.0
    @views α, β = αβ[1:2], αβ[3:4]
    for i in eachindex(X)
        zi = z[i]
        #oi = ψ(X[i],β)
        #logp += zi*log(oi) + (1-zi)*log(1-oi)
        for j in eachindex(W[i])
            dij = d(W[i][j], α)
            logp += log((zi*dij)^y[i][j]) + log((1-zi*dij)^(1-y[i][j]))
        end
    end
    logp 
end
logconditional(om::OccupancyModel, z, αβ) = logconditional(om.Y, z, αβ, om.W, om.X, z_known = om.z_known)

function loglikelihood(y,z,αβ,z_known,W,X)
    logp = 0.0
    @views α, β = αβ[1:2], αβ[3:4]
    k = 1
    for i in eachindex(X)
        zi = z_known[i] ? 1 : z[k]
        k += z_known[i] ? 0 : 1
        oi = ψ(X[i],β)
        logp += zi*log(oi) + (1-zi)*log(1-oi)#smooth_triple(zi*log(oi) + (1-zi)*log(1-oi))
        for j in eachindex(W[i])
            dij = d(W[i][j], α)
            logp += log((zi*dij)^y[i][j]) + log((1-zi*dij)^(1-y[i][j]))#smooth_triple(log((zi*dij)^y[i][j]) + log((1-zi*dij)^(1-y[i][j])))
        end
    end
    logp 
end
loglikelihood(z, αβ, om::OccupancyModel) = loglikelihood(om.Y, z, αβ, om.z_known, om.W, om.X)

function joint(y,z,αβ,prior,z_known,W,X)
    likelihood(y,z,αβ,z_known,W,X) * pdf(prior,αβ)
end
joint(z, αβ, om::OccupancyModel) = joint(om.Y, z, αβ, om.prior, om.z_known, om.W, om.X)

function logjoint(y,z,αβ,prior,z_known,W,X)
    loglikelihood(y,z,αβ,z_known,W,X) + logpdf(prior, αβ)
end
logjoint(z, αβ, om::OccupancyModel) = logjoint(om.Y, z, αβ, om.prior, om.z_known, om.W, om.X)

# marginalized over discrete latent variables
function logmarginal(y,z_known,αβ,W,X)
    logp = 0
    @views α, β = αβ[1:2], αβ[3:4]
    for i in eachindex(X)
        if z_known[i]    
            logp += log(ψ(X[i],β))
            for j in eachindex(W[i])
                logp += log((d(W[i][j], α))^y[i][j]*(1-d(W[i][j], α))^(1-y[i][j])) 
            end
        else 
            aux_0 = (1-ψ(X[i],β))
            aux_1 = ψ(X[i],β)
            for j in eachindex(W[i])
                #aux_0 *= y[i][j] == 0 ? 1 : 0 #0^y[i][j]*1^(1-y[i][j])
                aux_1 *= (1-d(W[i][j], α)) #d(W[i][j], α)^y[i][j]*(1-d(W[i][j], α))^(1-y[i][j])
            end
            logp += log(aux_0 + aux_1)
        end
    end
    logp
end
logmarginal(αβ, om::OccupancyModel) = logmarginal(om.Y, om.z_known, αβ, om.W, om.X)

function logjoint_marginal(y,z_known,αβ,prior, W, X)
    logmarginal(y,z_known,αβ, W, X) + logpdf(prior,αβ)
end
logjoint_marginal(αβ, om::OccupancyModel) = logjoint_marginal(om.Y, om.z_known, αβ, om.prior, om.W, om.X)

# variational distribution
function vec2ltri(v::AbstractVector{T}, z=zero(T)) where T 
    n = length(v)
    s = round(Int,(sqrt(8n+1)-1)/2)
    s*(s+1)/2 == n || error("vec2utri: length of vector is not triangular")
    [ i>=j ? v[round(Int, j*(j-1)/2+i)] : z for i=1:s, j=1:s ]
end

function reshape_params(ϕ)
    @views μ = ϕ[1:4]
    V = reshape(ϕ[5:20], 4, 4)#vec2ltri(ϕ[5:14])

    μ, V*V' + 0.01*I
end
function q(ϕ,z,αβ)
    μ, Σ = reshape_params(ϕ)
    #Σinv = Distributions.PDMat(V*V')
    #q_αβ = MvNormalCanon(μ, Σinv*μ, Σinv)
    q_αβ = MvNormal(μ, Σ)

    q_pdf = pdf(q_αβ,αβ)  
    for (i,zi) in enumerate(z)
        p = p_bern(ϕ[20+i])
        q_pdf *= p^zi * (1-p)^(1-zi)
    end
   
    q_pdf
end


function logq(ϕ,z,αβ)
    μ, Σ = reshape_params(ϕ)
    #Σinv = Distributions.PDMat(V*V')
    #q_αβ = MvNormalCanon(μ, Σinv*μ, Σinv)
    q_αβ = MvNormal(μ, Σ)
    q_logpdf = logpdf(q_αβ,αβ)
    for (i,zi) in enumerate(z) 
        p = p_bern(ϕ[20+i]) 
        q_logpdf += log(p^zi * (1-p)^(1-zi))#smooth_triple(log(p^zi * (1-p)^(1-zi)))
    end
    
    q_logpdf
end

function logq_marginal(ϕ,αβ)
    μ, Σ = reshape_params(ϕ)
    #Σinv = Distributions.PDMat(V*V')
    #q_αβ = MvNormalCanon(μ, Σinv*μ, Σinv)
    q_αβ = MvNormal(μ, Σ)
    
    logpdf(q_αβ, αβ)
end

function sample_q(ϕ)
    μ, Σ = reshape_params(ϕ)
    
    q_αβ = MvNormal(μ, Hermitian(Σ))
    αβ = rand(q_αβ)
    z = [rand(Bernoulli(p_bern(ϕ[i]))) for i in 21:length(ϕ)]
    
    z, αβ
end

function sample_q_marginal(ϕ)
    μ, Σ = reshape_params(ϕ)
    
    q_αβ = MvNormal(μ, Hermitian(Σ))
    rand(q_αβ)
end

#=
function sample_q!(z,αβ,ϕ)
    μ = ϕ[1:4]
    V = reshape(ϕ[5:20], 4, 4)
    
    q_αβ = MvNormal(μ, Hermitian(V*V' + 0.01*I))
    rand!(q_αβ, αβ)
    for i in eachindex(z)
        p = p_bern(ϕ[20+i])
        z[i] = rand(Bernoulli(p))
    end
end

function logq(ϕ,z,αβ)
    @views μ = ϕ[1:4]
    V = reshape(ϕ[5:20], 4, 4)
    #Σinv = Distributions.PDMat(V*V')
    #q_αβ = MvNormalCanon(μ, Σinv*μ, Σinv)
    q_αβ = MvNormal(μ, V*V' + 0.01*I)
    ps = [p_bern(ϕ[i]) for i in 21:length(ϕ)]
    
    logpdf(q_αβ, αβ) + sum(@. smooth_triple(log((ps ^ z) * ((1 - ps) ^ (1 - z)))))
end

function populate_triangular!(V, entries, n)
    k = 1
    for i in 1:n
        V[i,i] = entries[k] + 0.01
        k += 1
        for j in i+1:n
            V[i,j] = entries[k]
            k += 1
        end
    end
end

function populate_triangular(entries, n)
    V = zeros(n,n)
    k = 1
    for i in 1:n
        V[i,i] = entries[k] + 0.01
        k += 1
        for j in i+1:n
            V[i,j] = entries[k]
            k += 1
        end
    end
    return V
end
=#

function div(ϕ, z, αβ, Y, z_known, prior, W, X)
    logjoint(Y, z, αβ, prior, z_known, W, X) - logq(ϕ,z,αβ)
end
div(ϕ, z, αβ, om::OccupancyModel) = div(ϕ, z, αβ, om.Y, om.z_known, om.prior, om.W, om.X)

function div_marginal(ϕ, αβ, Y, z_known, prior, W, X)
    logjoint_marginal(Y, z_known, αβ, prior, W, X) - logq_marginal(ϕ,αβ)
end
div_marginal(ϕ, αβ, om::OccupancyModel) = div_marginal(ϕ, αβ, om.Y, om.z_known, om.prior, om.W, om.X)

@model function marginal_occupancy(Y, z_known, prior, W, X)
    αβ ~ prior
    Turing.@addlogprob! logmarginal(Y,z_known,αβ,W,X)
end    
marginal_occupancy(om::OccupancyModel) = marginal_occupancy(om.Y, om.z_known, om.prior, om.W, om.X)

@model function occupancy(Y, z_known, prior, W, X)
    αβ ~ prior
    
    @views α = αβ[1:2]
    @views β = αβ[3:4]
    #=
    z = fill(undef, length(Y))
    for i in eachindex(X)
        z[i] ~ Bernoulli(ψ(X[i],β))
        for j in eachindex(W[i])
            Y[i][j] ~  Bernoulli(z[i]*d(W[i][j],α))
        end
    end
    =#
    z = fill(undef, length(z_known) - sum(z_known))
    k = 1
    for i in eachindex(X)
        if z_known[i]
            z_known[i] ~ Bernoulli(ψ(X[i],β))
            for j in eachindex(W[i])
                Y[i][j] ~ Bernoulli(d(W[i][j],α))
            end
        else
            z[k] ~ Bernoulli(ψ(X[i],β))
            for j in eachindex(W[i])
                Y[i][j] ~ Bernoulli(z[k]*d(W[i][j],α))
            end
            k += 1
        end
    end
end    
occupancy(om::OccupancyModel) = occupancy(om.Y, om.z_known, om.prior, om.W, om.X)