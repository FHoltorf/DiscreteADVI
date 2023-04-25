cd(@__DIR__)
using Pkg
Pkg.activate(".")

using StochasticAD, Random, Distributions, Optimisers, Turing, LinearAlgebra

Random.seed!(123)

# probabilistc model
ψ(x,β) = 0.001 + 0.999/(1+exp(-β'*x)) # occupation probability
d(w,α) = 0.001 + 0.999/(1+exp(-α'*w)) # observation probability

function likelihood(y,z,αβ;W=W,X=X)
    p = 1.0
    @views α, β = αβ[1:2], αβ[3:4]
    for i in eachindex(X)
        p *= ψ(X[i],β)^z[i]*(1-ψ(X[i], β))^(1-z[i])
        for j in eachindex(W[i])
            p *= (z[i]*d(W[i][j], α))^y[i][j]*(1-z[i]*d(W[i][j], α))^(1-y[i][j])
        end
    end
    return p 
end

function loglikelihood(y,z,αβ;W=W,X=X)
    logp = 0.0
    @views α, β = αβ[1:2], αβ[3:4]
    for i in eachindex(X)
        logp += z[i]*log(ψ(X[i],β)) + (1-z[i])*log(1-ψ(X[i], β))
        for j in eachindex(W[i])
            logp += y[i][j]*log(z[i]*d(W[i][j], α))+ (1-y[i][j])*log(1-z[i]*d(W[i][j], α))
        end
    end
    return logp 
end

function joint(y,z,αβ,prior; W=W, X=X)
    return likelihood(y,z,αβ;W=W,X=X)*prior(αβ)
end

function logjoint(y,z,αβ,logprior;W=W,X=X)
    logp = 0.0
    @views α, β = αβ[1:2], αβ[3:4]
    for i in eachindex(X)
        if z_known[i]    
            logp += log(ψ(X[i],β))
            for j in eachindex(W[i])
                logp += log((d(W[i][j], α))^y[i][j]*(1-d(W[i][j], α))^(1-y[i][j])) 
            end
        else 
            logp += log(ψ(X[i],β)^z[i]*(1-ψ(X[i], β))^(1-z[i]))
            for j in eachindex(W[i])
                logp += log((z[i]*d(W[i][j], α))^y[i][j]*(1-z[i]*d(W[i][j], α))^(1-y[i][j]))
            end
        end
    end
    return logp+logprior(αβ)
end

# marginalized over discrete latent variables
function logmarginal(y,z_known,αβ; W=W, X=X)
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
                aux_0 *= y[i][j] == 0 ? 1 : 0 #0^y[i][j]*1^(1-y[i][j])
                aux_1 *= d(W[i][j], α)^y[i][j]*(1-d(W[i][j], α))^(1-y[i][j])
            end
            logp += log(aux_0 + aux_1)
        end
    end
    return logp
end

# variational distribution
function q(ϕ,z,αβ; z_known=z_known)
    @views α, β = αβ[1:2], αβ[3:4]
    μ = ϕ[1:4]
    V = reshape(ϕ[5:20], 4, 4)
    #Σinv = Distributions.PDMat(V*V')
    #q_αβ = MvNormalCanon(μ, Σinv*μ, Σinv)
    q_αβ = MvNormal(μ, V*V' + 0.01*I)
    ps = [z_known[i] == 1 ? 1 : 1/2*(1+0.999s*tanh(ϕ[20+i])) for i in eachindex(X)]
    return pdf(q_αβ, αβ)*prod((ps .^ z) .* ((1 .- ps) .^ (1 .- z)))
end

function logq(ϕ,z,αβ; z_known=z_known)
    @views α, β = αβ[1:2], αβ[3:4]
    μ = ϕ[1:4]
    V = reshape(ϕ[5:20], 4, 4)
    #Σinv = Distributions.PDMat(V*V')
    #q_αβ = MvNormalCanon(μ, Σinv*μ, Σinv)
    q_αβ = MvNormal(μ, V*V' + 0.01*I)
    ps = [z_known[i] == 1 ? 1 : 1/2*(1+0.999*tanh(ϕ[20+i])) for i in eachindex(X)]
    return logpdf(q_αβ, αβ) + sum(@. log((ps ^ z) * ((1 - ps) ^ (1 - z))))
end

function sample_q(ϕ; X=X, W=W, z_known=z_known)
    μ = ϕ[1:4]
    V = reshape(ϕ[5:20], 4, 4)
    #Σinv = Distributions.PDMat(V*V')
    #q_αβ = MvNormalCanon(μ, Σinv*μ, Σinv)
    ps = [z_known[i] == 1 ? 1.0 : 1/2*(1+0.999*tanh(ϕ[20+i])) for i in eachindex(X)]

    q_αβ = MvNormal(μ, Hermitian(V*V' + 0.01*I))
    αβ = rand(q_αβ)
    @views α, β = αβ[1:2], αβ[3:4]
    q_z = Bernoulli.(ps) 
    return [z_known[i] == 1 ? 1 : rand(q_z[i]) for i in eachindex(q_z)], αβ
end

function div(ϕ, z, αβ; y=Y, prior=prior, logprior=logprior, W=W, X=X)
    return logjoint(y, z, αβ, logprior, W=W, X=X) - logq(ϕ,z,αβ)
end

@model function marginal_occupancy(Y;z_known = z_known, prior = αβ_prior, W=W, X=X)
    αβ ~ prior
    Turing.@addlogprob! logmarginal(Y,z_known,αβ,W=W,X=X)
end    

@model function occupancy(Y;z_known = z_known, prior = αβ_prior, W=W, X=X)
    αβ ~ prior
    β = αβ[3:4]
    for i in eachindex(X[i])
        z[i] ~ Bernoulli(z_known[i] ? 1 : ψ(X[i],β))
    end
    Turing.@addlogprob! loglikelihood(Y,z_known,αβ,W=W,X=X)
end    
