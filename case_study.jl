cd(@__DIR__)
using Pkg
Pkg.activate(".")

using StochasticAD, Random, Distributions, Optimisers, CairoMakie, LinearAlgebra
using RData, DataFrames, Dates, CategoricalArrays
Random.seed!(413)

specs = [1,2,3,4,5]
for spec in specs
    data = load("design_matos/design_mat$(spec).Rdata")
    function organize_data(data)
        X = [vcat(x, 1) for x in data["design_mats"]["X"].AET_div_PET_s]
        Y = []
        W = []
        n = 0
        for k in data["design_mats"]["nvisits"]
            push!(Y, data["design_mats"]["Y"].V1[n+1:n+k])
            push!(W, [vcat(data["design_mats"]["W"].nspp[i],1) for i in n+1:n+k])
            n += k
        end
        return X, Y, W
    end

    X, Y, W = organize_data(data)

    K = [length(y) for y in Y] # visits per site
    n = length(Y) # number of sites

    # regression coeff priors
    αβ_prior = MvNormal(Float64[0,0,0,0], 1000*I)
    prior(αβ; αβ_prior=αβ_prior) = pdf(αβ_prior, αβ)
    logprior(αβ; αβ_prior=αβ_prior) = logpdf(αβ_prior, αβ)

    # probabilistc model
    ψ(x,β) = 0.001 + 0.999/(1+exp(-β'*x)) # occupation probability
    d(w,α) = 0.001 + 0.999/(1+exp(-α'*w)) # observation probability

    # generate data according to probabilistic model
    z_known = [!all(Y[i] .== 0) for i in 1:n]

    function conditional(y,z,αβ;W=W,X=X)
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
    function joint(y,z,αβ,prior;W=W, X=X)
        return conditional(y,z,αβ;W=W,X=X)*prior(αβ)
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
        ps = [z_known[i] == 1 ? 1 : 1/2*(1+0.95*tanh(ϕ[20+i])) for i in eachindex(X)]
        return pdf(q_αβ, αβ)*prod((ps .^ z) .* ((1 .- ps) .^ (1 .- z)))
    end
    function logq(ϕ,z,αβ; z_known=z_known)
        @views α, β = αβ[1:2], αβ[3:4]
        μ = ϕ[1:4]
        V = reshape(ϕ[5:20], 4, 4)
        #Σinv = Distributions.PDMat(V*V')
        #q_αβ = MvNormalCanon(μ, Σinv*μ, Σinv)
        q_αβ = MvNormal(μ, V*V' + 0.01*I)
        ps = [z_known[i] == 1 ? 1 : 1/2*(1+0.95*tanh(ϕ[20+i])) for i in eachindex(X)]
        return logpdf(q_αβ, αβ) + sum(@. log((ps ^ z) * ((1 - ps) ^ (1 .- z))))
    end

    function sample_q(ϕ; X=X, W=W, z_known=z_known)
        μ = ϕ[1:4]
        V = reshape(ϕ[5:20], 4, 4)
        #Σinv = Distributions.PDMat(V*V')
        #q_αβ = MvNormalCanon(μ, Σinv*μ, Σinv)
        ps = [z_known[i] == 1 ? 1.0 : 1/2*(1+tanh(ϕ[20+i])) for i in eachindex(X)]

        q_αβ = MvNormal(μ, Hermitian(V*V' + 0.01*I))
        αβ = rand(q_αβ)
        @views α, β = αβ[1:2], αβ[3:4]
        q_z = Bernoulli.(ps) 
        return [z_known[i] == 1 ? 1 : rand(q_z[i]) for i in eachindex(q_z)], αβ
    end

    function div(ϕ, z, αβ; y=Y, prior=prior, logprior=logprior, W=W, X=X)
        return logjoint(y, z, αβ, logprior, W=W, X=X) - logq(ϕ,z,αβ)
    end

    function elbo_estimator(ϕ; n=1, y=Y, prior=prior, W=W, X=X)
        elbo = 0.0
        for _ in 1:n
            z, αβ = sample_q(ϕ)
            elbo += div(ϕ, z, αβ, y=y, prior=prior, W=W, X=X)
        end
        return elbo/n
    end

    function optimize_elbo(batch_size, iterations, LEARNING_RATE; n_estimator = batch_size)
        ϕ0 = randn(n+20)
        obj = StochasticModel(ϕ -> -elbo_estimator(ϕ, n= batch_size), ϕ0)
        elbo_trace = Float64[]
        optimizer = Adam(LEARNING_RATE)
        setup = Optimisers.setup(optimizer, obj)
        for i in 1:iterations
            grad = stochastic_gradient(obj)
            if sum(isinf.(grad.p)) == 0 && sum(isnan.(grad.p)) == 0
                Optimisers.update!(setup, obj, grad)
                elbo = elbo_estimator(obj.p, n=n_estimator)
                println("iteration $i -- ELBO = $elbo")
                push!(elbo_trace, elbo)
            else
                elbo = elbo_estimator(obj.p, n=n_estimator)
                println("iteration $i -- ELBO = $elbo")
                push!(elbo_trace, elbo)
            end
        end
        return elbo_trace, obj.p
    end

    function plot_opt_trajectories(batch_sizes, traces)
        fig = Figure(fontsize=18)
        ax = Axis(fig[1,1], xlabel = "Iterations", ylabel = "ELBO", xscale=log2)
        for batch_size in batch_sizes
            lines!(ax, traces[batch_size], label = "batch size = $batch_size", linewidth=2)
        end
        axislegend(position=:rb)
        fig
    end

    # now an asymptotically correct Markov Chain approach
    include("turing_model.jl")

    n_chain = 500000
    sampler = NUTS(1000, 0.65)
    chain = sample(marginal_occupancy(αβ_prior,Y,z_known,W,X), sampler, n_chain, drop_warmup=true)

    function plot_chain(chain; burnin = 1000)
        fig = Figure(fontsize=28)
        ax_α = Axis(fig[1,1], xlabel = L"\alpha_1", ylabel=L"\alpha_2")
        scatter!(ax_α, chain.value[burnin:end, chain.value.axes[2][1]].data[:], 
                    chain.value[burnin:end, chain.value.axes[2][2]].data[:], 
                    color = (:black,0.5))
        ax_β = Axis(fig[1,2], xlabel = L"\beta_1", ylabel=L"\beta_2")
        scatter!(ax_β, chain.value[burnin:end, chain.value.axes[2][3]].data[:], 
                    chain.value[burnin:end, chain.value.axes[2][4]].data[:], 
                    color = (:black,0.5))
        fig
    end

    function plot_gaussian(Σ, μ, width)
        U, S, _ = svd(Σ)
        Σ_inv = inv(Σ)
        x1_range = range(-sqrt(S[1])*width,sqrt(S[1])*width,length=100)
        x2_range = range(-sqrt(S[2])*width,sqrt(S[2])*width,length=100)
        coords_x = [μ[1] .+ U[1,:]'*[x1,x2] for x1 in x1_range, x2 in x2_range]
        coords_y = [μ[2] .+ U[2,:]'*[x1,x2] for x1 in x1_range, x2 in x2_range]
        X = range(minimum(coords_x), maximum(coords_x), length=100)
        Y = range(minimum(coords_y), maximum(coords_y), length=100)
        Z = [1/2*([x,y]-μ)'*Σ_inv*([x,y]-μ) for x in X, y in Y]
        return X,Y,Z
    end

    function plot_results(chain, ϕ; burnin = 1000, width = 5, thinning = 1000)
        μ, V = ϕ[1:4], reshape(ϕ[5:20],4,4)
        Σ = V*V' + 0.01*I
        symbols = [Symbol("αβ[$i]") for i in 1:4]
        vars = [L"\alpha_1", L"\alpha_2", L"\beta_1", L"\beta_2"]
        chain_mean = [mean(chain[par][burnin:end]) for par in symbols]

        fig = Figure(fontsize=28, resolution = (1100,1100))

        axs = [[Axis(fig[i,j-1], xlabel = vars[j], ylabel=vars[i]) for j in i+1:4] for i in 1:3]
        burnin = 25000
        for i in 1:3
            k = 0
            for j in i+1:4
                k += 1
                X, Y, Z = plot_gaussian(Σ[[j,i],[j,i]], μ[[j,i]], width)
                scatter!(axs[i][k], chain[symbols[j]][burnin:end].data[1:thinning:end], 
                                    chain[symbols[i]][burnin:end].data[1:thinning:end], 
                                    color = (:black,0.2))
                global cf = contour!(axs[i][k], X, Y, -Z, linewidth = 2, 
                                    colorrange = (-width, 0), 
                                    colormap=:OrRd, levels= -width:1:-1)
                scatter!(axs[i][k], [chain_mean[j]], [chain_mean[i]], 
                                    color = :yellow, marker = :cross)
                scatter!(axs[i][k], [μ[j]], [μ[i]], marker = :circle, color = :red)
                #scatter!(axs[i][k], [true_pars[j]], [true_pars[i]], marker = :star8, color = :blue)
            end
        end
        legends = [[MarkerElement(marker=:circle, color = :red),
                    MarkerElement(marker=:cross, color = :yellow),
                    #MarkerElement(marker=:star8, color = :blue),
                    MarkerElement(marker=:circle, color = (:black, 0.2))]]
        labels = [[L"\text{VI mean}", 
                L"\text{HMC mean}", 
                #L"\text{True value}",
                L"\text{HMC samples}"]]
        ga = fig[2:3, 1] = GridLayout()
        Legend(ga[1,1], legends, labels, ["Legend"])
        Colorbar(ga[2,1], cf, #ticks=(collect(-width:1:0), ["$i" for i in abs.(-width:1:0)]),
                        label = L"\propto \log \, q", width=30)
        #colgap!(ga, 120)
        Label(ga[2,1,Top()], "Variational Distribution", font = :bold, padding = (0, 0, 10, 0))
        fig
    end

    opt_trace, ϕ_opt = optimize_elbo(5, 500, 0.05)
    fig = plot_results(chain, ϕ_opt, burnin=25000, thinning = 100)
    save("../figures/case_study_$(spec).pdf",fig)

    # just plot the marginals like in Fig 3 of the plot_opt_trajectories
    function plot_results(chain, ϕ; burnin = 1000, width = 4, opacity=0.1)
        μ, V = ϕ[1:4], reshape(ϕ[5:20],4,4)
        Σ = V*V' + 0.01*I
        vars = [L"\alpha_1", L"\alpha_2", L"\beta_1", L"\beta_2"]
        fig = Figure(fontsize=24, resolution = (1100,500))
        axs = [Axis(fig[1,i], xminorgridvisible=false, xgridvisible=false,
                            yminorgridvisible=false, ygridvisible=false,
                            xlabel = vars[i]) for i in 1:4]
        for i in 1:4
            dist = Normal(μ[i], sqrt.(Σ[i,i]))
            x_range = range(dist.μ-width*dist.σ, dist.μ+width*dist.σ, 100)
            samples = chain.value[burnin:end, chain.value.axes[2][i]].data[:]
            hist!(axs[i], samples, normalization=:pdf, strokcolor=:black, strokewidth=2, color=(:gray, opacity))
            lines!(axs[i], x_range, pdf.(dist, x_range), color = :red, linewidth=2)
        end
        
        fig
    end
    fig = plot_results(chain, ϕ_opt, burnin=25000, opacity=0.0)
    save("../figures/case_study_marginals_$(spec).pdf",fig)
end