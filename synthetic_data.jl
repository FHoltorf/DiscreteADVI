cd(@__DIR__)
using Pkg
Pkg.activate(".")

using StochasticAD, Random, Distributions, Optimisers, CairoMakie, LinearAlgebra, DynamicPPL

Random.seed!(123)

# number of sites
n = 20

# number of visits per site
K = [4 for i in 1:n]

# covariates 
x_dst = Uniform(-2,2)
w_dst = Uniform(-5,5)
X = [vcat((rand(x_dst, 1) .- mean(x_dst)) ./ std(x_dst),1) for i in 1:n]
W = [[vcat((rand(w_dst, 1) .- mean(w_dst)) ./ std(w_dst),1) for j in 1:K[i]] for i in eachindex(K)]

# regression coeff priors
αβ_prior = MvNormal(Float64[0,0,0,0], 1000*I)
prior(αβ; αβ_prior=αβ_prior) = pdf(αβ_prior, αβ)
logprior(αβ; αβ_prior=αβ_prior) = logpdf(αβ_prior, αβ)

# probabilistc model
ψ(x,β) = 0.001 + 0.999/(1+exp(-β'*x)) # occupation probability
d(w,α) = 0.001 + 0.999/(1+exp(-α'*w)) # observation probability

# generate data according to probabilistic model
α_true = [1.35,1.75]#[0, 1.75]
β_true = [-0.1,2.5]#[-1.85, 2.5]
z_latent = [rand(Bernoulli(ψ(X[i], β_true))) for i in 1:n]
Y = [[rand(Bernoulli(z_latent[i]*d(W[i][j], α_true))) for j in 1:K[i]] for i in 1:n]
z_known = [!all(Y[i] .== 0) for i in 1:n]

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
function joint(y,z,αβ,prior;W=W, X=X)
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
    ps = [z_known[i] == 1 ? 1 : 1/2*(1+0.99*tanh(ϕ[20+i])) for i in eachindex(X)]
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
    return logpdf(q_αβ, αβ) + sum(@. log((ps ^ z) * ((1 - ps) ^ (1 - z))))
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
function predictive_posterior_estimator(ϕ; n=1, y=Y, W=W, X=X)
    log_predictive = 0.0
    for _ in 1:n
        z, αβ = sample_q(ϕ)
        log_predictive += loglikelihood(y, z, αβ, W=W, X=X)
    end
    return log_predictive/n
end
function optimize_elbo(batch_size, iterations, LEARNING_RATE; n_estimator = batch_size, n_snapshots = 1)
    ϕ0 = randn(n+20)
    obj = StochasticModel(ϕ -> -elbo_estimator(ϕ, n= batch_size), ϕ0)
    elbo_trace = Float64[]
    log_predictive_trace = Float64[]
    wall_times = Float64[]
    ϕ_trace = Vector{Float64}[]
    optimizer = Adam(LEARNING_RATE)
    setup = Optimisers.setup(optimizer, obj)
    t = 0
    elbo = elbo_estimator(obj.p, n=n_estimator)
    push!(elbo_trace, elbo)
    push!(wall_times, t)
    for i in 1:iterations
        t += @elapsed grad = stochastic_gradient(obj)
        if sum(isinf.(grad.p)) == 0
            t += @elapsed Optimisers.update!(setup, obj, grad)
            push!(ϕ_trace, deepcopy(obj.p))
            println("iteration $i")
        end
        if i % n_snapshots == 0
            elbo = elbo_estimator(obj.p, n=n_estimator)
            log_predictive = predictive_posterior_estimator(obj.p, n=n_estimator)
            push!(log_predictive_trace, log_predictive)
            push!(elbo_trace, elbo)
            push!(wall_times, t)
        end
    end
    elbo = elbo_estimator(obj.p, n=n_estimator)
    log_predictive = predictive_posterior_estimator(obj.p, n=n_estimator)
    push!(log_predictive_trace, log_predictive)
    push!(elbo_trace, elbo)
    push!(wall_times, t)
    return obj.p, ϕ_trace, wall_times, elbo_trace, log_predictive_trace
end

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

function plot_results(chain, ϕ, true_pars; burnin = 5000, width = 5, thinning = 1000)
    μ, V = ϕ[1:4], reshape(ϕ[5:20],4,4)
    Σ = V*V' + 0.01*I
    symbols = [Symbol("αβ[$i]") for i in 1:4]
    vars = [L"\alpha_1", L"\alpha_2", L"\beta_1", L"\beta_2"]
    chain_mean = [mean(chain[par][burnin:end]) for par in symbols]

    fig = Figure(fontsize=28, resolution = (1000,1000))

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
            scatter!(axs[i][k], [true_pars[j]], [true_pars[i]], marker = :star8, color = :blue)
        end
    end
    legends = [[MarkerElement(marker=:circle, color = :red),
                MarkerElement(marker=:cross, color = :yellow),
                MarkerElement(marker=:star8, color = :blue),
                MarkerElement(marker=:circle, color = (:black, 0.2))]]
    labels = [[L"\text{VI mean}", 
              L"\text{HMC mean}", 
              L"\text{True value}",
              L"\text{HMC samples}"]]
    ga = fig[2:3, 1] = GridLayout()
    Legend(ga[1,1], legends, labels, ["Legend"])
    Colorbar(ga[2,1], cf, #ticks=(collect(-width:1:0), ["$i" for i in abs.(-width:1:0)]),
                      label = L"\propto \log \, q", width=30)
    #colgap!(ga, 120)
    Label(ga[2,1,Top()], "Variational Distribution", font = :bold, padding = (0, 0, 10, 0))
    fig
end

# plot a movie 
function plot_results_movie(chain, ϕ_trace, true_pars; filename = "VI_movie.mp4", burnin = 5000, width = 5, thinning = 1000)
    ϕ = ϕ_trace[end]
    obs = Dict()

    μ, V = ϕ[1:4], reshape(ϕ[5:20],4,4)
    Σ = V*V' + 0.01*I
    symbols = [Symbol("αβ[$i]") for i in 1:4]
    vars = [L"\alpha_1", L"\alpha_2", L"\beta_1", L"\beta_2"]
    chain_mean = [mean(chain[par][burnin:end]) for par in symbols]

    fig = Figure(fontsize=28, resolution = (1000,1000))

    axs = [[Axis(fig[i,j-1], xlabel = vars[j], ylabel=vars[i]) for j in i+1:4] for i in 1:3]
    burnin = 25000
    for i in 1:3
        k = 0
        for j in i+1:4
            k += 1
            scatter!(axs[i][k], chain[symbols[j]][burnin:end].data[1:thinning:end], 
                                chain[symbols[i]][burnin:end].data[1:thinning:end], 
                                color = (:black,0.2))
            scatter!(axs[i][k], [chain_mean[j]], [chain_mean[i]], 
                                color = :yellow, marker = :cross)
            scatter!(axs[i][k], [true_pars[j]], [true_pars[i]], marker = :star8, color = :blue)

            X, Y, Z = plot_gaussian(Σ[[j,i],[j,i]], μ[[j,i]], width)
            obs[i,j] = Observable(X), Observable(Y), Observable(-Z), Observable(Point2(μ[j], μ[i]))
                                
            global cf = contour!(axs[i][k], obs[i,j][1], obs[i,j][2], obs[i,j][3], linewidth = 2, 
                                 colorrange = (-width, 0), 
                                 colormap=:OrRd, levels= -width:1:-1)
            
            scatter!(axs[i][k], obs[i,j][4], marker = :circle, color = :red)
        end
    end
    legends = [[MarkerElement(marker=:circle, color = :red),
                MarkerElement(marker=:cross, color = :yellow),
                MarkerElement(marker=:star8, color = :blue),
                MarkerElement(marker=:circle, color = (:black, 0.2))]]
    labels = [[L"\text{VI mean}", 
              L"\text{HMC mean}", 
              L"\text{True value}",
              L"\text{HMC samples}"]]
    ga = fig[2:3, 1] = GridLayout()
    Legend(ga[1,1], legends, labels, ["Legend"])
    Colorbar(ga[2,1], cf, #ticks=(collect(-width:1:0), ["$i" for i in abs.(-width:1:0)]),
                      label = L"\propto \log \, q", width=30)
    Label(ga[2,1,Top()], "Variational Distribution", font = :bold, padding = (0, 0, 10, 0))
    #it_label = Observable(1)
    #gb = fig[3, 2] = GridLayout()
    #it = lift(i -> string("Iteration = ", it_label.val), 1)
    #Label(gb[1,1], it)#, font = :bold)
    
    record(fig, "../figures/"*filename, enumerate(ϕ_trace)) do (iter,ϕ)
        μ, V = ϕ[1:4], reshape(ϕ[5:20],4,4)
        Σ = V*V' + 0.01*I
        for i in 1:3
            for j in i+1:4
                X, Y, Z = plot_gaussian(Σ[[j,i],[j,i]], μ[[j,i]], width)
                obs[i,j][1][] = X
                obs[i,j][2][] = Y
                obs[i,j][3][] = -Z
                obs[i,j][4][] = Point2(μ[j], μ[i])
            end
        end
        #it_label[] = iter
        #it_label = it_label[]
    end
end

# sampling
include("turing_model.jl")

n_chain = 100000
sampler = NUTS() # HMC(0.05, 1000) 
t_NUTS = @elapsed chain = sample(occupancy(Y), sampler, n_chain, drop_warmup=false)
#MCMC_log_predictive_trace = cumsum(chain[:log_density][:,1]) ./ (1:length(chain[:log_density]))
MCMC_log_predictive_trace = cumsum([logmarginal(Y,z_known,ab,W=W,X=X) for ab in eachrow(chain[chain.name_map[1]].value.data[:,:,1])]) ./ (1:length(chain[:log_density]))
MCMC_times = range(0.0, t_NUTS, length=length(MCMC_log_predictive_trace))

# p(theta|Y) p(Y) = p(Y, theta) = p(Y|theta) p(theta)
# P(Y) = p(Y|theta) * p(theta)/p(theta|Y) 

# get a trajectory
# compare VI trajectories
# batch size = 10
ϕ_opt, ϕ_trace, times, elbo_trace, log_predictive_trace = optimize_elbo(10, 500, 0.05, n_snapshots=1000, n_estimator = 1000)
fig = plot_results(chain, ϕ_opt, burnin=25000, vcat(α_true, β_true), thinning = 100)
save("../figures/progress_report.pdf",fig)
plot_results_movie(chain, ϕ_trace, burnin=25000, vcat(α_true, β_true), filename = "VI_movie_10.mp4", thinning = 100)

# batch size = 1
ϕ_opt, ϕ_trace, times, elbo_trace, log_predictive_trace = optimize_elbo(1, 500, 0.05, n_snapshots=10, n_estimator = 1000)
fig = plot_results(chain, ϕ_opt, burnin=25000, vcat(α_true, β_true), thinning = 100)
save("../figures/progress_report.pdf",fig)
plot_results_movie(chain, ϕ_trace, burnin=25000, vcat(α_true, β_true), filename = "VI_movie_1.mp4", thinning = 100)

fig = Figure(fontsize=28)
ax = Axis(fig[1,1],xlabel=L"\text{time}",ylabel=L"\text{log predictive posterior}", xscale=log10)
lines!(ax, MCMC_times[2:end], MCMC_log_predictive_trace[2:end])
lines!(ax, times[2:end], log_predictive_trace)
fig

results = Dict()
n_batches = [1,10,50]
for n in n_batches
    ϕ_opt, ϕ_trace, times, elbo_trace, log_pred_trace = optimize_elbo(n, 200, 0.05, n_snapshots=5, n_estimator = 1000)
    results[n] = (ϕ_trace, times, elbo_trace, log_pred_trace)
end

fig = Figure(fontsize=28)
ax = Axis(fig[1,1],xlabel=L"\text{time [s]}",ylabel=L"\text{log predictive posterior}", xscale=log10)
ylims!(ax, -100, 0)
for n in n_batches
    lines!(ax, results[n][2][2:end], results[n][4], label = "m = $n", linewidth=3)
end
lines!(ax, MCMC_times[2:end], MCMC_log_predictive_trace[2:end], label = "NUTS", linewidth=3)
axislegend()
fig


# compare convergence in terms of log posterior predictive
#n_estimator = [1, 10, 50]
#for n in n_estimator
#    ϕ_opt, ϕ_trace, times, elbo_trace = optimize_elbo(1, 500, 0.05, n_snapshots=1000, n_estimator = 1000)

#end


# plot convergence
#=
batch_sizes = [1, 10, 50]
n_iterations = 100
step_size = 0.05
n_sample = [2000, 200, 10]

results = Dict()
for (i,batch_size) in enumerate(batch_sizes)
    sample_traces = zeros(0, n_iterations)
    for n in 1:n_sample[i]
        opt_trace, ϕ_opt = optimize_elbo(batch_size, n_iterations, step_size)
        sample_traces = vcat(sample_traces, opt_trace')
    end
    results[batch_size] = Dict(:mean => [mean(r) for r in eachcol(sample_traces)],
                               :std => [std(r) for r in eachcol(sample_traces)])
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

function plot_sgd_traces(results; colors = [:red, :blue, :purple, :dodgerblue, :orange],
                                  alpha = 0.4)
    fig = Figure(fontsize=28)
    ax = Axis(fig[1,1], xlabel = "Iteration", ylabel = "ELBO")
    k = 0
    for batch_size in keys(results)
        k += 1
        band!(ax, 1:length(results[batch_size][:mean]),
                  results[batch_size][:mean] .- results[batch_size][:std],
                  results[batch_size][:mean] .+ results[batch_size][:std], 
                  color = (colors[k], alpha))
        lines!(ax, results[batch_size][:mean], 
                   color = colors[k], label = "sample size = $batch_size", 
                   linewidth=4)
    end
    axislegend(position = :rb)
    fig
end
fig = plot_sgd_traces(results, alpha = 0.2)
save("../figures/sgd_traces.pdf",fig)
=#