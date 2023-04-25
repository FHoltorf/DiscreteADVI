include("model.jl") # model specifications 
include("utils.jl") # optimization routines
include("visualizations.jl") # plotting routines

using FileIO

# number of sites
n_range = [5^i for i in 1:5]
n_batches = [1]

results = Dict()
# number of visits per site
for n in n_range
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

    # generate data according to probabilistic model
    α_true = [1.35,1.75]#[0, 1.75]
    β_true = [-0.1,2.5]#[-1.85, 2.5]
    Z = [rand(Bernoulli(ψ(X[i], β_true))) for i in 1:n] # true latent variables
    Y = [[rand(Bernoulli(Z[i]*d(W[i][j], α_true))) for j in 1:K[i]] for i in 1:n] # observations
    z_known = [!all(Y[i] .== 0) for i in 1:n] # flag for which latent variables are known with certainty

    ϕ0 = randn(n+20)
    for n_batch in n_batches
        ϕ_opt, ϕ_trace, times, elbo_trace, log_pred_trace = optimize_elbo(ϕ0, n_batch, 200, 0.05, n_snapshots=5, n_estimator = 1000)
        results[n, n_batch] = (ϕ_trace, times, elbo_trace, log_pred_trace)
    end
end

# save results since expensive to compute
save("results/scaling_results.jld", "results", results)

fig = Figure(fontsize=28)
ax = Axis(fig[1,1],
          ylabel = "wall clock time [s]", 
          xlabel = "Number of sites n",
          xminorgridvisible = true,
          yminorgridvisible = true,
          xminorticks = IntervalsBetween(8),
          yminorticks = IntervalsBetween(8),
          xscale = log10,
          yscale = log10)
times = [results[n,1][2][end] for n in n_range]
logtimes = log10.(times)
logn = log10.(n_range)
a = hcat(logn[2:end], ones(length(logn)-1))\logtimes[2:end]
scatter!(ax, n_range, [results[n,1][2][end] for n in n_range], color = :black, markersize=20)
lines!(ax, n_range[2:end], 10 .^ (hcat(logn[2:end], ones(length(logn)-1))*a), color = :black, linewidth=2)
lines!(ax, [200,500], 10 .^ [[log10(200), 1]'*a, [log10(200), 1]'*a], linewidth=2, color = :red)
lines!(ax, [500,500], 10 .^ [[log10(200), 1]'*a, [log10(500), 1]'*a], linewidth=2, color = :red)
text!(ax, [550], [70], text=[string(round(a[1], digits=3))], fontsize=28)
fig
save("../figures/scaling.pdf",fig)

# scaling of just the gradients
grad_results = Dict()
for n in n_range
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

    # generate data according to probabilistic model
    α_true = [1.35,1.75]#[0, 1.75]
    β_true = [-0.1,2.5]#[-1.85, 2.5]
    Z = [rand(Bernoulli(ψ(X[i], β_true))) for i in 1:n] # true latent variables
    Y = [[rand(Bernoulli(Z[i]*d(W[i][j], α_true))) for j in 1:K[i]] for i in 1:n] # observations
    z_known = [!all(Y[i] .== 0) for i in 1:n] # flag for which latent variables are known with certainty

    ϕ0 = randn(n+20)
    
    obj = StochasticModel(ϕ -> -elbo_estimator(ϕ, n=1), ϕ0)
   
    grad_results[n] = @benchmark grad = stochastic_gradient($obj)
end

fig = Figure(fontsize=28)
ax = Axis(fig[1,1],
          ylabel = "wall clock time [s]", 
          xlabel = "number of sites n",
          xminorgridvisible = true,
          yminorgridvisible = true,
          xminorticks = IntervalsBetween(8),
          yminorticks = IntervalsBetween(8),
          xscale = log10,
          yscale = log10)
times = [mean(grad_results[n]).time*1e-9 for n in n_range]
logtimes = log10.(times)
logn = log10.(n_range)
a = hcat(logn, ones(length(logn)))\logtimes
scatter!(ax, n_range, times, color = :black, markersize=20)
lines!(ax, n_range, 10 .^ (hcat(logn, ones(length(logn)))*a), color = :black, linewidth=2)
lines!(ax, [200,500], 10 .^ [[log10(200), 1]'*a, [log10(200), 1]'*a], linewidth=2, color = :red)
lines!(ax, [500,500], 10 .^ [[log10(200), 1]'*a, [log10(500), 1]'*a], linewidth=2, color = :red)
text!(ax, [550], 10 .^ [[log10(250), 1]'*a], text = [string(round(a[1], digits=3))], fontsize=28)
fig

save("../figures/grad_scaling.pdf",fig)
