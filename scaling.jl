include("model.jl") # model specifications 
include("utils.jl") # optimization routines
include("visualizations.jl") # plotting routines

# number of sites
n_range = [5^i for i in 1:4]

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

    results = Dict()
    n_batches = [1,5,10,20]
    ϕ0 = randn(n+20)
    for n_batch in n_batches
        ϕ_opt, ϕ_trace, times, elbo_trace, log_pred_trace = optimize_elbo(ϕ0, n_batch, 200, 0.05, n_snapshots=5, n_estimator = 1000)
        results[n_batch] = (ϕ_trace, times, elbo_trace, log_pred_trace)
    end
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