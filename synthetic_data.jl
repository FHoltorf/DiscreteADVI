include("model_reduced.jl") # model specifications 
include("utils.jl") # optimization routines
include("visualizations.jl") # plotting routines

# number of sites
n = 25

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

# generate data according to probabilistic model
α_true = [1.35,1.75]#[0, 1.75]
β_true = [-0.1,2.5]#[-1.85, 2.5]
Z = [rand(Bernoulli(ψ(X[i], β_true))) for i in 1:n] # true latent variables
Y = [[rand(Bernoulli(Z[i]*d(W[i][j], α_true))) for j in 1:K[i]] for i in 1:n] # observations
z_known = [!all(Y[i] .== 0) for i in 1:n] # flag for which latent variables are known with certainty

# MCMC inference
n_chain = 100000
sampler = NUTS() 
t_NUTS = @elapsed chain_NUTS = sample(marginal_occupancy(Y), sampler, n_chain, drop_warmup=false)
NUTS_log_predictive_trace = cumsum([logmarginal(Y,z_known,ab,W=W,X=X) for ab in eachrow(chain_NUTS[chain_NUTS.name_map[1]].value.data[:,:,1])]) ./ (1:length(chain_NUTS[:log_density]))
NUTS_times = range(0.0, t_NUTS, length=length(NUTS_log_predictive_trace))
plot_chain(chain_NUTS, burnin=25000)

sampler = Gibbs(HMC(0.05,1000,:αβ), PG(2, :z)) 
t_Gibbs = @elapsed chain_Gibbs = sample(occupancy(Y), sampler, 20000, init_theta = vcat(zeros(4), ones(n-sum(z_known))))
Gibbs_loglikelihood = [loglikelihood(Y,pars[5:end],pars[1:4],W=W,X=X) for pars in eachrow(chain_Gibbs[chain_Gibbs.name_map[1]].value.data[:,:,1])]
Gibbs_log_predictive_trace = cumsum(Gibbs_loglikelihood) ./ (1:length(Gibbs_loglikelihood))
Gibbs_times = range(0.0, t_Gibbs, length=length(Gibbs_log_predictive_trace))
plot_chain(chain_Gibbs, burnin=5000)

# compare VI trajectories
ϕ0 = randn(20+n-sum(z_known))
struct OptResults
    ϕ_opt
    ϕ_trace
    times
    elbo_trace
    log_predictive_trace
end
# batch size = 10
res10 = OptResults(optimize_elbo(deepcopy(ϕ0), 10, 500, 0.05, n_snapshots=10, n_estimator = 1000)...)
fig = pairplot(chain_NUTS, res10.ϕ_opt, burnin=25000, vcat(α_true, β_true), thinning = 100)
save("../figures/VI_synthetic_10.pdf",fig)
pairplot_movie(chain_NUTS, res10.ϕ_trace, burnin=25000, vcat(α_true, β_true), filename = "VI_movie_10.mp4", thinning = 100)

# batch size = 1
res1 = OptResults(optimize_elbo(deepcopy(ϕ0), 1, 500, 0.05, n_snapshots=10, n_estimator = 1000)...)
fig = pairplot(chain_NUTS, res1.ϕ_opt, burnin=25000, vcat(α_true, β_true), thinning = 100)
save("../figures/VI_synthetic_1.pdf",fig)
pairplot_movie(chain_NUTS, res1.ϕ_trace, burnin=25000, vcat(α_true, β_true), filename = "VI_movie_1.mp4", thinning = 100)

# compare posterior predictive
fig = Figure(fontsize=28)
ax = Axis(fig[1,1], xlabel = "time [s]", 
                    xminorgridvisible = true,
                    xminorticks = IntervalsBetween(8),
                    ylabel = "log predictive posterior",
                    xticks = LogTicks(IntegerTicks()),
                    xscale = log10)
ylims!(ax, -80, -30)
lines!(ax, NUTS_times[2:end], NUTS_log_predictive_trace[2:end], color = :black, linewidth = 2, label = "NUTS + marginalization")
lines!(ax, Gibbs_times[2:end], Gibbs_log_predictive_trace[2:end], color = :orange, linewidth = 2, label = "Gibbs(HMC, PG)")
lines!(ax, res1.times[2:end], res1.log_predictive_trace, color = :red, linewidth=2, label = "VI (m=1)")
lines!(ax, res10.times[2:end], res10.log_predictive_trace, color = :blue, linewidth=2, label = "VI (m=10)")
axislegend(position=:lb)
fig
save("../figures/log_predictive_vs_time.pdf",fig)