include("model.jl") # model specifications 
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

# evaluation
n_chain = 100000
sampler = NUTS() # HMC(0.05, 1000) 
t_NUTS = @elapsed chain = sample(occupancy(Y), sampler, n_chain, drop_warmup=false)
MCMC_log_predictive_trace = cumsum([logmarginal(Y,z_known,ab,W=W,X=X) for ab in eachrow(chain[chain.name_map[1]].value.data[:,:,1])]) ./ (1:length(chain[:log_density]))
MCMC_times = range(0.0, t_NUTS, length=length(MCMC_log_predictive_trace))

# compare VI trajectories
ϕ0 = randn(20+n)

# batch size = 10
ϕ_opt, ϕ_trace, times, elbo_trace, log_predictive_trace = optimize_elbo(ϕ0, 10, 500, 0.05, n_snapshots=1000, n_estimator = 1000)
fig = pairplot(chain, ϕ_opt, burnin=25000, vcat(α_true, β_true), thinning = 100)
save("../figures/progress_report.pdf",fig)
pairplot_movie(chain, ϕ_trace, burnin=25000, vcat(α_true, β_true), filename = "VI_movie_10.mp4", thinning = 100)

# batch size = 1
ϕ_opt, ϕ_trace, times, elbo_trace, log_predictive_trace = optimize_elbo(ϕ0, 1, 500, 0.05, n_snapshots=10, n_estimator = 1000)
fig = pairplot(chain, ϕ_opt, burnin=25000, vcat(α_true, β_true), thinning = 100)
save("../figures/progress_report.pdf",fig)
pairplot_movie(chain, ϕ_trace, burnin=25000, vcat(α_true, β_true), filename = "VI_movie_1.mp4", thinning = 100)