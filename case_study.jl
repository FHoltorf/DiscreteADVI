include("model_reduced.jl")
include("visualizations.jl")
include("utils.jl")

using RData, DataFrames, Dates, CategoricalArrays, FileIO

specs = [1,2,3,4,5]
for spec in specs
    data = load("design_mats/design_mat$(spec).Rdata")
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

    # generate data according to probabilistic model
    z_known = [!all(Y[i] .== 0) for i in 1:n]

    n_chain = 100000
    sampler = NUTS() #1000, 0.65)
    t_MCMC = @elapsed chain = sample(marginal_occupancy(Y), sampler, n_chain, drop_warmup=true)

    ϕ0 = randn(20+n-sum(z_known))
    ϕ_opt, ϕ_trace, times, elbo_trace, log_predictive_trace = optimize_elbo(ϕ0, 5, 500, 0.05, n_snapshots=10, n_estimator = 1000)
    fig = pairplot(chain, ϕ_opt, [], burnin=25000, thinning = 100)
    save("../figures/case_study_$(spec).pdf",fig)

    fig = marginals(chain, ϕ_opt, burnin=25000, opacity=0.0)
    save("../figures/case_study_marginals_$(spec).pdf",fig)

    fig = PAO_plot(ϕ_opt, chain, z_known=z_known)
    save("../figures/PAO_dist_$(spec).pdf")
    
    save("results/case_study.jld2", "phi_opt", ϕ_opt, 
                                    "phi_trace", ϕ_trace,
                                    "times", times, 
                                    "elbo_trace", elbo_trace,
                                    "log_predictive_trace", log_predictive_trace,
                                    "chain", chain,
                                    "t_MCMC", t_MCMC)
end