include("model_fast.jl")
include("visualizations.jl")
include("utils.jl")
N_adaptation = 1000
N_chain = 100000
include("jags_model.jl")

using RData, DataFrames, Dates, CategoricalArrays, FileIO

#y_min_spec = [-260, -260,-240,-270,-220]
specs = [5]
N_samples = 50
for spec in specs
    Random.seed!(123)
    data = load("design_mats/design_mat$(spec).Rdata")
    function organize_data(data)
        X = [vcat(x, 1) for x in data["design_mats"]["X"].AET_div_PET_s]
        Y = Vector{Bool}[]
        W = Vector{Vector{Float64}}[]
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
    om = OccupancyModel(Y,Bool[],W,X,αβ_prior)

    z_known = om.z_known

    # n_chain = 100000
    # # NUTS
    # sampler = NUTS(50, 0.65) #1000, 0.65)
    # t_MCMC = @elapsed chain_NUTS = sample(marginal_occupancy(om), sampler, n_chain, drop_warmup=true)
    # lp_trace_NUTS = compute_log_posterior_trace(chain_NUTS, chain_NUTS.name_map[1], om, :marginal)

    # JAGS  
    W_mat = zeros(n, 6)
    Y_mat = zeros(n, 6)
    for i in 1:length(W)
        W_mat[i,1:length(W[i])] = [w[1] for w in W[i]]
        Y_mat[i,1:length(Y[i])] = Y[i]
    end
    X_mat = [x[1] for x in X]

    data = Dict(
            "x" => X_mat,
            "w" => W_mat,
            "y" => Y_mat,
            "n" => n,
            "alpha_mean" => [0.0,0.0],
            "alpha_prec" => 1e-3*Matrix(I,2,2),
            "beta_mean" => [0.0,0.0],
            "beta_prec" => 1e-3*Matrix(I,2,2),
            "K" => K,
            )

    #ϕ0 = randn(14+n-sum(z_known)) 
    traces_ensemble = []
    times_ensemble = []
    for i in 1:N_samples 
        ϕ0 = randn(20+n-sum(z_known))  #vcat(zeros(4), ones(10), zeros(20+n-sum(z_known))) 

        inits = [
            Dict("alpha" => ϕ0[1:2], "beta" => ϕ0[3:4], "z" => Float64[z_known[i] == 1 ? 1 : rand(0:1) for i in eachindex(X)]),
            ]

        t_jags = @elapsed chain_JAGS = jags(jagsmodel, data, inits, ProjDir)
        jags_parnames = [Symbol("alpha[1]"), Symbol("alpha[2]"), Symbol("beta[1]"), Symbol("beta[2]")]
        lp_trace_JAGS = compute_log_posterior_trace(chain_JAGS, [Symbol("alpha[1]"), Symbol("alpha[2]"), 
                                                                Symbol("beta[1]"), Symbol("beta[2]")], om) 
        
        # NUTS                                                         
        n_chain = N_chain
        sampler = NUTS(N_adaptation, 0.65)
        t_NUTS = @elapsed chain_NUTS = sample(marginal_occupancy(om), sampler, n_chain, drop_warmup=true, init_theta=ϕ0[1:4])
        lp_trace_NUTS = compute_log_posterior_trace(chain_NUTS, chain_NUTS.name_map[1], om)
        
        # variational inference
        
        # 5, 500, 0.05, 100 is good but kinda slow
        ϕ_opt, ϕ_trace, VI_times, elbo_trace, log_predictive_trace = optimize_elbo(om, ϕ0, 1, 1000, 0.05, n_switch=300, n_snapshots=10, n_estimator = 1000)

        traces = [log_predictive_trace, lp_trace_JAGS, lp_trace_NUTS]
        times = [VI_times[2:end], range(0,t_jags,length=length(lp_trace_JAGS)+1)[2:end], range(0, t_NUTS, length=length(lp_trace_NUTS)+1)[2:end]]
        names = ["VI", "JAGS", "NUTS + marginalization"]
        colors = [:black, :red, :blue]
        
        push!(traces_ensemble, traces)
        push!(times_ensemble, times)
        # visualizations
        if i == 1
            # make visualizations
            fig = log_predictive_trace_plot(times, traces, names, ymin = log_predictive_trace[end] - 20, colors = colors)
            display(fig)
            save("../figures/case_study_post_$(spec).pdf", fig)

            fig = pairplot(chain_NUTS, ϕ_opt, [], burnin=25000, thinning = 100)
            save("../figures/case_study_pair_$(spec).pdf",fig)

            fig = marginals(chain_NUTS, ϕ_opt, burnin=25000, opacity=0.0, linecolor=:gray)
            save("../figures/case_study_marginals_$(spec).pdf",fig)

            fig = PAO_plot_JAGS(ϕ_opt, chain_JAGS, om)
            save("../figures/PAO_dist_JAGS_$(spec).pdf",fig)    
            
            fig = PAO_plot_JAGS(ϕ_opt, chain_JAGS, om)
            save("../figures/PAO_dist_NUTS_$(spec).pdf",fig) 
            
            # save results
            save("results/case_study_$(spec).jld2", "phi_opt", ϕ_opt, 
                                                "phi_trace", ϕ_trace,
                                                "times", VI_times, 
                                                "elbo_trace", elbo_trace,
                                                "log_predictive_trace", log_predictive_trace,
                                                "chain_JAGS", chain_JAGS,
                                                "chain_NUTS", chain_NUTS,
                                                "t_jags", t_jags,
                                                "t_NUTS", t_NUTS)   
        end
        push!(traces_ensemble, traces)
        push!(times_ensemble, times)
    end
    mean_traces = [mean([traces[k] for traces in traces_ensemble]) for k in 1:3]
    spreads = [[minimum(reduce(hcat, traces[k] for traces in traces_ensemble), dims=2)[:],
                maximum(reduce(hcat, traces[k] for traces in traces_ensemble), dims=2)[:]] for k in 1:3]
    #std_traces = [std([traces[k] for traces in traces_ensemble]) for k in 1:3]
    #quants = [quantile([traces[k] for traces in traces_ensemble],[0.05,0.95]) for k in 1:3]            
    mean_times = [mean([times[k] for times in times_ensemble]) for k in 1:3]
    names = ["VI", "JAGS", "NUTS + marginalization"]
    
    fig = trace_plot_with_spreads(mean_times, mean_traces, spreads, names, ylabel = "log posterior predictive",
                                  yscale=identity, ymin = mean_traces[3][end] - 10, ymax = mean_traces[3][end] + 5, colors = [:black, :red, :blue])
    save("../figures/log_predictive_spread_$(spec).pdf", fig)
    save("results/case_study_log_predictive_$(spec).jld2", "traces_ensemble", traces_ensemble, 
                                                           "times_ensemble", times_ensemble)

    display(fig)
end

