struct ELBO{oType}
    om::oType
    n_estim::Int
end
function (elbo::ELBO)(ϕ)
    return elbo_estimator(ϕ; n=elbo.n_estim, y=elbo.om.Y, z_known=elbo.om.z_known, prior=elbo.om.prior, W=elbo.om.W, X=elbo.om.X)
end
function elbo_estimator(ϕ; z_known=z_known, n=1, y=Y, prior=prior, W=W, X=X)
    elbo = 0.0
    for _ in 1:n
        z, αβ = sample_q(ϕ)
        elbo += div(ϕ, z, αβ, y, z_known, prior, W, X)
    end
    return elbo/n
end

struct ELBO_marginal{oType}
    om::oType
    n_estim::Int
end
function (elbo::ELBO_marginal)(ϕ)
    return elbo_estimator_marginal(ϕ; n=elbo.n_estim, y=elbo.om.Y, z_known=elbo.om.z_known, prior=elbo.om.prior, W=elbo.om.W, X=elbo.om.X)
end
function elbo_estimator_marginal(ϕ; n=1, y=Y, z_known=z_known, prior=prior, W=W, X=X)
    elbo = 0.0
    for _ in 1:n
        αβ = sample_q_marginal(ϕ)
        elbo += div_marginal(ϕ, αβ, y, z_known, prior, W, X)
    end
    return elbo/n
end

function predictive_posterior_estimator(ϕ, y, w, x, z_known; n=1, marginal = true)
    log_predictive = 0.0
    if marginal 
        for _ in 1:n
            z, αβ = sample_q(ϕ)
            log_predictive += logmarginal(y, z_known, αβ, w, x) 
        end
    else
        for _ in 1:n
            z, αβ = sample_q(ϕ)
            log_predictive += logconditional(y, z, αβ, w, x, z_known=z_known)
        end
    end
    return log_predictive/n
end
predictive_posterior_estimator(ϕ, om::OccupancyModel; n=1, marginal = true) = predictive_posterior_estimator(ϕ, om.Y, om.W, om.X, om.z_known, n=n, marginal = marginal)

function estimate_gradient(obj, n_estim)
    grad = stochastic_gradient(obj)
    grad_val = grad.p

    for _ in 1:n_estim-1
        grad_val += stochastic_gradient(obj).p
    end
    StochasticModel(grad.X, grad_val/n_estim)
end
function optimize_elbo(om::OccupancyModel, ϕ0, batch_size, iterations, LEARNING_RATE; n_switch = 100, n_estimator = batch_size, n_snapshots = 1, marginal = true)
    elbo_trace = Float64[] 
    log_predictive_trace = Float64[]
    wall_times = Float64[]
    ϕ_trace = Vector{Float64}[]
    
    elbo_opt = ELBO(om, 1) #elbo_estimator(obj.p, n=n_estimator)
    elbo_est = ELBO(om, n_estimator)
    obj = StochasticModel(ϕ -> -elbo_opt(ϕ), ϕ0)
    optimizer = Adam(LEARNING_RATE) #Optimiser(ExpDecay Adam())
    setup = Optimisers.setup(optimizer, obj)

    elbo = elbo_est(obj.p)
    t = 0
    push!(elbo_trace, elbo)
    push!(wall_times, t)
    k = 1
    switched = false
    for i in 1:iterations
        t += @elapsed grad = estimate_gradient(obj, batch_size) #stochastic_gradient(obj)
        if sum(isinf.(grad.p)) == 0
            if i > n_switch #&& !switched 
                #optimizer = Adam(LEARNING_RATE/(i-n_switch)^(1/2 + 1e-8))
                #setup = Optimisers.setup(optimizer, obj)
                Optimisers.adjust!(setup, eta = LEARNING_RATE/(i-n_switch)^(1/2 + 1e-8))
            end
            t += @elapsed Optimisers.update!(setup, obj, grad)
            push!(ϕ_trace, deepcopy(obj.p))
            println("iteration $i")
        end
        if i % n_snapshots == 0
            elbo = elbo_est(obj.p) #elbo_estimator(obj.p, n=n_estimator)
            log_predictive = predictive_posterior_estimator(obj.p, om, n=n_estimator, marginal = marginal)
            push!(log_predictive_trace, log_predictive)
            push!(elbo_trace, elbo)
            push!(wall_times, t)
        end
    end
    elbo = elbo_est(obj.p)
    log_predictive = predictive_posterior_estimator(obj.p, om, n=n_estimator, marginal = marginal)
    push!(log_predictive_trace, log_predictive)
    push!(elbo_trace, elbo)
    push!(wall_times, t)
    return obj.p, ϕ_trace, wall_times, elbo_trace, log_predictive_trace
end

function compute_mean_trace(chain, parnames, true_mean) 
    data = chain[parnames].value.data[:,:,1]
    agg_data = cumsum(data, dims=1)
    means = agg_data ./ (1:size(data,1))
    means .- true_mean'
end

# hardcoded for JAGS output
function compute_log_posterior_trace(chain, names, om, mode = :marginal)
    if mode == :marginal 
        log_predictive_trace =  cumsum([logmarginal(om.Y,om.z_known,ab,om.W,om.X) for ab in eachrow(chain[names].value.data[:,:,1])]) 
    else
        log_predictive_trace = []
        last_entry = 0
        ab_range = chain[names[1:4]].value.data[:,:,1]
        z_range = chain[names[5:end]].value.data[:,:,1]
        for i in 1:size(chain,1)
            ab = ab_range[i,:]
            z = z_range[i,:]
            push!(log_predictive_trace, last_entry + logconditional(om.Y,z,ab, om.W, om.X))         
            last_entry = log_predictive_trace[end]
        end
    end
    log_predictive_trace ./ (1:length(log_predictive_trace))
end