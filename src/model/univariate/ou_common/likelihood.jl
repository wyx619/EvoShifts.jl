function _ou_root_prior(spec::OUSpec, bundle::OUParameterBundle; cache = nothing)
    mean =
        if spec.root_mean_mode === :theta
            bundle.theta[1]
        elseif spec.root_mean_mode === :root_regime_theta
            cache === nothing && throw(ArgumentError("cache is required for root_regime_theta"))
            bundle.theta[cache.root_regime]
        elseif spec.root_mean_mode === :free_theta0
            bundle.theta0 === nothing && throw(ArgumentError("theta0 is required for free_theta0"))
            bundle.theta0
        else
            throw(ArgumentError("Unsupported OU root_mean_mode=$(spec.root_mean_mode)"))
        end

    if spec.root_cov_mode === :stationary
        root_state = cache === nothing ? 1 : cache.root_regime
        alpha = _ou_regime_value(spec.alpha_mode, bundle.alpha, root_state)
        sigma2 = _ou_regime_value(spec.sigma_mode, bundle.sigma2, root_state)
        return (mean = mean, var = sigma2 / (2.0 * alpha), profile_root = false)
    elseif spec.root_cov_mode === :fixed
        return (mean = mean, var = 0.0, profile_root = false)
    elseif spec.root_cov_mode === :nonstationary
        return (mean = mean, var = Inf, profile_root = true)
    elseif spec.root_cov_mode === :free
        return (mean = mean, var = Inf, profile_root = true)
    end
    throw(ArgumentError("Unsupported OU root_cov_mode=$(spec.root_cov_mode)"))
end

function _ou_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec,
    bundle::OUParameterBundle;
    cache = nothing,
)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    edges = _build_ou_edges(tree, spec, bundle; cache = cache)
    root = _ou_root_prior(spec, bundle; cache = cache)
    prof = _linear_gaussian_loglik(
        tree,
        trait,
        edges.edge_a,
        edges.edge_b,
        edges.edge_v;
        root_prior_mean = root.mean,
        root_prior_var = root.var,
        profile_root = root.profile_root,
    )
    return (
        success = prof.success,
        loglik = prof.loglik,
        root_state = prof.root_state,
        edges = edges,
        root = root,
    )
end
