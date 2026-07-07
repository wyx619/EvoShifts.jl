function _ou_fit(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec;
    cache = nothing,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-6,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)

    nregimes = cache === nothing ? 1 : cache.nregimes
    init = _ou_initial_params(spec, nregimes; tree = tree, trait = tr)
    objective = function (par)
        bundle = _ou_unpack_params(spec, par, nregimes)
        prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
        return prof.success ? -prof.loglik : Inf
    end
    lower_bounds = vcat(
        fill(1e-8, spec.alpha_mode === :shared ? 1 : nregimes),
        fill(1e-8, spec.sigma_mode === :shared ? 1 : nregimes),
        fill(-Inf, spec.theta_mode === :shared ? 1 : nregimes),
        spec.root_mean_mode === :free_theta0 ? [-Inf] : Float64[],
    )
    result = _ou_optimize_from_initial(
        objective,
        init,
        spec,
        nregimes,
        tree,
        tr,
        cache;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = lower_bounds,
    )

    minimizer = _continuous_result_minimizer(result)
    bundle = _ou_unpack_params(spec, minimizer, nregimes)
    prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
    return (bundle = bundle, profile = prof, result = result, nregimes = nregimes)
end

function _ou_fit_with_starts(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec;
    cache = nothing,
    method::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    start_alpha::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_sigma2::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_theta::Union{Nothing, Real} = nothing,
    start_theta_regimes::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)

    nregimes = cache === nothing ? 1 : cache.nregimes
    init = _ou_initial_params_from_starts(
        spec,
        nregimes;
        tree = tree,
        trait = tr,
        start_alpha = start_alpha,
        start_sigma2 = start_sigma2,
        start_theta = start_theta,
        start_theta_regimes = start_theta_regimes,
    )
    objective = function (par)
        bundle = _ou_unpack_params(spec, par, nregimes)
        prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
        return prof.success ? -prof.loglik : Inf
    end
    lower_bounds = vcat(
        fill(1e-8, spec.alpha_mode === :shared ? 1 : nregimes),
        fill(1e-8, spec.sigma_mode === :shared ? 1 : nregimes),
        fill(-Inf, spec.theta_mode === :shared ? 1 : nregimes),
        spec.root_mean_mode === :free_theta0 ? [-Inf] : Float64[],
    )
    result =
        if method in (:SBPLX_L_BFGS, :LN_SBPLX_L_BFGS, :TWO_STAGE)
            _ou_optimize_from_initial(
                objective,
                init,
                spec,
                nregimes,
                tree,
                tr,
                cache;
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        elseif method === :LN_SBPLX
            _continuous_optimize_objective(
                objective,
                init;
                method = :LN_SBPLX,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        else
            method === :L_BFGS || throw(ArgumentError("Unsupported internal method=$method"))
            _continuous_optimize_objective(
                objective,
                init;
                method = :L_BFGS,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        end

    minimizer = _continuous_result_minimizer(result)
    bundle = _ou_unpack_params(spec, minimizer, nregimes)
    prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
    return (bundle = bundle, profile = prof, result = result, nregimes = nregimes)
end
