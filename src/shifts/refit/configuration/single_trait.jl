function _refit_ou_shift_config(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    shift_edges::AbstractVector{<:Integer};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    merge_map::Union{Nothing, Dict{Int,Int}} = nothing,
    start_alpha::Union{Nothing, Real} = nothing,
    start_sigma2::Union{Nothing, Real} = nothing,
    start_theta_regimes::Union{Nothing, AbstractVector{<:Real}} = nothing,
    profile_workspace::Union{Nothing, _ShiftCrossproductWorkspace} = nothing,
    profile_alpha_floor::Real = 1e-4,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    tr = _validate_univariate_trait(tree, trait)
    has_merge = merge_map !== nothing && !isempty(merge_map)
    expected_k = length(shift_edges) + 2
    cache =
        if has_merge
            edge_segments = _shift_edge_segments_with_merge(tree, shift_edges, merge_map)
            _prepare_oum_edge_cache(tree, edge_segments)
        elseif profile_workspace !== nothing && _compatible_shift_workspace(profile_workspace, tree, expected_k)
            _shift_oum_cache_from_edges!(profile_workspace, tree, shift_edges)
        else
            _shift_oum_cache_from_edges(tree, shift_edges)
        end
    nregimes = cache.nregimes

    profiled = _profile_refit_ou_shift_config(
        tree,
        tr,
        cache;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        start_alpha = start_alpha,
        cross_workspace = profile_workspace,
        alpha_floor = profile_alpha_floor,
        root_model = root_model,
    )
    if profiled.fit.success
        return (
            success = true,
            loglik = profiled.fit.loglik,
            alpha = profiled.fit.alpha,
            sigma2 = profiled.fit.sigma2,
            theta = copy(profiled.fit.theta),
            nregimes = nregimes,
            n_shifts = nregimes - 1,
            nparams = nregimes + 2,
        )
    end

    spec = ou_spec(:OUM)
    alpha0 = start_alpha === nothing ? max(log(2.0) / max(maximum(tree.dist_from_root[tree.tip_ids]) / 4.0, 1e-8), 1e-8) : Float64(start_alpha)
    sigma0 = start_sigma2 === nothing ? max(var(tr), 1e-8) : Float64(start_sigma2)
    theta0 =
        start_theta_regimes === nothing || length(start_theta_regimes) != nregimes ?
        fill(mean(tr), nregimes) :
        Float64.(start_theta_regimes)

    fit = _ou_fit_with_starts(
        tree,
        trait,
        spec;
        cache = cache,
        method = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        start_alpha = alpha0,
        start_sigma2 = sigma0,
        start_theta_regimes = theta0,
    )

    return (
        success = fit.profile.success,
        loglik = fit.profile.loglik,
        alpha = fit.bundle.alpha[1],
        sigma2 = fit.bundle.sigma2[1],
        theta = copy(fit.bundle.theta),
        nregimes = nregimes,
        n_shifts = nregimes - 1,
        nparams = nregimes + 2,
    )
end

