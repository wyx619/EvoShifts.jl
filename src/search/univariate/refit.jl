function _score_refit_univariate(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    shift_edges::AbstractVector{<:Integer},
    n::Integer,
;
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    if isempty(shift_edges)
        fit = _fit_ou1_for_shift_detection(tree, trait;
            optimization = optimization,
            max_iterations = max_iterations,
            rel_tol = rel_tol,
            root_model = root_model,
        )
        return (
            success = fit.profile.success,
            loglik = fit.profile.loglik,
            alpha = fit.bundle.alpha[1],
            sigma2 = fit.bundle.sigma2[1],
            theta = copy(fit.bundle.theta),
            shift_values = Float64[],
            shift_means = Float64[],
            n_shifts = 0,
            score = _score_configuration_full(cache, fit.profile.loglik, 0, Int[], n; criterion = criterion),
        )
    end

    refit = _refit_ou_shift_config(tree, trait, shift_edges;
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        root_model = root_model,
    )
    if !refit.success
        return (success = false, loglik = NaN, alpha = NaN, sigma2 = NaN, theta = Float64[], shift_values = Float64[], shift_means = Float64[], n_shifts = length(shift_edges), score = Inf)
    end
    edge_segments = shift_edges_to_edge_segments(tree, shift_edges)
    shift_values = _shift_values_from_theta(tree, edge_segments, shift_edges, refit.theta)
    score = _score_configuration_full(
        cache, refit.loglik, refit.n_shifts, shift_edges, n;
        criterion = criterion,
    )
    return (
        success = true,
        loglik = refit.loglik,
        alpha = refit.alpha,
        sigma2 = refit.sigma2,
        theta = refit.theta,
        shift_values = shift_values,
        shift_means = _shift_means_from_shift_values(tree, shift_edges, shift_values, refit.alpha),
        n_shifts = refit.n_shifts,
        score = score,
    )
end

function _cached_score_refit_univariate(
    refit_cache::Dict{Tuple, NamedTuple},
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    shift_edges::AbstractVector{<:Integer},
    n::Integer,
;
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    key = _shift_edges_key(shift_edges)
    if haskey(refit_cache, key)
        return refit_cache[key]
    end
    refit = _score_refit_univariate(tree, cache, trait, shift_edges, n;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        root_model = root_model,
    )
    refit_cache[key] = refit
    return refit
end

function _prune_shift_config(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    shift_edges::AbstractVector{<:Integer};
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    max_edge_elimination_passes::Integer = 1,
    refit_cache::Dict{Tuple, NamedTuple} = Dict{Tuple, NamedTuple}(),
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    n = tree.ntips
    current = correct_shift_configuration(cache, shift_edges)
    score_fn = edges -> _cached_score_refit_univariate(refit_cache, tree, cache, trait, edges, n;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        root_model = root_model,
    )
    pruned = _prune_shift_edges_by_score(
        current,
        score_fn;
        max_edge_elimination_passes = max_edge_elimination_passes,
        min_prunable_shifts = 3,
    )
    return (
        shift_edges = pruned.shift_edges,
        removed_edges = pruned.removed_edges,
        refit = pruned.score,
    )
end

