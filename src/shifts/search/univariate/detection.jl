function _score_and_sort_configs!(
    configs::Vector{OUShiftConfiguration},
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real};
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    edge_elimination::Bool = true,
    max_edge_elimination_passes::Integer = 1,
    refit_cache::Dict{Tuple, NamedTuple} = Dict{Tuple, NamedTuple}(),
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    n = tree.ntips
    for cfg in configs
        corrected = correct_shift_configuration(cache, cfg.shift_edges)
        if edge_elimination
            bc = _prune_shift_config(tree, cache, trait, corrected;
                criterion = criterion,
                optimization = optimization,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                max_edge_elimination_passes = max_edge_elimination_passes,
                refit_cache = refit_cache,
                root_model = root_model,
            )
            corrected = bc.shift_edges
            refit = bc.refit
        else
            refit = _cached_score_refit_univariate(refit_cache, tree, cache, trait, corrected, n;
                criterion = criterion,
                optimization = optimization,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                root_model = root_model,
            )
        end
        if refit.success
            cfg.shift_edges = corrected
            cfg.loglik = [refit.loglik]
            cfg.alpha = [refit.alpha]
            cfg.sigma2 = [refit.sigma2]
            edge_segments = shift_edges_to_edge_segments(tree, corrected)
            theta = copy(refit.theta)
            fitted_means = _ou_shift_fitted_means(tree, edge_segments, theta, refit.alpha)
            cfg.theta = theta
            cfg.shift_values = refit.shift_values
            cfg.shift_means = refit.shift_means
            cfg.fitted_means = fitted_means
            cfg.residuals = Float64.(trait) .- fitted_means
            cfg.edge_optima = _edge_optima_from_theta(edge_segments, theta)
            cfg.n_shifts = refit.n_shifts
            cfg.score = refit.score
            cfg.criterion = criterion
        end
    end
    return _sort_scorable_configs!(configs)
end

