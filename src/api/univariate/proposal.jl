function _run_screening_round(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha::Float64;
    round_label::Symbol = :screening_alpha0,
    max_shifts::Integer = typemax(Int),
    n_lambda::Integer = 100,
    lambda_min_ratio::Float64 = 1e-5,
    intercept_mode::Symbol = :phylogenetic_intercept,
)
    tr = Float64.(trait)

    proposal = _propose_shift_configs_screening(
        tree, cache, tr, candidates, alpha;
        max_shifts = max_shifts,
        n_lambda = n_lambda, lambda_min_ratio = lambda_min_ratio,
        intercept_mode = intercept_mode,
    )

    configs = OUShiftConfiguration[]
    for edges in proposal.configs
        push!(configs, OUShiftConfiguration(
            shift_edges = edges,
            n_shifts = length(edges),
            source = round_label,
        ))
    end
    return configs
end

function _run_screening_prefix_anchor_round(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha::Float64;
    round_label::Symbol = :screening_prefix_anchor_alpha0,
    max_shifts::Integer = typemax(Int),
    max_prefix_edges::Integer = typemax(Int),
    min_standardized_score::Union{Nothing, Float64} = nothing,
    intercept_mode::Symbol = :phylogenetic_intercept,
)
    tr = Float64.(trait)

    proposal = _propose_shift_configs_screening_prefix_anchor(
        tree, cache, tr, candidates, alpha;
        max_shifts = max_shifts,
        max_prefix_edges = max_prefix_edges,
        min_standardized_score = min_standardized_score,
        intercept_mode = intercept_mode,
    )

    configs = OUShiftConfiguration[]
    for edges in proposal.configs
        push!(configs, OUShiftConfiguration(
            shift_edges = edges,
            n_shifts = length(edges),
            source = round_label,
        ))
    end
    return configs
end

function _estimate_alpha_from_configs(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    configs::Vector{OUShiftConfiguration};
    alpha_lower::Float64 = 0.0,
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    refit_cache::Dict{Tuple, NamedTuple} = Dict{Tuple, NamedTuple}(),
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    best_alpha = 0.0
    best_ll = -Inf
    n = tree.ntips
    for cfg in configs
        cfg.n_shifts == 0 && continue
        refit = _cached_score_refit_univariate(refit_cache, tree, cache, trait, cfg.shift_edges, n;
            criterion = criterion,
            optimization = optimization,
            max_iterations = max_iterations,
            rel_tol = rel_tol,
            root_model = root_model,
        )
        if refit.success && refit.loglik > best_ll
            best_ll = refit.loglik
            best_alpha = max(refit.alpha, alpha_lower)
        end
    end
    if best_ll == -Inf
        for cfg in configs
            refit = _cached_score_refit_univariate(refit_cache, tree, cache, trait, cfg.shift_edges, n;
                criterion = criterion,
                optimization = optimization,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                root_model = root_model,
            )
            if refit.success
                return max(refit.alpha, alpha_lower)
            end
        end
    end
    return best_alpha
end

