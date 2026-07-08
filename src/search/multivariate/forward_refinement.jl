function _forward_refine_mv_configs!(
    configs::Vector{OUShiftConfiguration},
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    ranked_edges::AbstractVector{<:Integer};
    max_shifts::Integer,
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    max_passes::Integer = 5,
    max_edges_per_pass::Integer = 40,
    min_score_improvement::Float64 = 1e-6,
    refit_cache::Dict{Tuple, NamedTuple} = Dict{Tuple, NamedTuple}(),
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}} = nothing,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    root_model::Symbol = :OUfixedRoot,
)
    isempty(configs) && return (accepted = 0, tried = 0)
    root_model = _normalize_ou_root_model(root_model)
    _sort_scorable_configs!(configs)
    isempty(configs) && return (accepted = 0, tried = 0)

    current = copy(configs[1].shift_edges)
    current_set = Set(current)
    best_score = configs[1].score
    warm_cache = warm_start_cache === nothing ? Dict{Int, NamedTuple}() : warm_start_cache
    ctx = _mv_exact_scoring_context(
        tree,
        cache,
        trait_mat;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        missing_context = missing_context,
        refit_cache = refit_cache,
        warm_start_cache = warm_cache,
        root_model = root_model,
    )
    accepted = 0
    tried_total = 0

    for _ in 1:max(Int(max_passes), 0)
        best_trial = nothing
        tried_pass = 0
        for edge0 in ranked_edges
            edge = Int(edge0)
            edge in current_set && continue
            raw = vcat(current, edge)
            corrected = _order_shift_edges_like(correct_shift_configuration_l1ou(cache, raw), raw)
            isempty(corrected) && continue
            length(corrected) > max_shifts && continue
            trial_set = Set(corrected)
            trial_set == current_set && continue

            cfg = OUShiftConfiguration(
                shift_edges = corrected,
                n_shifts = length(corrected),
                source = :forward_refinement,
            )
            ok = _score_mv_config_from_edges!(cfg, ctx, corrected; store_details = false)
            tried_pass += 1
            tried_total += 1
            if ok &&
               cfg.score < best_score - min_score_improvement &&
               (best_trial === nothing || cfg.score < best_trial.score)
                best_trial = cfg
            end
            tried_pass >= max_edges_per_pass && break
        end

        best_trial === nothing && break
        push!(configs, best_trial)
        current = copy(best_trial.shift_edges)
        current_set = Set(current)
        best_score = best_trial.score
        accepted += 1
    end

    _sort_scorable_configs!(configs)
    return (accepted = accepted, tried = tried_total)
end
