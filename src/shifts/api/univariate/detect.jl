function detect_ou_shifts(
    tree::CompactTree,
    trait::AbstractVector{<:Real};
    max_shifts::Integer = floor(Int, tree.ntips / 2),
    criterion::Symbol = :mBIC,
    candidate_edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    edge_length_threshold::Float64 = eps(Float64),
    min_descendant_tips::Integer = 1,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    alpha_lower = 0.0
    n_lambda = 100
    lambda_min_ratio = 1e-5
    prefix_anchor_min_standardized_score = 2.0
    max_profile_configs = nothing
    optimization = :L_BFGS
    max_iterations = 200
    rel_tol = 1e-8
    max_descendant_tips = nothing
    max_candidate_edges = nothing
    candidate_sort = :descendant_length
    intercept_mode = :phylogenetic_intercept
    edge_elimination = true
    max_edge_elimination_passes = 1
    path_anchor = true

    tr = _validate_univariate_trait(tree, trait)
    _validate_ultrametric_tree(tree)
    cache = build_shift_tree_cache(tree)
    candidate_edges0 = candidate_edges === nothing ? l1ou_default_candidate_edges(cache) : Int.(candidate_edges)
    n_candidates_raw = length(unique(Int.(candidate_edges0)))
    candidates = filter_candidate_edges(
        cache;
        candidate_edges = candidate_edges0,
        edge_length_threshold = edge_length_threshold,
        min_descendant_tips = min_descendant_tips,
        max_descendant_tips = max_descendant_tips,
        max_candidate_edges = max_candidate_edges,
        candidate_sort = candidate_sort,
        exclude_root_edges = false,
    )
    sort_edges_l1ou!(cache, candidates)
    if isempty(candidates)
        return _empty_shift_detection_result(
            tree;
            ntraits = 1,
            criterion = criterion,
            message = "empty candidate set",
            diagnostics = (
                intercept_mode = intercept_mode,
                root_model = root_model,
                n_candidates_raw = n_candidates_raw,
                n_candidates = length(candidates),
            ),
        )
    end

    all_configs = OUShiftConfiguration[]
    refit_cache = Dict{Tuple, NamedTuple}()

    path_configs1 =
        path_anchor ?
        _run_screening_prefix_anchor_round(
            tree, cache, tr, candidates, 0.0;
            round_label = :screening_prefix_anchor_alpha0,
            max_shifts = max_shifts,
            min_standardized_score = prefix_anchor_min_standardized_score,
            intercept_mode = intercept_mode,
        ) :
        OUShiftConfiguration[]

    configs1 = _run_screening_round(
        tree, cache, tr, candidates, 0.0;
        round_label = :screening_alpha0,
        max_shifts = max_shifts,
        n_lambda = n_lambda,
        lambda_min_ratio = lambda_min_ratio,
        intercept_mode = intercept_mode,
    )
    append!(all_configs, path_configs1)
    append!(all_configs, configs1)

    alpha_hat = _estimate_alpha_from_configs(tree, cache, tr, all_configs;
        alpha_lower = alpha_lower,
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        refit_cache = refit_cache,
        root_model = root_model,
    )

    if isfinite(alpha_hat) && alpha_hat > 0.0
        if path_anchor
            path_configs2 = _run_screening_prefix_anchor_round(
                tree, cache, tr, candidates, alpha_hat;
                round_label = :screening_prefix_anchor_alpha_refined,
                max_shifts = max_shifts,
                min_standardized_score = prefix_anchor_min_standardized_score,
                intercept_mode = intercept_mode,
            )
            append!(all_configs, path_configs2)
        else
            path_configs2 = OUShiftConfiguration[]
        end
        configs2 = _run_screening_round(
            tree, cache, tr, candidates, alpha_hat;
            round_label = :screening_alpha_refined,
            max_shifts = max_shifts,
            n_lambda = n_lambda,
            lambda_min_ratio = lambda_min_ratio,
            intercept_mode = intercept_mode,
        )
        for cfg in configs2
            push!(all_configs, cfg)
        end
    else
        path_configs2 = OUShiftConfiguration[]
        configs2 = OUShiftConfiguration[]
    end

    if isempty(all_configs)
        return _empty_shift_detection_result(
            tree;
            ntraits = 1,
            criterion = criterion,
            message = "no configurations found by screening proposal",
            diagnostics = (
                intercept_mode = intercept_mode,
                root_model = root_model,
                n_candidates = length(candidates),
                n_candidates_raw = n_candidates_raw,
                n_configs_round1 = length(path_configs1) + length(configs1),
                n_configs_round2 = length(path_configs2) + length(configs2),
            ),
        )
    end

    _deduplicate_configs!(all_configs)
    n_configs_pretrim = length(all_configs)
    if max_profile_configs !== nothing && n_configs_pretrim > max_profile_configs
        _limit_profile_configs_diverse!(all_configs, Int(max_profile_configs))
    end
    _score_and_sort_configs!(all_configs, tree, cache, tr;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        edge_elimination = edge_elimination,
        max_edge_elimination_passes = max_edge_elimination_passes,
        refit_cache = refit_cache,
        root_model = root_model,
    )
    path_anchor_n = path_anchor ? _apply_path_anchor_selection!(all_configs) : nothing

    if isempty(all_configs)
        return _empty_shift_detection_result(
            tree;
            ntraits = 1,
            criterion = criterion,
            message = "no scorable configurations",
            diagnostics = (
                intercept_mode = intercept_mode,
                root_model = root_model,
            ),
        )
    end

    best = all_configs[1]
    return _best_shift_detection_result(
        tree,
        best,
        all_configs;
        ntraits = 1,
        criterion = criterion,
        diagnostics = (
            alpha_hat = alpha_hat,
            n_candidates = length(candidates),
            n_candidates_raw = n_candidates_raw,
            candidate_filters = (
                edge_length_threshold = edge_length_threshold,
                min_descendant_tips = min_descendant_tips,
                max_descendant_tips = max_descendant_tips,
                max_candidate_edges = max_candidate_edges,
                candidate_sort = candidate_sort,
            ),
            max_profile_configs = max_profile_configs,
            proposal_method = path_anchor ? :tree_pruning_screening_with_prefix_z_anchor : :tree_pruning_screening,
            n_lambda = n_lambda,
            lambda_min_ratio = lambda_min_ratio,
            path_anchor = path_anchor,
            path_anchor_source = path_anchor ? :tree_pruning_prefix_z : :none,
            prefix_anchor_min_standardized_score = prefix_anchor_min_standardized_score,
            path_anchor_n_shifts = path_anchor_n,
            n_path_configs_round1 = length(path_configs1),
            n_path_configs_round2 = length(path_configs2),
            n_screening_configs_round1 = length(configs1),
            n_screening_configs_round2 = length(configs2),
            n_configs_round1 = length(path_configs1) + length(configs1),
            n_configs_round2 = length(path_configs2) + length(configs2),
            n_configs_pretrim = n_configs_pretrim,
            n_configs_scored = length(all_configs),
            intercept_mode = intercept_mode,
            root_model = root_model,
            criterion = criterion,
            edge_elimination = edge_elimination,
            max_edge_elimination_passes = max_edge_elimination_passes,
            source_tree = tree,
            source_trait = tr,
            warnings = (),
        ),
    )
end

