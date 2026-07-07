function detect_ou_shifts(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real};
    max_shifts::Integer = floor(Int, tree.ntips / 2),
    criterion::Symbol = :mBIC,
    candidate_edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    edge_length_threshold::Float64 = eps(Float64),
    min_descendant_tips::Integer = 1,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    alpha_lower = 0.0
    n_lambda = 81
    proposal_max_iterations = 1000
    proposal_tol = 1e-6
    max_profile_configs = nothing
    optimization = :L_BFGS
    max_iterations = 200
    rel_tol = 1e-8
    vote_threshold = 0.5
    rescale = true
    max_descendant_tips = nothing
    max_candidate_edges = nothing
    candidate_sort = :descendant_length
    intercept_mode = :phylogenetic_intercept
    max_alpha_configs = 32
    forward_refinement = false
    max_forward_refinement_passes = 5
    max_forward_refinement_edges_per_pass = 40
    min_forward_refinement_score_improvement = 1e-6
    path_anchor = false

    tr_mat = _validate_multivariate_trait_shift(tree, trait)
    proposal_mat = rescale ? _l1ou_rescale_matrix(tr_mat) : tr_mat
    _validate_ultrametric_tree(tree)
    cache = build_shift_tree_cache(tree)
    missing_context = _build_mv_shift_missing_context(tree, cache, tr_mat)
    candidate_edges0 = candidate_edges === nothing ? l1ou_default_candidate_edges(cache) : Int.(candidate_edges)
    n_candidates_raw = length(unique(Int.(candidate_edges0)))
    candidates_vec = filter_candidate_edges(
        cache;
        candidate_edges = candidate_edges0,
        edge_length_threshold = edge_length_threshold,
        min_descendant_tips = min_descendant_tips,
        max_descendant_tips = max_descendant_tips,
        max_candidate_edges = max_candidate_edges,
        candidate_sort = candidate_sort,
        exclude_root_edges = false,
    )
    sort_edges_l1ou!(cache, candidates_vec)

    if isempty(candidates_vec)
        return _empty_shift_detection_result(
            tree;
            ntraits = size(tr_mat, 2),
            criterion = criterion,
            message = "empty candidate set",
            diagnostics = (
                intercept_mode = intercept_mode,
                root_model = root_model,
                n_candidates_raw = n_candidates_raw,
                n_candidates = length(candidates_vec),
            ),
        )
    end

    m = size(tr_mat, 2)
    alpha0_vec = zeros(m)
    proposal_column_cache =
        missing_context.has_missing ? nothing :
        [Matrix{Float64}(undef, tree.ntips - 1, length(candidates_vec)) for _ in 1:m]
    proposal_round1_time = @elapsed configs1 = _run_multivariate_path_round(
        tree, cache, proposal_mat, candidates_vec, alpha0_vec;
        round_label = :multivariate_path_round1,
        max_shifts = max_shifts,
        n_lambda = n_lambda,
        proposal_max_iterations = proposal_max_iterations,
        proposal_tol = proposal_tol,
        vote_threshold = vote_threshold,
        intercept_mode = intercept_mode,
        missing_context = missing_context,
        column_cache = proposal_column_cache,
        root_model = root_model,
    )
    shared_refit_cache = Dict{Tuple, NamedTuple}()
    shared_warm_start_cache = Dict{Int, NamedTuple}()
    shared_profile_workspace_caches = _mv_profile_workspace_caches(m)
    raw_configs1 = _mv_collect_raw_path_configs(cache, configs1, OUShiftConfiguration[])
    round1_search = _mv_run_l1ou_style_search(
        raw_configs1,
        tree,
        cache,
        tr_mat;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        max_shifts = max_shifts,
        missing_context = missing_context,
        refit_cache = shared_refit_cache,
        warm_start_cache = shared_warm_start_cache,
        profile_workspace_caches = shared_profile_workspace_caches,
        root_model = root_model,
        fill_best = true,
    )
    alpha_estimate_time = round1_search.diagnostics.exact_scoring.initial_scoring +
        round1_search.diagnostics.exact_scoring.sort_filter +
        round1_search.diagnostics.edge_elimination.edge_elimination +
        round1_search.diagnostics.final_fill
    configs1 = round1_search.diagnostics.profile
    alpha_hat =
        isempty(configs1) || !isfinite(configs1[1].score) ?
        zeros(m) :
        Float64[isfinite(a) && a > alpha_lower ? a : alpha_lower for a in configs1[1].alpha]

    if any(a -> isfinite(a) && a > 0.0, alpha_hat)
        alpha2 = Float64[isfinite(a) && a > 0.0 ? a : 0.0 for a in alpha_hat]
        proposal_round2_time = @elapsed configs2 = _run_multivariate_path_round(
            tree, cache, proposal_mat, candidates_vec, alpha2;
            round_label = :multivariate_path_round2,
            max_shifts = max_shifts,
            n_lambda = n_lambda,
            proposal_max_iterations = proposal_max_iterations,
            proposal_tol = proposal_tol,
            vote_threshold = vote_threshold,
            intercept_mode = intercept_mode,
            missing_context = missing_context,
            column_cache = proposal_column_cache,
            root_model = root_model,
        )
    else
        configs2 = OUShiftConfiguration[]
        proposal_round2_time = 0.0
    end

    round2_scoring_time = 0.0
    raw_configs2 = _mv_collect_raw_path_configs(cache, OUShiftConfiguration[], configs2)
    round2_search = _mv_run_l1ou_style_search(
        raw_configs2,
        tree,
        cache,
        tr_mat;
        max_shifts = max_shifts,
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        initial_alpha = alpha_hat,
        missing_context = missing_context,
        refit_cache = shared_refit_cache,
        warm_start_cache = shared_warm_start_cache,
        profile_workspace_caches = shared_profile_workspace_caches,
        root_model = root_model,
        fill_best = true,
    )

    all_configs = OUShiftConfiguration[]
    append!(all_configs, round1_search.diagnostics.profile)
    append!(all_configs, round2_search.diagnostics.profile)
    _deduplicate_configs!(all_configs)
    _sort_scorable_configs!(all_configs)
    n_configs_round1 = length(configs1)
    n_configs_round2 = length(configs2)
    n_configs_pretrim = length(round1_search.raw_path_configs) + length(round2_search.raw_path_configs)
    proposal_column_cache = nothing
    length(candidates_vec) * m > 100_000 && GC.gc()
    path_anchor_n = nothing
    forward_refinement_stats = (accepted = 0, tried = 0)
    forward_refinement_time = 0.0
    forward_refinement_max_shifts = path_anchor_n === nothing ? Int(max_shifts) : min(Int(max_shifts), Int(path_anchor_n))
    if forward_refinement && !path_anchor && !isempty(all_configs)
        ranked_refinement_edges = Int[]
        seen_refinement_edges = Set{Int}()
        for alpha_ref in (alpha0_vec, alpha_hat)
            alpha_rank = Float64[isfinite(a) && a > 0.0 ? a : 0.0 for a in alpha_ref]
            trait_scores = _multivariate_screening_trait_scores(
                tree,
                cache,
                proposal_mat,
                candidates_vec,
                alpha_rank;
                missing_context = missing_context,
                intercept_mode = intercept_mode,
            )
            ranked = _rank_multivariate_screening_edges(
                cache,
                candidates_vec,
                trait_scores;
                vote_threshold = vote_threshold,
            )
            for edge in ranked.edges
                edge in seen_refinement_edges && continue
                push!(ranked_refinement_edges, edge)
                push!(seen_refinement_edges, edge)
            end
        end
        forward_refinement_time = @elapsed begin
            forward_refinement_stats = _forward_refine_mv_configs!(
                all_configs,
                tree,
                cache,
                tr_mat,
                ranked_refinement_edges;
                max_shifts = forward_refinement_max_shifts,
                criterion = criterion,
                optimization = optimization,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                max_passes = max_forward_refinement_passes,
                max_edges_per_pass = max_forward_refinement_edges_per_pass,
                min_score_improvement = min_forward_refinement_score_improvement,
                refit_cache = shared_refit_cache,
                warm_start_cache = shared_warm_start_cache,
                missing_context = missing_context,
                root_model = root_model,
            )
        end
    end
    scoring_timings = (
        round1 = (
            round1_search.diagnostics.exact_scoring...,
            edge_elimination = round1_search.diagnostics.edge_elimination.edge_elimination,
            edge_elimination_trials = round1_search.diagnostics.edge_elimination.edge_elimination_trials,
            edge_elimination_removed = round1_search.diagnostics.edge_elimination.edge_elimination_removed,
        ),
        round2 = (
            round2_search.diagnostics.exact_scoring...,
            edge_elimination = round2_search.diagnostics.edge_elimination.edge_elimination,
            edge_elimination_trials = round2_search.diagnostics.edge_elimination.edge_elimination_trials,
            edge_elimination_removed = round2_search.diagnostics.edge_elimination.edge_elimination_removed,
        ),
        round2_scoring = round2_scoring_time,
        forward_refinement = forward_refinement_time,
    )

    if isempty(all_configs)
        return _empty_shift_detection_result(
            tree;
            ntraits = m,
            criterion = criterion,
            message = "no scorable configurations",
            diagnostics = (
                intercept_mode = intercept_mode,
                root_model = root_model,
            ),
        )
    end

    best = all_configs[1]
    _fill_mv_config_from_edges!(best, tree, cache, tr_mat, best.shift_edges;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        refit_cache = Dict{Tuple, NamedTuple}(),
        warm_start_cache = Dict{Int, NamedTuple}(),
        missing_context = missing_context,
        root_model = root_model,
    ) || return _empty_shift_detection_result(
        tree;
        ntraits = m,
        criterion = criterion,
        message = "best configuration final refit failed",
        diagnostics = (
            intercept_mode = intercept_mode,
        ),
    )
    return _best_shift_detection_result(
        tree,
        best,
        all_configs;
        ntraits = m,
        criterion = criterion,
        diagnostics = (
            alpha_hat = alpha_hat,
            n_candidates = length(candidates_vec),
            n_candidates_raw = n_candidates_raw,
            candidate_filters = (
                edge_length_threshold = edge_length_threshold,
                min_descendant_tips = min_descendant_tips,
                max_descendant_tips = max_descendant_tips,
                max_candidate_edges = max_candidate_edges,
                candidate_sort = candidate_sort,
            ),
            max_profile_configs = max_profile_configs,
            max_alpha_configs = max_alpha_configs,
            proposal_method = :multivariate_path,
            proposal_max_iterations = proposal_max_iterations,
            proposal_tol = proposal_tol,
            path_anchor = false,
            path_anchor_source = :none,
            path_anchor_n_shifts = path_anchor_n,
            n_path_configs_round1 = length(round1_search.raw_path_configs),
            n_path_configs_round2 = length(round2_search.raw_path_configs),
            n_anchor_configs_round1 = 0,
            n_anchor_configs_round2 = 0,
            n_configs_round1 = n_configs_round1,
            n_configs_round2 = n_configs_round2,
            n_configs_pretrim = n_configs_pretrim,
            n_configs_scored = length(all_configs),
            path_search = (
                round1 = (
                    raw_path_configs = round1_search.diagnostics.raw_path_configs,
                    raw_path_configs_deduped = round1_search.diagnostics.raw_path_configs_deduped,
                    compressed_path_configs = round1_search.diagnostics.compressed_path_configs,
                    edge_eliminated_configs = round1_search.diagnostics.edge_eliminated_configs,
                    stability = round1_search.diagnostics.stability,
                ),
                round2 = (
                    raw_path_configs = round2_search.diagnostics.raw_path_configs,
                    raw_path_configs_deduped = round2_search.diagnostics.raw_path_configs_deduped,
                    compressed_path_configs = round2_search.diagnostics.compressed_path_configs,
                    edge_eliminated_configs = round2_search.diagnostics.edge_eliminated_configs,
                    stability = round2_search.diagnostics.stability,
                ),
            ),
            timings = (
                proposal_round1 = proposal_round1_time,
                alpha_estimate = alpha_estimate_time,
                proposal_round2 = proposal_round2_time,
                full_scoring_detail = scoring_timings,
                path_search_fill = (
                    round1 = round1_search.diagnostics.final_fill,
                    round2 = round2_search.diagnostics.final_fill,
                ),
            ),
            intercept_mode = intercept_mode,
            root_model = root_model,
            rescale = rescale,
            missing_pattern = missing_context.has_missing ? :multivariate_partial : :none,
            observed_counts = copy(missing_context.observed_counts),
            criterion = criterion,
            forward_refinement = forward_refinement,
            forward_refinement_max_shifts = forward_refinement_max_shifts,
            max_forward_refinement_passes = max_forward_refinement_passes,
            max_forward_refinement_edges_per_pass = max_forward_refinement_edges_per_pass,
            min_forward_refinement_score_improvement = min_forward_refinement_score_improvement,
            forward_refinement_stats = forward_refinement_stats,
            source_tree = tree,
            source_trait = tr_mat,
            warnings = (),
        ),
    )
end

