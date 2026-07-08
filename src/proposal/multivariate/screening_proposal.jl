function _propose_shift_configs_multivariate_screening(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha_vec::AbstractVector{<:Real};
    max_shifts::Integer = typemax(Int),
    n_lambda::Integer = 100,
    max_iterations::Integer = 1000,
    tol::Float64 = 1e-6,
    intercept_mode::Symbol = :phylogenetic_intercept,
    vote_threshold::Float64 = 0.0,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    column_cache::Union{Nothing, Vector{Matrix{Float64}}} = nothing,
    keep_path::Bool = false,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    trait_scores = _multivariate_screening_trait_scores(
        tree,
        cache,
        trait_mat,
        candidates,
        alpha_vec;
        missing_context = missing_context,
        intercept_mode = intercept_mode,
    )
    ranked = _rank_multivariate_screening_edges(
        cache,
        candidates,
        trait_scores;
        vote_threshold = vote_threshold,
    )
    configs = _build_multivariate_screening_candidates(
        cache,
        ranked.edges;
        max_shifts = max_shifts,
    )
    return (
        configs = configs,
        diagnostics = (
            proposal_method = :tree_pruning_multivariate_screening,
            n_candidates = length(candidates),
            n_ranked_edges = length(ranked.edges),
            n_configs = length(configs),
            max_shifts = Int(max_shifts),
            vote_threshold = vote_threshold,
            has_missing = missing_context !== nothing && missing_context.has_missing,
            visible_counts = ranked.visible_counts,
            shared_scores = ranked.shared_scores,
            n_lambda = Int(n_lambda),
            max_iterations = Int(max_iterations),
            tol = Float64(tol),
            keep_path = keep_path,
            root_model = root_model,
            used_column_cache = column_cache !== nothing,
        ),
    )
end

function _propose_shift_configs_multivariate_screening_prefix_anchor(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha_vec::AbstractVector{<:Real};
    max_shifts::Integer = typemax(Int),
    max_prefix_edges::Integer = typemax(Int),
    min_shared_score::Union{Nothing, Float64} = nothing,
    intercept_mode::Symbol = :phylogenetic_intercept,
    vote_threshold::Float64 = 0.0,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    trait_scores = _multivariate_screening_trait_scores(
        tree,
        cache,
        trait_mat,
        candidates,
        alpha_vec;
        missing_context = missing_context,
        intercept_mode = intercept_mode,
    )
    ranked = _rank_multivariate_screening_edges(
        cache,
        candidates,
        trait_scores;
        vote_threshold = vote_threshold,
    )
    prefix_limit = Int(max_prefix_edges)
    if min_shared_score !== nothing
        threshold = Float64(min_shared_score)
        prefix_limit = 0
        @inbounds for score in ranked.shared_scores
            score >= threshold || break
            prefix_limit += 1
        end
        prefix_limit = min(prefix_limit, Int(max_prefix_edges))
    end
    configs = _build_multivariate_screening_prefix_anchor_candidates(
        cache,
        ranked.edges;
        max_shifts = max_shifts,
        max_prefix_edges = prefix_limit,
    )
    return (
        configs = configs,
        diagnostics = (
            proposal_method = :tree_pruning_multivariate_screening_prefix_anchor,
            proposal_family = :tree_pruning_multivariate_screening,
            n_candidates = length(candidates),
            n_ranked_edges = length(ranked.edges),
            n_configs = length(configs),
            max_shifts = Int(max_shifts),
            max_prefix_edges = Int(max_prefix_edges),
            prefix_edges_used = prefix_limit,
            min_shared_score = min_shared_score,
            vote_threshold = vote_threshold,
            has_missing = missing_context !== nothing && missing_context.has_missing,
            visible_counts = ranked.visible_counts,
            shared_scores = ranked.shared_scores,
            root_model = root_model,
        ),
    )
end

