function _propose_shift_configs_screening(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha::Float64;
    max_shifts::Integer = typemax(Int),
    n_lambda::Integer = 100,
    lambda_min_ratio::Float64 = 1e-5,
    intercept_mode::Symbol = :phylogenetic_intercept,
)
    ranked = _rank_shift_screening_edges(
        tree,
        cache,
        trait,
        candidates,
        alpha;
        intercept_mode = intercept_mode,
    )
    configs = _build_univariate_screening_candidates(
        cache,
        ranked.edges;
        max_shifts = max_shifts,
    )
    return (
        configs = configs,
        diagnostics = (
            proposal_method = :tree_pruning_screening,
            proposal_family = :tree_pruning_screening,
            n_contrasts = tree.ntips - 1,
            n_candidates = length(candidates),
            n_ranked_edges = length(ranked.edges),
            n_configs = length(configs),
            max_shifts = Int(max_shifts),
            n_lambda = Int(n_lambda),
            lambda_min_ratio = Float64(lambda_min_ratio),
        ),
    )
end

function _propose_shift_configs_screening_prefix_anchor(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha::Float64;
    max_shifts::Integer = typemax(Int),
    max_prefix_edges::Integer = typemax(Int),
    min_standardized_score::Union{Nothing, Float64} = nothing,
    intercept_mode::Symbol = :phylogenetic_intercept,
)
    ranked = _rank_shift_screening_edges(
        tree,
        cache,
        trait,
        candidates,
        alpha;
        intercept_mode = intercept_mode,
    )
    prefix_limit = Int(max_prefix_edges)
    if min_standardized_score !== nothing
        threshold = Float64(min_standardized_score)
        prefix_limit = 0
        @inbounds for z in ranked.standardized_scores
            abs(z) >= threshold || break
            prefix_limit += 1
        end
        prefix_limit = min(prefix_limit, Int(max_prefix_edges))
    end
    configs = _build_univariate_screening_prefix_anchor_candidates(
        cache,
        ranked.edges;
        max_shifts = max_shifts,
        max_prefix_edges = prefix_limit,
    )
    return (
        configs = configs,
        diagnostics = (
            proposal_method = :tree_pruning_screening_prefix_anchor,
            proposal_family = :tree_pruning_screening,
            n_contrasts = tree.ntips - 1,
            n_candidates = length(candidates),
            n_ranked_edges = length(ranked.edges),
            n_configs = length(configs),
            max_shifts = Int(max_shifts),
            max_prefix_edges = Int(max_prefix_edges),
            prefix_edges_used = prefix_limit,
            min_standardized_score = min_standardized_score,
        ),
    )
end

