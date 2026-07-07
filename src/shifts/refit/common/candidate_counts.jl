function _candidate_count_for_ic(
    tree::CompactTree;
    candidate_edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    edge_length_threshold::Float64 = eps(Float64),
    min_descendant_tips::Integer = 1,
    max_descendant_tips::Union{Nothing, Integer} = nothing,
    max_candidate_edges::Union{Nothing, Integer} = nothing,
    candidate_sort::Symbol = :descendant_length,
)
    cache = build_shift_tree_cache(tree)
    candidates = filter_candidate_edges(
        cache;
        candidate_edges = candidate_edges,
        edge_length_threshold = edge_length_threshold,
        min_descendant_tips = min_descendant_tips,
        max_descendant_tips = max_descendant_tips,
        max_candidate_edges = max_candidate_edges,
        candidate_sort = candidate_sort,
    )
    return length(candidates), cache
end


