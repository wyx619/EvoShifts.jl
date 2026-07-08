function _build_multivariate_screening_candidates(
    cache::OUShiftTreeCache,
    ranked_edges::AbstractVector{<:Integer};
    max_shifts::Integer = typemax(Int),
)
    return _build_screening_candidate_family(
        cache,
        ranked_edges;
        max_shifts = max_shifts,
    )
end

function _build_multivariate_screening_prefix_anchor_candidates(
    cache::OUShiftTreeCache,
    ranked_edges::AbstractVector{<:Integer};
    max_shifts::Integer = typemax(Int),
    max_prefix_edges::Integer = typemax(Int),
)
    return _build_screening_prefix_anchor_family(
        cache,
        ranked_edges;
        max_shifts = max_shifts,
        max_prefix_edges = max_prefix_edges,
    )
end

