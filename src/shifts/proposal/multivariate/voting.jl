function _rank_multivariate_screening_edges(
    cache::OUShiftTreeCache,
    candidates::AbstractVector{<:Integer},
    trait_scores::NamedTuple;
    vote_threshold::Float64 = 0.0,
)
    standardized = trait_scores.standardized_scores
    visible = trait_scores.visible
    p, m = size(standardized)
    length(candidates) == p || throw(ArgumentError("candidate length does not match screening scores"))
    min_visible =
        vote_threshold <= 0.0 ?
        1 :
        max(1, ceil(Int, vote_threshold * m))

    shared_scores = zeros(Float64, p)
    visible_counts = zeros(Int, p)
    for j in 1:p
        s2 = 0.0
        nvis = 0
        @inbounds for i in 1:m
            if visible[j, i]
                z = standardized[j, i]
                s2 += z * z
                nvis += 1
            end
        end
        visible_counts[j] = nvis
        if nvis >= min_visible
            shared_scores[j] = sqrt(s2 / nvis) * sqrt(nvis)
        end
    end

    order = sortperm(
        1:p;
        by = j -> (-shared_scores[j], -visible_counts[j], Int(candidates[j])),
    )
    filter!(j -> shared_scores[j] > 0.0 && Int(cache.edge_parent[Int(candidates[j])]) != Int(cache.root), order)
    return (
        edges = Int[Int(candidates[j]) for j in order],
        shared_scores = Float64[shared_scores[j] for j in order],
        visible_counts = Int[visible_counts[j] for j in order],
    )
end

