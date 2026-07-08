function _fill_shift_screening_edges!(
    edge_a::AbstractVector{Float64},
    edge_v::AbstractVector{Float64},
    tree::CompactTree,
    alpha::Float64,
)
    length(edge_a) == tree.nedges || throw(ArgumentError("edge_a has wrong length"))
    length(edge_v) == tree.nedges || throw(ArgumentError("edge_v has wrong length"))
    @inbounds for e in 1:tree.nedges
        a = exp(-alpha * tree.edge_length[e])
        edge_a[e] = a
        edge_v[e] = (1.0 - a * a) / (2.0 * alpha)
    end
    return edge_a, edge_v
end

