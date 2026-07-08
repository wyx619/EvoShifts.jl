function _edge_of_child(tree::CompactTree)
    edge_of_child = zeros(Int32, tree.nnodes)
    @inbounds for e in 1:tree.nedges
        edge_of_child[Int(tree.child_of_edge[e])] = Int32(e)
    end
    return edge_of_child
end

function _ou_shift_fitted_means(
    tree::CompactTree,
    edge_segments::Vector{Vector{SimmapSegment}},
    theta::AbstractVector{<:Real},
    alpha::Real,
)
    node_mean = fill(NaN, tree.nnodes)
    node_mean[Int(tree.root)] = Float64(theta[1])
    a = Float64(alpha)
    @inbounds for node32 in tree.preorder
        node = Int(node32)
        tree.is_tip[node] && continue
        parent_mean = node_mean[node]
        for edge in Int(tree.first_child_edge[node]):Int(tree.last_child_edge[node])
            child_mean = parent_mean
            for seg in edge_segments[edge]
                th = Float64(theta[Int(seg.state)])
                child_mean = th + exp(-a * seg.length) * (child_mean - th)
            end
            node_mean[Int(tree.child_of_edge[edge])] = child_mean
        end
    end
    fitted = Vector{Float64}(undef, tree.ntips)
    @inbounds for (i, tip) in enumerate(tree.tip_ids)
        fitted[i] = node_mean[Int(tip)]
    end
    return fitted
end

function _shift_values_from_theta(
    tree::CompactTree,
    edge_segments::Vector{Vector{SimmapSegment}},
    shift_edges::AbstractVector{<:Integer},
    theta::AbstractVector{<:Real},
)
    edge_of_child = _edge_of_child(tree)
    values = Vector{Float64}(undef, length(shift_edges))
    @inbounds for (i, edge0) in enumerate(shift_edges)
        edge = Int(edge0)
        child_state = Int(edge_segments[edge][1].state)
        parent = Int(tree.parent_of_edge[edge])
        parent_state =
            parent == Int(tree.root) ? 1 :
            Int(edge_segments[Int(edge_of_child[parent])][1].state)
        values[i] = Float64(theta[child_state]) - Float64(theta[parent_state])
    end
    return values
end

function _shift_means_from_shift_values(
    tree::CompactTree,
    shift_edges::AbstractVector{<:Integer},
    shift_values::AbstractVector{<:Real},
    alpha::Real,
)
    if isempty(shift_edges)
        return Float64[]
    end
    a = Float64(alpha)
    shift_means = Vector{Float64}(undef, length(shift_edges))
    tree_height = maximum(tree.dist_from_root[tree.tip_ids])
    @inbounds for (i, edge0) in enumerate(shift_edges)
        edge = Int(edge0)
        parent = Int(tree.parent_of_edge[edge])
        age = tree_height - tree.dist_from_root[parent]
        scale = 1.0 - exp(-a * age)
        shift_means[i] = abs(scale) <= 1e-12 ? Float64(shift_values[i]) : Float64(shift_values[i]) / scale
    end
    return shift_means
end

function _edge_optima_from_theta(
    edge_segments::Vector{Vector{SimmapSegment}},
    theta::AbstractVector{<:Real},
)
    edge_optima = Vector{Float64}(undef, length(edge_segments))
    @inbounds for e in eachindex(edge_segments)
        edge_optima[e] = Float64(theta[Int(edge_segments[e][1].state)])
    end
    return edge_optima
end


