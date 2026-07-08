function _shift_screening_edges(tree::CompactTree, alpha::Float64, sigma2::Float64)
    edge_a = ones(Float64, tree.nedges)
    edge_v = similar(edge_a)
    if alpha <= 0.0
        @inbounds for e in 1:tree.nedges
            edge_v[e] = sigma2 * tree.edge_length[e]
        end
    else
        @inbounds for e in 1:tree.nedges
            a = exp(-alpha * tree.edge_length[e])
            edge_a[e] = a
            edge_v[e] = sigma2 * (1.0 - a * a) / (2.0 * alpha)
        end
    end
    return edge_a, edge_v
end

function _tree_whiten_vector(
    tree::CompactTree,
    values::AbstractVector{<:Real},
    edge_a::AbstractVector{<:Real},
    edge_v::AbstractVector{<:Real},
)
    y = Float64.(values)
    length(y) == tree.ntips || throw(ArgumentError("values must have $(tree.ntips) entries"))
    means = zeros(Float64, tree.nnodes)
    spreads = zeros(Float64, tree.nnodes)
    contrasts = Float64[]
    sizehint!(contrasts, tree.ntips - 1)

    @inbounds for (i, node) in enumerate(tree.tip_ids)
        means[node] = y[i]
    end

    @inbounds for node in tree.postorder_internal
        length(tree.children[node]) == 2 || throw(ArgumentError("tree-operator whitening requires a bifurcating tree"))
        child1 = tree.children[node][1]
        child2 = tree.children[node][2]
        edge1 = Int(tree.first_child_edge[node])
        edge2 = Int(tree.last_child_edge[node])

        a1 = Float64(edge_a[edge1])
        a2 = Float64(edge_a[edge2])
        abs(a1) > 0.0 && abs(a2) > 0.0 || throw(ArgumentError("edge transition coefficient must be non-zero"))

        v1 = (spreads[child1] + Float64(edge_v[edge1])) / (a1 * a1)
        v2 = (spreads[child2] + Float64(edge_v[edge2])) / (a2 * a2)
        m1 = means[child1] / a1
        m2 = means[child2] / a2
        denom = v1 + v2
        denom > 0.0 || throw(ArgumentError("non-positive contrast variance in tree whitening"))

        push!(contrasts, (m1 - m2) / sqrt(denom))
        invdenom = 1.0 / denom
        means[node] = (m1 * v2 + m2 * v1) * invdenom
        spreads[node] = (v1 * v2) * invdenom
    end
    return contrasts
end

struct _TreeWhitenColumnWorkspace
    means::Vector{Float64}
    spreads::Vector{Float64}
    contrast_child1::Vector{Float64}
    contrast_child2::Vector{Float64}
    parent_child1::Vector{Float64}
    parent_child2::Vector{Float64}
end

_tree_whiten_column_workspace(tree::CompactTree) =
    _TreeWhitenColumnWorkspace(
        zeros(Float64, tree.nnodes),
        zeros(Float64, tree.nnodes),
        zeros(Float64, tree.nnodes),
        zeros(Float64, tree.nnodes),
        zeros(Float64, tree.nnodes),
        zeros(Float64, tree.nnodes),
    )

function _tree_whiten_column_workspace(
    tree::CompactTree,
    edge_a::AbstractVector{<:Real},
    edge_v::AbstractVector{<:Real},
)
    ws = _tree_whiten_column_workspace(tree)
    spreads = ws.spreads
    @inbounds for node0 in tree.tip_ids
        spreads[Int(node0)] = 0.0
    end
    @inbounds for node0 in tree.postorder_internal
        node = Int(node0)
        length(tree.children[node]) == 2 ||
            throw(ArgumentError("tree-operator whitening requires a bifurcating tree"))
        child1 = Int(tree.children[node][1])
        child2 = Int(tree.children[node][2])
        edge1 = Int(tree.first_child_edge[node])
        edge2 = Int(tree.last_child_edge[node])
        a1 = Float64(edge_a[edge1])
        a2 = Float64(edge_a[edge2])
        abs(a1) > 0.0 && abs(a2) > 0.0 || throw(ArgumentError("edge transition coefficient must be non-zero"))

        v1 = (spreads[child1] + Float64(edge_v[edge1])) / (a1 * a1)
        v2 = (spreads[child2] + Float64(edge_v[edge2])) / (a2 * a2)
        denom = v1 + v2
        denom > 0.0 || throw(ArgumentError("non-positive contrast variance in tree whitening"))
        invsqrt = 1.0 / sqrt(denom)
        invdenom = 1.0 / denom
        ws.contrast_child1[node] = invsqrt / a1
        ws.contrast_child2[node] = -invsqrt / a2
        ws.parent_child1[node] = v2 * invdenom / a1
        ws.parent_child2[node] = v1 * invdenom / a2
        spreads[node] = (v1 * v2) * invdenom
    end
    return ws
end

function _tree_whiten_shift_column!(
    out::AbstractVector{Float64},
    tree::CompactTree,
    cache::OUShiftTreeCache,
    edge::Integer,
    weight::Float64,
    edge_a::AbstractVector{<:Real},
    edge_v::AbstractVector{<:Real},
    workspace::_TreeWhitenColumnWorkspace,
)
    length(out) == tree.ntips - 1 || throw(ArgumentError("out must have $(tree.ntips - 1) entries"))
    means = workspace.means
    spreads = workspace.spreads

    @inbounds for node0 in tree.tip_ids
        node = Int(node0)
        means[node] = 0.0
        spreads[node] = 0.0
    end
    @inbounds for p in _get_descendant_tips_positions(cache, edge)
        means[Int(tree.tip_ids[p])] = weight
    end

    idx = 1
    @inbounds for node0 in tree.postorder_internal
        node = Int(node0)
        length(tree.children[node]) == 2 ||
            throw(ArgumentError("tree-operator whitening requires a bifurcating tree"))
        child1 = Int(tree.children[node][1])
        child2 = Int(tree.children[node][2])
        m1 = means[child1]
        m2 = means[child2]
        out[idx] = workspace.contrast_child1[node] * m1 + workspace.contrast_child2[node] * m2
        idx += 1
        means[node] = workspace.parent_child1[node] * m1 + workspace.parent_child2[node] * m2
    end
    return out
end

