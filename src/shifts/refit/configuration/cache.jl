function _shift_edge_segments_with_merge(
    tree::CompactTree,
    shift_edges::AbstractVector{<:Integer},
    merge_map::Union{Nothing, Dict{Int,Int}} = nothing,
)
    edge_segments = shift_edges_to_edge_segments(tree, shift_edges)
    if merge_map === nothing || isempty(merge_map)
        return edge_segments
    end

    used_states = Set{Int}([1])
    for e in 1:tree.nedges
        seg = edge_segments[e][1]
        new_state = get(merge_map, Int(seg.state), Int(seg.state))
        push!(used_states, new_state)
        edge_segments[e] = [SimmapSegment(state = Int32(new_state), length = seg.length)]
    end

    sorted_states = sort!(collect(used_states))
    compact = Dict{Int,Int32}(1 => Int32(1))
    next_state = Int32(2)
    for state in sorted_states
        state == 1 && continue
        compact[state] = next_state
        next_state += Int32(1)
    end
    for e in 1:tree.nedges
        seg = edge_segments[e][1]
        edge_segments[e] = [SimmapSegment(state = compact[Int(seg.state)], length = seg.length)]
    end
    return edge_segments
end

function _shift_oum_cache_from_edges(tree::CompactTree, shift_edges::AbstractVector{<:Integer})
    shift_set = Set{Int32}(Int32.(shift_edges))
    edge_regime = fill(Int32(0), tree.nedges)
    node_regime = fill(Int32(0), tree.nnodes)
    root = Int(tree.root)
    node_regime[root] = Int32(1)
    regime_counter = Int32(1)

    for node in tree.preorder
        tree.is_tip[node] && continue
        for e in Int(tree.first_child_edge[node]):Int(tree.last_child_edge[node])
            if Int32(e) in shift_set
                regime_counter += Int32(1)
                edge_regime[e] = regime_counter
            else
                edge_regime[e] = node_regime[node]
            end
            child = Int(tree.child_of_edge[e])
            node_regime[child] = edge_regime[e]
        end
    end

    edge_first_segment = Vector{Int32}(undef, tree.nedges)
    edge_last_segment = Vector{Int32}(undef, tree.nedges)
    segment_states = Vector{Int32}(undef, tree.nedges)
    segment_lengths = Vector{Float64}(undef, tree.nedges)
    @inbounds for e in 1:tree.nedges
        edge_first_segment[e] = Int32(e)
        edge_last_segment[e] = Int32(e)
        segment_states[e] = edge_regime[e]
        segment_lengths[e] = tree.edge_length[e]
    end

    return OUMEdgeSegmentCache(
        Int(regime_counter),
        1,
        edge_first_segment,
        edge_last_segment,
        segment_states,
        segment_lengths,
    )
end

struct _ShiftCrossproductWorkspace
    spreads::Vector{Float64}
    precision::Vector{Float64}
    means::Matrix{Float64}
    cross::Matrix{Float64}
    xx::Matrix{Float64}
    xx_factor::Matrix{Float64}
    xy::Vector{Float64}
    theta::Vector{Float64}
    contrast::Vector{Float64}
    z::Matrix{Float64}
    edge_a::Vector{Float64}
    edge_v::Vector{Float64}
    shift_mark::Vector{Bool}
    edge_regime::Vector{Int32}
    node_regime::Vector{Int32}
    edge_first_segment::Vector{Int32}
    edge_last_segment::Vector{Int32}
    segment_states::Vector{Int32}
    segment_lengths::Vector{Float64}
end

function _shift_crossproduct_workspace(tree::CompactTree, k::Integer)
    kk = Int(k)
    return _ShiftCrossproductWorkspace(
        zeros(Float64, tree.nnodes),
        zeros(Float64, tree.nnodes),
        zeros(Float64, tree.nnodes, kk),
        Matrix{Float64}(undef, kk, kk),
        Matrix{Float64}(undef, max(kk - 1, 0), max(kk - 1, 0)),
        Matrix{Float64}(undef, max(kk - 1, 0), max(kk - 1, 0)),
        Vector{Float64}(undef, max(kk - 1, 0)),
        Vector{Float64}(undef, max(kk - 1, 0)),
        Vector{Float64}(undef, kk),
        Matrix{Float64}(undef, tree.ntips, kk),
        Vector{Float64}(undef, tree.nedges),
        Vector{Float64}(undef, tree.nedges),
        falses(tree.nedges),
        Vector{Int32}(undef, tree.nedges),
        Vector{Int32}(undef, tree.nnodes),
        Vector{Int32}(undef, tree.nedges),
        Vector{Int32}(undef, tree.nedges),
        Vector{Int32}(undef, tree.nedges),
        Vector{Float64}(undef, tree.nedges),
    )
end

function _compatible_shift_workspace(
    workspace::_ShiftCrossproductWorkspace,
    tree::CompactTree,
    k::Integer,
)
    kk = Int(k)
    return (
        length(workspace.spreads) == tree.nnodes &&
        length(workspace.precision) == tree.nnodes &&
        size(workspace.means, 1) == tree.nnodes &&
        size(workspace.means, 2) == kk &&
        size(workspace.cross, 1) == kk &&
        size(workspace.cross, 2) == kk &&
        size(workspace.z, 1) == tree.ntips &&
        size(workspace.z, 2) == kk &&
        length(workspace.shift_mark) == tree.nedges &&
        length(workspace.edge_regime) == tree.nedges &&
        length(workspace.node_regime) == tree.nnodes &&
        length(workspace.edge_first_segment) == tree.nedges &&
        length(workspace.edge_last_segment) == tree.nedges &&
        length(workspace.segment_states) == tree.nedges &&
        length(workspace.segment_lengths) == tree.nedges
    )
end

function _shift_oum_cache_from_edges!(
    workspace::_ShiftCrossproductWorkspace,
    tree::CompactTree,
    shift_edges::AbstractVector{<:Integer},
)
    shift_mark = workspace.shift_mark
    @inbounds for edge0 in shift_edges
        shift_mark[Int(edge0)] = true
    end

    edge_regime = workspace.edge_regime
    node_regime = workspace.node_regime
    root = Int(tree.root)
    node_regime[root] = Int32(1)
    regime_counter = Int32(1)

    @inbounds for node0 in tree.preorder
        node = Int(node0)
        tree.is_tip[node] && continue
        for e in Int(tree.first_child_edge[node]):Int(tree.last_child_edge[node])
            if shift_mark[e]
                regime_counter += Int32(1)
                edge_regime[e] = regime_counter
            else
                edge_regime[e] = node_regime[node]
            end
            child = Int(tree.child_of_edge[e])
            node_regime[child] = edge_regime[e]
        end
    end

    @inbounds for e in 1:tree.nedges
        workspace.edge_first_segment[e] = Int32(e)
        workspace.edge_last_segment[e] = Int32(e)
        workspace.segment_states[e] = edge_regime[e]
        workspace.segment_lengths[e] = tree.edge_length[e]
    end
    @inbounds for edge0 in shift_edges
        shift_mark[Int(edge0)] = false
    end

    return OUMEdgeSegmentCache(
        Int(regime_counter),
        1,
        workspace.edge_first_segment,
        workspace.edge_last_segment,
        workspace.segment_states,
        workspace.segment_lengths,
    )
end

