function _fill_oum_profile_z!(
    z::AbstractMatrix{Float64},
    tree::CompactTree,
    cache::OUMEdgeSegmentCache,
    trait::AbstractVector{<:Real},
    alpha::Float64,
    basis_workspace::AbstractMatrix{Float64},
)
    nregimes = size(z, 2) - 1
    size(basis_workspace, 1) == tree.nnodes || throw(ArgumentError("basis workspace has wrong node count"))
    size(basis_workspace, 2) >= nregimes || throw(ArgumentError("basis workspace has too few columns"))
    @inbounds for i in 1:tree.ntips
        z[i, 1] = Float64(trait[i])
    end

    root = Int(tree.root)
    @inbounds for j in 1:nregimes
        basis_workspace[root, j] = j == cache.root_regime ? 1.0 : 0.0
    end

    @inbounds for node0 in tree.preorder
        node = Int(node0)
        tree.is_tip[node] && continue
        for edge in Int(tree.first_child_edge[node]):Int(tree.last_child_edge[node])
            child = Int(tree.child_of_edge[edge])
            for j in 1:nregimes
                basis_workspace[child, j] = basis_workspace[node, j]
            end
            first_seg = Int(cache.edge_first_segment[edge])
            last_seg = Int(cache.edge_last_segment[edge])
            for seg_idx in first_seg:last_seg
                decay = exp(-alpha * cache.segment_lengths[seg_idx])
                state = Int(cache.segment_states[seg_idx])
                for j in 1:nregimes
                    basis_workspace[child, j] *= decay
                end
                basis_workspace[child, state] += 1.0 - decay
            end
        end
    end

    @inbounds for (i, tip0) in enumerate(tree.tip_ids)
        tip = Int(tip0)
        for j in 1:nregimes
            z[i, j + 1] = basis_workspace[tip, j]
        end
    end
    return z
end

