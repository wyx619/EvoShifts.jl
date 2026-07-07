function _pruning_group_path_lambdas(lambda_max::Float64, base_seq::AbstractVector{<:Real}; lmax::Union{Nothing, Float64} = nothing)
    lmax = lmax === nothing ? 1.2 * lambda_max + 1.0 : lmax
    return lmax .* (0.5 .^ Float64.(base_seq))
end

@inline _l1ou_proposal_alpha(alpha::Real) = abs(Float64(alpha)) <= 1e-6 ? 0.0 : Float64(alpha)

function _l1ou_r_node_ids(tree::CompactTree)
    ids = zeros(Int, tree.nnodes)
    @inbounds for (i, tip) in enumerate(tree.tip_ids)
        ids[Int(tip)] = i
    end
    next_id = tree.ntips + 1
    @inbounds for node0 in tree.preorder
        node = Int(node0)
        if !tree.is_tip[node]
            ids[node] = next_id
            next_id += 1
        end
    end
    return ids
end

function _l1ou_postorder_edges(tree::CompactTree, r_node_id::AbstractVector{<:Integer})
    edges = collect(1:tree.nedges)
    sort!(edges; by = e -> (-Int(r_node_id[Int(tree.parent_of_edge[e])]), e))
    return edges
end

@inline function _l1ou_ou_edge_length(tree::CompactTree, edge::Integer, alpha::Float64, tree_height::Float64)
    alpha <= 0.0 && return tree.edge_length[Int(edge)]
    parent = Int(tree.parent_of_edge[Int(edge)])
    child = Int(tree.child_of_edge[Int(edge)])
    tp = tree.dist_from_root[parent]
    tc = tree.dist_from_root[child]
    return (exp(-2.0 * alpha * (tree_height - tc)) - exp(-2.0 * alpha * (tree_height - tp))) / (2.0 * alpha)
end

@inline function _l1ou_root_edge_length(tree::CompactTree, alpha::Float64, root_model::Symbol, tree_height::Float64)
    (root_model === :OUrandomRoot && alpha > 0.0) || return 0.0
    return exp(-2.0 * alpha * tree_height) / (2.0 * alpha)
end

struct _L1OURowWhiteningPlan
    left_node::Vector{Int}
    right_node::Vector{Int}
    parent_node::Vector{Int}
    contrast_inv_scale::Vector{Float64}
    parent_left_weight::Vector{Float64}
    parent_right_weight::Vector{Float64}
    root_node::Int
    root_inv_scale::Float64
end

struct _L1OURowWhiteningWorkspace
    means::Vector{Float64}
    transformed::Vector{Float64}
end

function _l1ou_row_whitening_plan(
    tree::CompactTree,
    alpha::Float64;
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    n = tree.ntips
    r_node_id = _l1ou_r_node_ids(tree)
    edge_order = _l1ou_postorder_edges(tree, r_node_id)
    tree_height = maximum(tree.dist_from_root[tree.tip_ids])

    edge_len = zeros(Float64, tree.nedges)
    edge_by_child = Dict{Int, Int}()
    @inbounds for e in 1:tree.nedges
        edge_len[e] = _l1ou_ou_edge_length(tree, e, alpha, tree_height)
        edge_by_child[Int(tree.child_of_edge[e])] = e
    end

    active = collect(1:n)
    left_node = Int[]
    right_node = Int[]
    parent_node = Int[]
    contrast_inv_scale = Float64[]
    parent_left_weight = Float64[]
    parent_right_weight = Float64[]
    sizehint!(left_node, max(n - 1, 0))
    sizehint!(right_node, max(n - 1, 0))
    sizehint!(parent_node, max(n - 1, 0))
    sizehint!(contrast_inv_scale, max(n - 1, 0))
    sizehint!(parent_left_weight, max(n - 1, 0))
    sizehint!(parent_right_weight, max(n - 1, 0))

    root_edge_len = 0.0
    idx = 1
    while idx <= length(edge_order) - 1 && length(active) > 1
        e1 = edge_order[idx]
        e2 = edge_order[idx + 1]
        p1 = Int(r_node_id[Int(tree.parent_of_edge[e1])])
        p2 = Int(r_node_id[Int(tree.parent_of_edge[e2])])
        if p1 != p2
            idx += 1
            continue
        end
        child1 = Int(tree.child_of_edge[e1])
        child2 = Int(tree.child_of_edge[e2])
        parent = Int(tree.parent_of_edge[e1])
        i1 = Int(r_node_id[child1])
        i2 = Int(r_node_id[child2])
        t1 = edge_len[e1]
        t2 = edge_len[e2]
        u = t1 + t2
        u > 0.0 || throw(ArgumentError("non-positive l1ou merge variance in row whitening"))
        push!(left_node, child1)
        push!(right_node, child2)
        push!(parent_node, parent)
        push!(contrast_inv_scale, 1.0 / sqrt(u))
        push!(parent_left_weight, t2 / u)
        push!(parent_right_weight, t1 / u)

        e3 = get(edge_by_child, parent, 0)
        if e3 != 0
            edge_len[e3] += 1.0 / (1.0 / t1 + 1.0 / t2)
        else
            root_edge_len += 1.0 / (1.0 / t1 + 1.0 / t2)
        end
        filter!(x -> x != i1 && x != i2, active)
        push!(active, p1)
        idx += 2
    end
    length(active) == 1 || throw(ArgumentError("l1ou row whitening requires a binary postorder-compatible tree"))
    length(left_node) == n - 1 || throw(ArgumentError("l1ou row whitening expected $(n - 1) contrasts"))
    root_len = root_edge_len + _l1ou_root_edge_length(tree, alpha, root_model, tree_height)
    root_len > 0.0 || throw(ArgumentError("non-positive l1ou root variance in row whitening"))
    root_node = findfirst(==(active[1]), r_node_id)
    root_node === nothing && throw(ArgumentError("l1ou row whitening could not recover root node"))
    return _L1OURowWhiteningPlan(
        left_node,
        right_node,
        parent_node,
        contrast_inv_scale,
        parent_left_weight,
        parent_right_weight,
        Int(root_node),
        1.0 / sqrt(root_len),
    )
end

_l1ou_row_whitening_workspace(tree::CompactTree, plan::_L1OURowWhiteningPlan) =
    _L1OURowWhiteningWorkspace(zeros(Float64, tree.nnodes), zeros(Float64, length(plan.left_node) + 1))

function _l1ou_finish_row_whitening!(
    out::AbstractVector{Float64},
    plan::_L1OURowWhiteningPlan,
    rows::AbstractVector{<:Integer},
    workspace::_L1OURowWhiteningWorkspace,
)
    length(out) == length(rows) || throw(ArgumentError("row-whitened output length does not match requested rows"))
    means = workspace.means
    transformed = workspace.transformed
    @inbounds for k in eachindex(plan.left_node)
        left = plan.left_node[k]
        right = plan.right_node[k]
        parent = plan.parent_node[k]
        m1 = means[left]
        m2 = means[right]
        transformed[k] = (m1 - m2) * plan.contrast_inv_scale[k]
        means[parent] = m1 * plan.parent_left_weight[k] + m2 * plan.parent_right_weight[k]
    end
    transformed[length(plan.left_node) + 1] = means[plan.root_node] * plan.root_inv_scale
    @inbounds for (rr, row0) in enumerate(rows)
        row = Int(row0)
        1 <= row <= length(transformed) || throw(ArgumentError("requested l1ou row $row outside 1:$(length(transformed))"))
        out[rr] = transformed[row]
    end
    return out
end

function _l1ou_sqrt_inv_covariance_transpose(
    tree::CompactTree,
    alpha::Float64;
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    n = tree.ntips
    r_node_id = _l1ou_r_node_ids(tree)
    edge_order = _l1ou_postorder_edges(tree, r_node_id)
    tree_height = maximum(tree.dist_from_root[tree.tip_ids])

    F = zeros(Float64, n, 2n - 1)
    G = zeros(Float64, n, 2n - 1)
    @inbounds for i in 1:n
        F[i, i] = 1.0
        G[i, i] = 1.0
    end
    D = zeros(Float64, n, n)

    edge_len = zeros(Float64, tree.nedges)
    edge_by_child = Dict{Int, Int}()
    @inbounds for e in 1:tree.nedges
        edge_len[e] = _l1ou_ou_edge_length(tree, e, alpha, tree_height)
        edge_by_child[Int(r_node_id[Int(tree.child_of_edge[e])])] = e
    end

    active = collect(1:n)
    root_edge_len = 0.0
    counter = 1
    idx = 1
    while idx <= length(edge_order) - 1 && length(active) > 1
        e1 = edge_order[idx]
        e2 = edge_order[idx + 1]
        p1 = Int(r_node_id[Int(tree.parent_of_edge[e1])])
        p2 = Int(r_node_id[Int(tree.parent_of_edge[e2])])
        if p1 != p2
            idx += 1
            continue
        end
        i1 = Int(r_node_id[Int(tree.child_of_edge[e1])])
        i2 = Int(r_node_id[Int(tree.child_of_edge[e2])])
        i3 = p1
        t1 = edge_len[e1]
        t2 = edge_len[e2]
        u = t1 + t2
        us = sqrt(u)
        @inbounds for r in 1:n
            D[r, counter] = (F[r, i1] - F[r, i2]) / us
            F[r, i3] = (F[r, i1] * t2 + F[r, i2] * t1) / u
            G[r, i3] = G[r, i1] + G[r, i2]
        end
        e3 = get(edge_by_child, i3, 0)
        if e3 != 0
            edge_len[e3] += 1.0 / (1.0 / t1 + 1.0 / t2)
        else
            root_edge_len += 1.0 / (1.0 / t1 + 1.0 / t2)
        end
        filter!(x -> x != i1 && x != i2, active)
        push!(active, i3)
        counter += 1
        idx += 2
    end
    length(active) == 1 || throw(ArgumentError("l1ou sqrt-inverse covariance requires a binary postorder-compatible tree"))
    root_len = root_edge_len + _l1ou_root_edge_length(tree, alpha, root_model, tree_height)
    @inbounds for r in 1:n
        D[r, counter] = F[r, active[1]] / sqrt(root_len)
    end
    return transpose(D)
end

function _l1ou_design_matrix(tree::CompactTree, cache::OUShiftTreeCache, alpha::Float64)
    X = zeros(Float64, tree.ntips, tree.nedges)
    weights = _precompute_design_weights(tree, 1:tree.nedges, alpha)
    @inbounds for e in 1:tree.nedges
        w = weights[e]
        for tip_pos in cache.descendant_tip_positions[e]
            X[tip_pos, e] = w
        end
    end
    return X
end

function _l1ou_design_matrix(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    alpha::Float64,
    candidates::AbstractVector{<:Integer},
)
    X = zeros(Float64, tree.ntips, length(candidates))
    weights = _precompute_design_weights(tree, candidates, alpha)
    @inbounds for (j, e0) in enumerate(candidates)
        e = Int(e0)
        w = weights[j]
        for tip_pos in cache.descendant_tip_positions[e]
            X[tip_pos, j] = w
        end
    end
    return X
end

function _l1ou_whiten_observed_response_rows!(
    out::AbstractVector{Float64},
    sqrt_inv_cov_t::AbstractMatrix{Float64},
    trait::AbstractVector{<:Real},
    rows::AbstractVector{<:Integer},
    obs_idx::AbstractVector{<:Integer},
)
    length(out) == length(rows) || throw(ArgumentError("whitened response output length does not match observed rows"))
    @inbounds for (rr, row0) in enumerate(rows)
        r = Int(row0)
        s = 0.0
        for cc in obs_idx
            c = Int(cc)
            s += sqrt_inv_cov_t[r, c] * Float64(trait[c])
        end
        out[rr] = s
    end
    return out
end

function _l1ou_whiten_observed_response_rows_pruning!(
    out::AbstractVector{Float64},
    plan::_L1OURowWhiteningPlan,
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    rows::AbstractVector{<:Integer},
    obs_idx::AbstractVector{<:Integer},
    workspace::_L1OURowWhiteningWorkspace,
)
    length(trait) == tree.ntips || throw(ArgumentError("trait must have $(tree.ntips) entries"))
    fill!(workspace.means, 0.0)
    @inbounds for pos0 in obs_idx
        pos = Int(pos0)
        1 <= pos <= tree.ntips || throw(ArgumentError("observed tip index $pos outside 1:$(tree.ntips)"))
        workspace.means[Int(tree.tip_ids[pos])] = Float64(trait[pos])
    end
    return _l1ou_finish_row_whitening!(out, plan, rows, workspace)
end

function _l1ou_whiten_shift_column_rows_pruning!(
    out::AbstractVector{Float64},
    plan::_L1OURowWhiteningPlan,
    tree::CompactTree,
    cache::OUShiftTreeCache,
    edge::Integer,
    weight::Float64,
    rows::AbstractVector{<:Integer},
    workspace::_L1OURowWhiteningWorkspace,
)
    fill!(workspace.means, 0.0)
    @inbounds for pos in _get_descendant_tips_positions(cache, edge)
        workspace.means[Int(tree.tip_ids[pos])] = weight
    end
    return _l1ou_finish_row_whitening!(out, plan, rows, workspace)
end

function _l1ou_alpha_groups(alpha_vec::AbstractVector{<:Real})
    groups = Dict{Float64, Vector{Int}}()
    order = Float64[]
    @inbounds for i in eachindex(alpha_vec)
        alpha = _l1ou_proposal_alpha(alpha_vec[i])
        if !haskey(groups, alpha)
            groups[alpha] = Int[]
            push!(order, alpha)
        end
        push!(groups[alpha], Int(i))
    end
    return order, groups
end

