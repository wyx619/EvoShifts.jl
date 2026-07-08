mutable struct _PrunedTreeNode
    label::String
    branch_length::Float64
    children::Vector{_PrunedTreeNode}
end

function _normalize_tip_set(tree::CompactTree, tips)
    tip_vec =
        if tips isa AbstractString
            String[tips]
        else
            String.(collect(tips))
        end
    unknown = setdiff(tip_vec, tree.tip_labels)
    isempty(unknown) || throw(ArgumentError("Unknown tip labels: $(join(unknown, ", "))"))
    return Set(tip_vec)
end

function _edge_for_child(tree::CompactTree, parent::Integer, child::Integer)
    first_edge = Int(tree.first_child_edge[parent])
    last_edge = Int(tree.last_child_edge[parent])
    for edge in first_edge:last_edge
        tree.child_of_edge[edge] == child && return edge
    end
    throw(ArgumentError("No edge found for parent=$parent child=$child"))
end

function _prune_tree_edge(tree::CompactTree, edge::Integer, keep::Set{String})
    child = Int(tree.child_of_edge[edge])
    branch_length = tree.edge_length[edge]
    if tree.is_tip[child]
        label = tree.node_labels[child]
        label in keep || return nothing
        return _PrunedTreeNode(label, branch_length, _PrunedTreeNode[])
    end

    children = _PrunedTreeNode[]
    for grandchild in tree.children[child]
        grand_edge = _edge_for_child(tree, child, grandchild)
        branch = _prune_tree_edge(tree, grand_edge, keep)
        branch !== nothing && push!(children, branch)
    end
    isempty(children) && return nothing

    if length(children) == 1
        only_child = children[1]
        only_child.branch_length += branch_length
        return only_child
    end

    return _PrunedTreeNode(tree.node_labels[child], branch_length, children)
end

function _build_pruned_tree_root(tree::CompactTree, keep::Set{String})
    children = _PrunedTreeNode[]
    for child in tree.children[Int(tree.root)]
        edge = _edge_for_child(tree, Int(tree.root), child)
        branch = _prune_tree_edge(tree, edge, keep)
        branch !== nothing && push!(children, branch)
    end
    isempty(children) && throw(ArgumentError("Cannot drop all tips from a tree"))
    root = _PrunedTreeNode(tree.node_labels[Int(tree.root)], 0.0, children)
    while length(root.children) == 1 && !isempty(root.children[1].children)
        child = root.children[1]
        root = _PrunedTreeNode(child.label, 0.0, child.children)
    end
    return root
end

function _compact_tree_from_pruned_root(root::_PrunedTreeNode)
    parent_of_node = Int32[]
    children = Vector{Int32}[]
    node_edge_length = Float64[]
    is_tip = BitVector()
    node_labels = String[]

    function add_node!(node::_PrunedTreeNode, parent::Int32)
        idx = Int32(length(parent_of_node) + 1)
        push!(parent_of_node, parent)
        push!(children, Int32[])
        push!(node_edge_length, node.branch_length)
        push!(is_tip, isempty(node.children))
        push!(node_labels, node.label)
        if parent != 0
            push!(children[parent], idx)
        end
        for child in node.children
            add_node!(child, idx)
        end
        return idx
    end

    root_id = add_node!(root, Int32(0))
    nnodes = length(parent_of_node)
    dist_from_root = zeros(Float64, nnodes)
    preorder = Int32[]
    stack = Int32[root_id]
    while !isempty(stack)
        node = pop!(stack)
        push!(preorder, node)
        for child in Iterators.reverse(children[node])
            dist_from_root[child] = dist_from_root[node] + node_edge_length[child]
            push!(stack, child)
        end
    end

    postorder = Int32[]
    postorder_internal = Int32[]
    function dfs(node::Int32)
        for child in children[node]
            dfs(child)
        end
        push!(postorder, node)
        if !is_tip[node]
            push!(postorder_internal, node)
        end
        return nothing
    end
    dfs(root_id)

    tip_ids = Int32[findall(is_tip)...]
    ntips = length(tip_ids)
    nedges = nnodes - 1
    tip_labels = [node_labels[idx] for idx in tip_ids]
    tipname_to_id = Dict{String, Int32}(tip_labels[i] => tip_ids[i] for i in eachindex(tip_ids))
    parent_of_edge = Vector{Int32}(undef, nedges)
    child_of_edge = Vector{Int32}(undef, nedges)
    edge_length = Vector{Float64}(undef, nedges)
    first_child_edge = fill(Int32(0), nnodes)
    last_child_edge = fill(Int32(0), nnodes)

    edge_idx = Int32(1)
    for parent in Int32.(1:nnodes)
        for child in children[parent]
            parent_of_edge[edge_idx] = parent
            child_of_edge[edge_idx] = child
            edge_length[edge_idx] = node_edge_length[child]
            if first_child_edge[parent] == 0
                first_child_edge[parent] = edge_idx
            end
            last_child_edge[parent] = edge_idx
            edge_idx += 1
        end
    end

    return CompactTree(
        ntips,
        nnodes,
        nedges,
        parent_of_edge,
        child_of_edge,
        edge_length,
        root_id,
        parent_of_node,
        dist_from_root,
        is_tip,
        tip_ids,
        postorder,
        postorder_internal,
        preorder,
        children,
        first_child_edge,
        last_child_edge,
        tipname_to_id,
        tip_labels,
        node_labels,
    )
end

function keep_tip(tree::CompactTree, tips)
    keep = _normalize_tip_set(tree, tips)
    isempty(keep) && throw(ArgumentError("Cannot keep zero tips from a tree"))
    length(keep) == tree.ntips && return tree
    root = _build_pruned_tree_root(tree, keep)
    return _compact_tree_from_pruned_root(root)
end

function drop_tip(tree::CompactTree, tips)
    drop = _normalize_tip_set(tree, tips)
    keep = setdiff(Set(tree.tip_labels), drop)
    isempty(keep) && throw(ArgumentError("Cannot drop all tips from a tree"))
    return keep_tip(tree, collect(keep))
end
