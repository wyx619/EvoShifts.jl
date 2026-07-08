"""
    load_newick_tree(path::AbstractString)

Load a tree from a Newick file.

This function intentionally reads the file contents first and then calls
`NewickTree.readnw` on the Newick string itself. This avoids path-handling
ambiguities on some platforms, especially with non-ASCII file names.

If the raw parse fails, the loader retries after stripping internal node labels
such as bootstrap/support values (`)100.0:` -> `):`). This keeps the engine
robust to large empirical trees whose support annotations are irrelevant to the
core likelihood kernels.
"""
function load_newick_tree(path::AbstractString)
    isfile(path) || throw(ArgumentError("Tree file does not exist: $path"))
    newick = String(strip(read(path, String)))
    endswith(newick, ';') || throw(ArgumentError("Malformed Newick file: missing trailing semicolon"))
    try
        return NewickTree.readnw(newick)
    catch first_error
        cleaned = String(replace(newick, r"\)([^():;,]+):" => "):"))
        cleaned == newick && rethrow(first_error)
        try
            return NewickTree.readnw(cleaned)
        catch
            rethrow(first_error)
        end
    end
end

@inline function _safe_node_children(node::T) where {T}
    if !isdefined(node, :children)
        return T[]
    end
    raw_children = getfield(node, :children)
    child_nodes = T[]
    for i in eachindex(raw_children)
        if isassigned(raw_children, i)
            push!(child_nodes, raw_children[i])
        end
    end
    return child_nodes
end

"""
    to_compact_tree(tree::NewickTree.Node)

Convert a parsed `NewickTree` tree into the engine's internal `CompactTree`
representation. This is the canonical ingest boundary for external tree input.
"""
function to_compact_tree(tree::T) where {T <: NewickTree.Node}
    root_node = tree
    while !NewickTree.isroot(root_node)
        root_node = root_node.parent
    end

    parent_of_node = Int32[]
    children = Vector{Int32}[]
    node_edge_length = Float64[]
    is_tip = BitVector()
    node_labels = String[]
    node_to_idx = IdDict{T, Int32}()

    stack = Tuple{T, Int32}[(root_node, 0)]
    while !isempty(stack)
        node, parent_idx = pop!(stack)
        haskey(node_to_idx, node) && continue

        idx = Int32(length(parent_of_node) + 1)
        node_to_idx[node] = idx
        push!(parent_of_node, parent_idx)
        push!(children, Int32[])
        push!(node_edge_length, NewickTree.isroot(node) ? 0.0 : Float64(NewickTree.distance(node)))
        push!(node_labels, String(NewickTree.name(node)))

        child_nodes = _safe_node_children(node)
        push!(is_tip, isempty(child_nodes))

        if parent_idx != 0
            push!(children[parent_idx], idx)
        end

        for child in reverse(child_nodes)
            push!(stack, (child, idx))
        end
    end

    nnodes = length(parent_of_node)
    root = get(node_to_idx, root_node, Int32(0))
    root == 0 && throw(ErrorException("Failed to identify root node while building CompactTree"))

    dist_from_root = zeros(Float64, nnodes)
    preorder = Int32[]
    stack2 = Int32[root]
    while !isempty(stack2)
        node = pop!(stack2)
        push!(preorder, node)
        for child in Iterators.reverse(children[node])
            dist_from_root[child] = dist_from_root[node] + node_edge_length[child]
            push!(stack2, child)
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
    dfs(root)

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
        root,
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

preprocess_tree(tree::T) where {T <: NewickTree.Node} = to_compact_tree(tree)


