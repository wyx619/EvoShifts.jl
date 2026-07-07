# ============================================================================
# Tree I/O for the simulate stack
# ----------------------------------------------------------------------------
#
# Functional role: convert between the three tree representations used by the
# simulation entry points, and persist them to disk.
#
#     SimulatedTree  (in-memory, produced by yule.jl / birth_death.jl)
#            |
#            | to_compact_tree    to_newick (string)
#            v
#     CompactTree    (canonical internal representation)
#            |
#            | from_compact_tree
#            v
#     realTree = NewickTree.Node  (third-party type, used by src/io.jl)
#            |
#            | save_newick_tree   /   load_newick_tree (in src/io.jl)
#            v
#     disk file (.tre / .nwk)
#
# Newick strings are never used as the internal intermediate between
# SimulatedTree and CompactTree. They are only produced for disk persistence
# or for compatibility with external tools.
#
# The legacy random-coalescence simulator `simulate_ultrametric_newick` is
# kept here because it returns a Newick string and is used by tests as a
# fast standalone tree source. It is **not** a recommended path for new code;
# use `simulate_yule_simtree` or `simulate_birth_death_simtree` instead.
# ============================================================================

# ---------------------------------------------------------------------------
# Lightweight tree node used by the random-coalescence simulator
# ---------------------------------------------------------------------------

mutable struct _MVSimNode
    name::String
    age::Float64
    children::Vector{_MVSimNode}
end

function _mvsim_node_to_newick(node::_MVSimNode, parent_age::Union{Nothing, Float64})
    if isempty(node.children)
        parent_age === nothing && return string(node.name, ';')
        return string(node.name, ':', parent_age - node.age)
    end
    inner = join((_mvsim_node_to_newick(child, node.age) for child in node.children), ',')
    parent_age === nothing && return string('(', inner, ");")
    return string('(', inner, "):", parent_age - node.age)
end

# ---------------------------------------------------------------------------
# SimulatedTree <-> CompactTree conversions
# ---------------------------------------------------------------------------

function _simtree_validate(simtree::SimulatedTree)
    nnodes = length(simtree.parent)
    nnodes >= 1 || throw(ArgumentError("simulated tree must contain at least one node"))
    length(simtree.children) == nnodes || throw(ArgumentError("children length must match parent length"))
    length(simtree.node_time) == nnodes || throw(ArgumentError("node_time length must match parent length"))
    length(simtree.is_tip) == nnodes || throw(ArgumentError("is_tip length must match parent length"))
    roots = findall(==(Int32(0)), simtree.parent)
    length(roots) == 1 || throw(ArgumentError("simulated tree must contain exactly one root"))
    for node in 1:nnodes
        isfinite(simtree.node_time[node]) || throw(ArgumentError("node_time must be finite"))
        if simtree.parent[node] != 0
            parent = Int(simtree.parent[node])
            1 <= parent <= nnodes || throw(ArgumentError("parent index out of range"))
            simtree.node_time[node] >= simtree.node_time[parent] ||
                throw(ArgumentError("node_time must be nondecreasing from root to tips"))
        end
    end
    count(simtree.is_tip) == length(simtree.tip_labels) ||
        throw(ArgumentError("tip_labels length must match number of tips"))
    return Int32(only(roots))
end

function to_compact_tree(simtree::SimulatedTree)
    root = _simtree_validate(simtree)
    nnodes = length(simtree.parent)
    parent_of_node = copy(simtree.parent)
    children = [copy(ch) for ch in simtree.children]
    is_tip = copy(simtree.is_tip)
    node_labels = fill("", nnodes)
    tip_ids = Int32[]
    tip_i = 1
    for node in 1:nnodes
        if is_tip[node]
            push!(tip_ids, Int32(node))
            node_labels[node] = simtree.tip_labels[tip_i]
            tip_i += 1
        end
    end
    ntips = length(tip_ids)
    nedges = nnodes - 1

    dist_from_root = zeros(Float64, nnodes)
    preorder = Int32[]
    stack = Int32[root]
    while !isempty(stack)
        node = pop!(stack)
        push!(preorder, node)
        for child in Iterators.reverse(children[Int(node)])
            dist_from_root[Int(child)] = simtree.node_time[Int(child)] - simtree.node_time[Int(root)]
            push!(stack, child)
        end
    end

    postorder = Int32[]
    postorder_internal = Int32[]
    function dfs(node::Int32)
        for child in children[Int(node)]
            dfs(child)
        end
        push!(postorder, node)
        !is_tip[Int(node)] && push!(postorder_internal, node)
        return nothing
    end
    dfs(root)

    tip_labels = [node_labels[Int(idx)] for idx in tip_ids]
    tipname_to_id = Dict{String, Int32}(tip_labels[i] => tip_ids[i] for i in eachindex(tip_ids))
    parent_of_edge = Vector{Int32}(undef, nedges)
    child_of_edge = Vector{Int32}(undef, nedges)
    edge_length = Vector{Float64}(undef, nedges)
    first_child_edge = fill(Int32(0), nnodes)
    last_child_edge = fill(Int32(0), nnodes)
    edge_idx = Int32(1)
    for parent in Int32.(1:nnodes)
        for child in children[Int(parent)]
            parent_of_edge[edge_idx] = parent
            child_of_edge[edge_idx] = child
            edge_length[edge_idx] = simtree.node_time[Int(child)] - simtree.node_time[Int(parent)]
            edge_length[edge_idx] >= 0.0 || throw(ArgumentError("branch lengths must be non-negative"))
            if first_child_edge[Int(parent)] == 0
                first_child_edge[Int(parent)] = edge_idx
            end
            last_child_edge[Int(parent)] = edge_idx
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

function _simtree_tip_label_by_node(simtree::SimulatedTree)
    labels = fill("", length(simtree.parent))
    tip_i = 1
    for node in eachindex(simtree.parent)
        if simtree.is_tip[node]
            labels[node] = simtree.tip_labels[tip_i]
            tip_i += 1
        end
    end
    return labels
end

function _simtree_newick_node(simtree::SimulatedTree, node::Int32, node_labels::Vector{String})
    inode = Int(node)
    label = node_labels[inode]
    if isempty(simtree.children[inode])
        body = label
    else
        body = "(" * join((_simtree_newick_node(simtree, child, node_labels) for child in simtree.children[inode]), ",") * ")"
    end
    parent = simtree.parent[inode]
    parent == 0 && return body * ";"
    branch = simtree.node_time[inode] - simtree.node_time[Int(parent)]
    return body * ":" * string(branch)
end

"""
    to_newick(simtree::SimulatedTree) -> String

Serialize a `SimulatedTree` to a Newick string. The returned string ends with
a trailing semicolon. Use `save_newick_tree(path, simtree)` to write it
directly to disk.
"""
function to_newick(simtree::SimulatedTree)
    root = _simtree_validate(simtree)
    return _simsim_newick(simtree, root)
end

function _simsim_newick(simtree::SimulatedTree, root::Int32)
    return _simtree_newick_node(simtree, root, _simtree_tip_label_by_node(simtree))
end

# ---------------------------------------------------------------------------
# CompactTree -> realTree (NewickTree.Node) conversion
# ---------------------------------------------------------------------------

"""
    from_compact_tree(tree::CompactTree) -> NewickTree.Node

Convert the engine's internal `CompactTree` into a `NewickTree.Node` (a
"realTree" object). The root is preserved as the top-level element, branch
lengths and tip labels are carried over, internal node labels are written
when non-empty.

This is the canonical bridge to third-party Newick consumers (`ape`,
`phytools`) when you want to hand a tree to external code without going
through the disk roundtrip.
"""
function from_compact_tree(tree::CompactTree)
    children_dl = _build_children_with_lengths(tree)
    return _build_nw_node(tree, Int(tree.root), children_dl)
end

function _build_children_with_lengths(tree::CompactTree)
    children_dl = Dict{Int, Vector{Tuple{Int, Float64}}}()
    for e in 1:tree.nedges
        p = Int(tree.parent_of_edge[e])
        c = Int(tree.child_of_edge[e])
        d = tree.edge_length[e]
        list = get!(children_dl, p) do
            return Tuple{Int, Float64}[]
        end
        push!(list, (c, d))
    end
    return children_dl
end

function _build_nw_node(tree::CompactTree, node::Int, children_dl::Dict{Int, Vector{Tuple{Int, Float64}}})
    has_children = haskey(children_dl, node)
    label = tree.is_tip[node] ? _compact_tip_label(tree, node) : _compact_internal_label(tree, node)
    if !has_children
        return NewickTree.Node(UInt32(node), NewickData(d = 0.0, n = label))
    end
    nw = NewickTree.Node(UInt32(node), NewickData(d = 0.0, n = label))
    for (child, d) in children_dl[node]
        cnw = _build_nw_node_with_distance(tree, child, d, children_dl)
        push!(nw, cnw)
    end
    return nw
end

function _build_nw_node_with_distance(tree::CompactTree, node::Int, d::Float64, children_dl::Dict{Int, Vector{Tuple{Int, Float64}}})
    has_children = haskey(children_dl, node)
    label = tree.is_tip[node] ? _compact_tip_label(tree, node) : _compact_internal_label(tree, node)
    if !has_children
        return NewickTree.Node(UInt32(node), NewickData(d = d, n = label))
    end
    nw = NewickTree.Node(UInt32(node), NewickData(d = d, n = label))
    for (child, cd) in children_dl[node]
        push!(nw, _build_nw_node_with_distance(tree, child, cd, children_dl))
    end
    return nw
end

function _compact_tip_label(tree::CompactTree, node::Int)
    idx = findfirst(==(Int32(node)), tree.tip_ids)
    idx === nothing && return ""
    return tree.tip_labels[idx]
end

function _compact_internal_label(tree::CompactTree, node::Int)
    label = tree.node_labels[node]
    return isempty(label) ? "" : label
end

"""
    to_real_tree(tree::CompactTree) -> NewickTree.Node
    to_real_tree(simtree::SimulatedTree) -> NewickTree.Node

Convenience aliases for [`from_compact_tree`](@ref). The first form takes a
`CompactTree` directly; the second form converts a `SimulatedTree` first
through `to_compact_tree`.
"""
to_real_tree(tree::CompactTree) = from_compact_tree(tree)
to_real_tree(simtree::SimulatedTree) = from_compact_tree(to_compact_tree(simtree))

# ---------------------------------------------------------------------------
# Random-coalescence tree simulator (Newick string output)
# ---------------------------------------------------------------------------

"""
    simulate_ultrametric_newick(
        n_tips::Integer;
        tree_height::Real = 1.0,
        tip_prefix::AbstractString = "t",
        rng::AbstractRNG = Random.GLOBAL_RNG,
    )

Generate a random rooted bifurcating ultrametric tree in Newick format.

The topology is built by random backward coalescence and all tip-to-root path
lengths equal `tree_height`. This helper is intended for tests and small
simulation studies, **not** as a fitted birth-death or Yule process. For
realistic tree simulation use `simulate_yule_simtree` or
`simulate_birth_death_simtree` and then `to_newick`.
"""
function simulate_ultrametric_newick(
    n_tips::Integer;
    tree_height::Real = 1.0,
    tip_prefix::AbstractString = "t",
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    n_tips >= 2 || throw(ArgumentError("simulate_ultrametric_newick requires at least 2 tips"))
    tree_height > 0 || throw(ArgumentError("tree_height must be positive"))

    lineages = [_MVSimNode("$(tip_prefix)$(i)", 0.0, _MVSimNode[]) for i in 1:n_tips]
    ages = randexp(rng, n_tips - 1)
    sort!(ages)
    ages .*= Float64(tree_height) / ages[end]

    for age in ages
        idx = randperm(rng, length(lineages))[1:2]
        i, j = minmax(idx...)
        left = lineages[i]
        right = lineages[j]
        parent = _MVSimNode("", age, _MVSimNode[left, right])
        deleteat!(lineages, j)
        deleteat!(lineages, i)
        push!(lineages, parent)
    end

    return _mvsim_node_to_newick(only(lineages), nothing)
end

# ---------------------------------------------------------------------------
# Disk I/O: serialize any tree representation to a .tre / .nwk file
# ---------------------------------------------------------------------------

"""
    save_newick_tree(path, source; append::Bool = false) -> String

Persist a Newick string to a file on disk. The `source` argument may be a
`String`, a `SimulatedTree`, a `CompactTree`, or a `NewickTree.Node`. The
trailing semicolon is added if missing.

Examples:

```julia
save_newick_tree("tree.tre", simulate_yule_simtree(50))             # SimulatedTree
save_newick_tree("tree.tre", to_compact_tree(simtree))               # CompactTree
save_newick_tree("tree.tre", from_compact_tree(tree))                # NewickTree.Node
save_newick_tree("tree.tre", simulate_ultrametric_newick(50))        # Newick string
```

Returns the Newick string that was written.
"""
function save_newick_tree(path::AbstractString, simtree::SimulatedTree; append::Bool = false)
    return _write_newick_file(path, to_newick(simtree), append)
end

function save_newick_tree(path::AbstractString, tree::CompactTree; append::Bool = false)
    io = IOBuffer()
    NewickTree.writenw(io, from_compact_tree(tree))
    return _write_newick_file(path, _buffer_to_string(io), append)
end

function save_newick_tree(path::AbstractString, nw::NewickTree.Node; append::Bool = false)
    io = IOBuffer()
    NewickTree.writenw(io, nw)
    return _write_newick_file(path, _buffer_to_string(io), append)
end

function save_newick_tree(path::AbstractString, newick::AbstractString; append::Bool = false)
    content = String(newick)
    endswith(strip(content), ';') || (content = string(strip(content), ';'))
    return _write_newick_file(path, content, append)
end

function _write_newick_file(path::AbstractString, content::AbstractString, append::Bool)
    open(path, append ? "a" : "w") do io
        write(io, content)
        write(io, '\n')
    end
    return String(content)
end

_buffer_to_string(io::IOBuffer) = String(take!(io))
