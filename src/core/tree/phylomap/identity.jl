@inline function _phylomap_descendant_labels(tree::CompactTree, tip_positions::Vector{Int32})
    labels = Vector{String}(undef, length(tip_positions))
    @inbounds for (i, tip_pos) in enumerate(tip_positions)
        labels[i] = tree.tip_labels[Int(tip_pos)]
    end
    return labels
end

function _phylomap_first_tip_label(tree::CompactTree, map::PhyloMap, node::Integer)
    inode = Int(node)
    1 <= inode <= tree.nnodes || throw(ArgumentError("node $node is outside 1:$(tree.nnodes)"))
    positions = map.descendant_tip_positions_by_node[inode]
    isempty(positions) && throw(ArgumentError("node $node has no descendant tips"))
    return tree.tip_labels[Int(first(positions))]
end

"""
    phylo_node_anchor(tree, node; map = build_phylomap(tree))

Return a stable two-tip anchor for `node`. If `node` is a tip, the result is
`(tip, tip)`. Otherwise the result is the lexicographically ordered pair of the
first descendant tips under the first two child lineages.
"""
function phylo_node_anchor(
    tree::CompactTree,
    node::Integer;
    map::PhyloMap = build_phylomap(tree),
)
    inode = Int(node)
    1 <= inode <= tree.nnodes || throw(ArgumentError("node $node is outside 1:$(tree.nnodes)"))
    if tree.is_tip[inode]
        tip = tree.node_labels[inode]
        return tip, tip
    end
    length(tree.children[inode]) >= 2 || throw(ArgumentError("internal node $node has fewer than two child lineages"))
    left = _phylomap_first_tip_label(tree, map, tree.children[inode][1])
    right = _phylomap_first_tip_label(tree, map, tree.children[inode][2])
    return left <= right ? (left, right) : (right, left)
end

"""
    phylo_branch_anchor(tree, edge; map = build_phylomap(tree))

Return the stable two-tip anchor for the child node of `edge`.
"""
function phylo_branch_anchor(
    tree::CompactTree,
    edge::Integer;
    map::PhyloMap = build_phylomap(tree),
)
    e = Int(edge)
    1 <= e <= tree.nedges || throw(ArgumentError("edge $edge is outside 1:$(tree.nedges)"))
    return phylo_node_anchor(tree, tree.child_of_edge[e]; map = map)
end

"""
    phylo_edge_signature(tree, edge; sep = "|", map = build_phylomap(tree))

Return a stable branch identifier as the sorted descendant tip-label set joined
by `sep`.
"""
function phylo_edge_signature(
    tree::CompactTree,
    edge::Integer;
    sep::AbstractString = "|",
    map::PhyloMap = build_phylomap(tree),
)
    e = Int(edge)
    1 <= e <= tree.nedges || throw(ArgumentError("edge $edge is outside 1:$(tree.nedges)"))
    labels = _phylomap_descendant_labels(tree, map.descendant_tip_positions_by_edge[e])
    sort!(labels)
    return join(labels, sep)
end

function phylo_edge_signatures(
    tree::CompactTree,
    edges::AbstractVector{<:Integer};
    sep::AbstractString = "|",
    map::PhyloMap = build_phylomap(tree),
)
    signatures = Vector{String}(undef, length(edges))
    @inbounds for (i, edge) in enumerate(edges)
        signatures[i] = phylo_edge_signature(tree, edge; sep = sep, map = map)
    end
    return signatures
end

function _normalize_phylo_edge_signature(sig::AbstractString; sep::AbstractString = "|")
    labels = split(String(sig), sep; keepempty = false)
    isempty(labels) && throw(ArgumentError("empty edge signature"))
    sort!(labels)
    return join(labels, sep)
end

function phylo_edges_from_signatures(
    tree::CompactTree,
    signatures::AbstractVector{<:AbstractString};
    sep::AbstractString = "|",
    map::PhyloMap = build_phylomap(tree),
)
    index = Dict{String, Int}()
    @inbounds for edge in 1:tree.nedges
        sig = phylo_edge_signature(tree, edge; sep = sep, map = map)
        haskey(index, sig) && throw(ArgumentError("edge signature is not unique: $sig"))
        index[sig] = edge
    end
    edges = Vector{Int}(undef, length(signatures))
    @inbounds for (i, sig0) in enumerate(signatures)
        sig = _normalize_phylo_edge_signature(sig0; sep = sep)
        edge = get(index, sig, 0)
        edge == 0 && throw(ArgumentError("no edge in tree has descendant-tip signature: $sig0"))
        edges[i] = edge
    end
    return edges
end

"""
    R_node_id(tree, evotraits_node_id; map = build_phylomap(tree))

Return the R/ape-style node id for an EvoTraits internal node id.
"""
function R_node_id(
    tree::CompactTree,
    evotraits_node_id::Integer;
    map::PhyloMap = build_phylomap(tree),
)
    node = Int(evotraits_node_id)
    1 <= node <= tree.nnodes || throw(ArgumentError("node $evotraits_node_id is outside 1:$(tree.nnodes)"))
    return map.ape_node_ids[node]
end

"""
    evotraits_node_id(tree, R_node_id; map = build_phylomap(tree))

Return the EvoTraits node id for an R/ape-style node id.
"""
function evotraits_node_id(
    tree::CompactTree,
    R_node_id_value::Integer;
    map::PhyloMap = build_phylomap(tree),
)
    rid = Int(R_node_id_value)
    1 <= rid <= tree.nnodes || throw(ArgumentError("R node id $R_node_id_value is outside 1:$(tree.nnodes)"))
    node = map.evotraits_node_ids_from_ape[rid]
    node != 0 || throw(ArgumentError("R node id $R_node_id_value has no EvoTraits mapping"))
    return node
end

function R_node_ids(
    tree::CompactTree,
    evotraits_node_ids::AbstractVector{<:Integer};
    map::PhyloMap = build_phylomap(tree),
)
    out = Vector{Int}(undef, length(evotraits_node_ids))
    @inbounds for (i, node) in enumerate(evotraits_node_ids)
        out[i] = R_node_id(tree, node; map = map)
    end
    return out
end

function evotraits_node_ids(
    tree::CompactTree,
    R_node_ids_values::AbstractVector{<:Integer};
    map::PhyloMap = build_phylomap(tree),
)
    out = Vector{Int}(undef, length(R_node_ids_values))
    @inbounds for (i, rid) in enumerate(R_node_ids_values)
        out[i] = evotraits_node_id(tree, rid; map = map)
    end
    return out
end

"""
    R_edge_id_cladewise(tree, evotraits_edge_id; map = build_phylomap(tree))

Return the R/ape cladewise edge rank for an EvoTraits edge id.
"""
function R_edge_id_cladewise(
    tree::CompactTree,
    evotraits_edge_id::Integer;
    map::PhyloMap = build_phylomap(tree),
)
    edge = Int(evotraits_edge_id)
    1 <= edge <= tree.nedges || throw(ArgumentError("edge $evotraits_edge_id is outside 1:$(tree.nedges)"))
    return map.ape_cladewise_edge_ranks[edge]
end

"""
    R_edge_id_postorder(tree, evotraits_edge_id; map = build_phylomap(tree))

Return the R/ape postorder edge rank for an EvoTraits edge id.
"""
function R_edge_id_postorder(
    tree::CompactTree,
    evotraits_edge_id::Integer;
    map::PhyloMap = build_phylomap(tree),
)
    edge = Int(evotraits_edge_id)
    1 <= edge <= tree.nedges || throw(ArgumentError("edge $evotraits_edge_id is outside 1:$(tree.nedges)"))
    return map.ape_postorder_edge_ranks[edge]
end

"""
    evotraits_edge_id_from_R_cladewise(tree, R_edge_id; map = build_phylomap(tree))

Return the EvoTraits edge id for an R/ape cladewise edge rank.
"""
function evotraits_edge_id_from_R_cladewise(
    tree::CompactTree,
    R_edge_id_value::Integer;
    map::PhyloMap = build_phylomap(tree),
)
    rid = Int(R_edge_id_value)
    1 <= rid <= tree.nedges || throw(ArgumentError("R cladewise edge id $R_edge_id_value is outside 1:$(tree.nedges)"))
    return map.evotraits_edge_ids_from_ape_cladewise[rid]
end

function R_edge_ids_cladewise(
    tree::CompactTree,
    evotraits_edge_ids::AbstractVector{<:Integer};
    map::PhyloMap = build_phylomap(tree),
)
    out = Vector{Int}(undef, length(evotraits_edge_ids))
    @inbounds for (i, edge) in enumerate(evotraits_edge_ids)
        out[i] = R_edge_id_cladewise(tree, edge; map = map)
    end
    return out
end

function evotraits_edge_ids_from_R_cladewise(
    tree::CompactTree,
    R_edge_ids_values::AbstractVector{<:Integer};
    map::PhyloMap = build_phylomap(tree),
)
    out = Vector{Int}(undef, length(R_edge_ids_values))
    @inbounds for (i, rid) in enumerate(R_edge_ids_values)
        out[i] = evotraits_edge_id_from_R_cladewise(tree, rid; map = map)
    end
    return out
end

"""
    evotraits_edge_id_from_R_postorder(tree, R_edge_id; map = build_phylomap(tree))

Return the EvoTraits edge id for an R/ape postorder edge rank.
"""
function evotraits_edge_id_from_R_postorder(
    tree::CompactTree,
    R_edge_id_value::Integer;
    map::PhyloMap = build_phylomap(tree),
)
    rid = Int(R_edge_id_value)
    1 <= rid <= tree.nedges || throw(ArgumentError("R postorder edge id $R_edge_id_value is outside 1:$(tree.nedges)"))
    return map.evotraits_edge_ids_from_ape_postorder[rid]
end

function R_edge_ids_postorder(
    tree::CompactTree,
    evotraits_edge_ids::AbstractVector{<:Integer};
    map::PhyloMap = build_phylomap(tree),
)
    out = Vector{Int}(undef, length(evotraits_edge_ids))
    @inbounds for (i, edge) in enumerate(evotraits_edge_ids)
        out[i] = R_edge_id_postorder(tree, edge; map = map)
    end
    return out
end

function evotraits_edge_ids_from_R_postorder(
    tree::CompactTree,
    R_edge_ids_values::AbstractVector{<:Integer};
    map::PhyloMap = build_phylomap(tree),
)
    out = Vector{Int}(undef, length(R_edge_ids_values))
    @inbounds for (i, rid) in enumerate(R_edge_ids_values)
        out[i] = evotraits_edge_id_from_R_postorder(tree, rid; map = map)
    end
    return out
end

"""
    R_edge_matrix(tree; order = :cladewise, map = build_phylomap(tree))

Return an integer matrix with the same node-numbering convention as `ape::phylo\$edge`.
Supported orders are `:cladewise` and `:postorder`.
"""
function R_edge_matrix(
    tree::CompactTree;
    order::Symbol = :cladewise,
    map::PhyloMap = build_phylomap(tree),
)
    ranks =
        order === :cladewise ? map.ape_cladewise_edge_ranks :
        order === :postorder ? map.ape_postorder_edge_ranks :
        throw(ArgumentError("Unsupported order=$order; expected :cladewise or :postorder"))
    edge_ids = sortperm(ranks)
    M = Matrix{Int}(undef, tree.nedges, 2)
    @inbounds for (i, edge) in enumerate(edge_ids)
        M[i, 1] = map.ape_node_ids[Int(tree.parent_of_edge[edge])]
        M[i, 2] = map.ape_node_ids[Int(tree.child_of_edge[edge])]
    end
    return M
end
