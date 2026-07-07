"""
    shift_edge_signature(tree, edge; sep = "|", cache = build_shift_tree_cache(tree))

Return a stable, parser-independent identifier for an edge: the sorted set of
descendant tip labels joined by `sep`. Internally this delegates to
`phylo_edge_signature`.
"""
function shift_edge_signature(
    tree::CompactTree,
    edge::Integer;
    sep::AbstractString = "|",
    cache::OUShiftTreeCache = build_shift_tree_cache(tree),
)
    e = Int(edge)
    1 <= e <= cache.nedges || throw(ArgumentError("edge $edge is outside 1:$(cache.nedges)"))
    return phylo_edge_signature(tree, e; sep = sep)
end

"""
    shift_edge_signatures(tree, edges; sep = "|", cache = build_shift_tree_cache(tree))

Return `shift_edge_signature(tree, edge)` for each edge in `edges`.
"""
function shift_edge_signatures(
    tree::CompactTree,
    edges::AbstractVector{<:Integer};
    sep::AbstractString = "|",
    cache::OUShiftTreeCache = build_shift_tree_cache(tree),
)
    return String[shift_edge_signature(tree, edge; sep = sep, cache = cache) for edge in edges]
end

"""
    shift_edges_from_signatures(tree, signatures; sep = "|", cache = build_shift_tree_cache(tree))

Map descendant-tip-set signatures back to EvoTraits edge ids for `tree`.
Each signature may list descendant tips in any order as long as it uses `sep`.
"""
function shift_edges_from_signatures(
    tree::CompactTree,
    signatures::AbstractVector{<:AbstractString};
    sep::AbstractString = "|",
    cache::OUShiftTreeCache = build_shift_tree_cache(tree),
)
    return phylo_edges_from_signatures(tree, signatures; sep = sep)
end

function shift_node_anchor(tree::CompactTree, node::Integer)
    return phylo_node_anchor(tree, node)
end

function shift_branch_anchor(tree::CompactTree, edge::Integer)
    return phylo_branch_anchor(tree, edge)
end

"""
    shift_edge_table(tree; edges = nothing, cache = build_shift_tree_cache(tree))

Return a minimal DataFrame describing branches in `tree`.

Each branch is identified by its child node. The child node is represented as
`MRCA(tipX, tipY)`. If the child is itself a tip, then `tipX == tipY`.
"""
function shift_edge_table(
    tree::CompactTree;
    edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    cache::OUShiftTreeCache = build_shift_tree_cache(tree),
)
    edge_ids = edges === nothing ? collect(1:cache.nedges) : Int.(edges)
    all(1 <= edge <= cache.nedges for edge in edge_ids) || throw(ArgumentError("shift_edge_table received an out-of-range edge id"))
    tbl = phylomap_edge_table(tree; edges = edge_ids)
    return DataFrame(
        edge_id = Int.(tbl.evotraits_edge_id),
        tipX = String.(tbl.tipX),
        tipY = String.(tbl.tipY),
    )
end


