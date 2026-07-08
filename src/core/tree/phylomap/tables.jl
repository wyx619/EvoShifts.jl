"""
    phylomap_node_table(tree; map = build_phylomap(tree))

Return a DataFrame translating EvoTraits internal node ids to the `ape` node
numbering convention where tips are `1:ntips` and internal nodes are numbered
after tips in cladewise / preorder-internal order.
"""
function phylomap_node_table(
    tree::CompactTree;
    map::PhyloMap = build_phylomap(tree),
)
    n = tree.nnodes
    ape_node_id = Vector{Int}(undef, n)
    is_tip = Vector{Bool}(undef, n)
    label = Vector{String}(undef, n)
    tipX = Vector{String}(undef, n)
    tipY = Vector{String}(undef, n)

    @inbounds for node in 1:n
        ape_node_id[node] = map.ape_node_ids[node]
        is_tip[node] = tree.is_tip[node]
        label[node] = tree.node_labels[node]
        tipX[node], tipY[node] = phylo_node_anchor(tree, node; map = map)
    end

    return DataFrame(
        evotraits_node_id = collect(1:n),
        ape_node_id = ape_node_id,
        is_tip = is_tip,
        label = label,
        tipX = tipX,
        tipY = tipY,
    )
end

"""
    phylomap_edge_table(tree; edges = nothing, map = build_phylomap(tree))

Return a DataFrame translating EvoTraits internal edge ids to branch identities
expressed through parent/child `ape` node ids, `ape` edge ranks in cladewise
and postorder orderings, two-tip anchors, and descendant-tip signatures.
"""
function phylomap_edge_table(
    tree::CompactTree;
    edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    map::PhyloMap = build_phylomap(tree),
)
    edge_ids = edges === nothing ? collect(1:tree.nedges) : Int.(edges)
    n = length(edge_ids)
    parent_node_id = Vector{Int}(undef, n)
    child_node_id = Vector{Int}(undef, n)
    ape_parent_node_id = Vector{Int}(undef, n)
    ape_child_node_id = Vector{Int}(undef, n)
    ape_cladewise_edge_rank = Vector{Int}(undef, n)
    ape_postorder_edge_rank = Vector{Int}(undef, n)
    branch_length = Vector{Float64}(undef, n)
    tipX = Vector{String}(undef, n)
    tipY = Vector{String}(undef, n)
    descendant_signature = Vector{String}(undef, n)

    @inbounds for (i, edge) in enumerate(edge_ids)
        1 <= edge <= tree.nedges || throw(ArgumentError("edge $edge is outside 1:$(tree.nedges)"))
        parent = Int(tree.parent_of_edge[edge])
        child = Int(tree.child_of_edge[edge])
        parent_node_id[i] = parent
        child_node_id[i] = child
        ape_parent_node_id[i] = map.ape_node_ids[parent]
        ape_child_node_id[i] = map.ape_node_ids[child]
        ape_cladewise_edge_rank[i] = map.ape_cladewise_edge_ranks[edge]
        ape_postorder_edge_rank[i] = map.ape_postorder_edge_ranks[edge]
        branch_length[i] = tree.edge_length[edge]
        tipX[i], tipY[i] = phylo_branch_anchor(tree, edge; map = map)
        descendant_signature[i] = phylo_edge_signature(tree, edge; map = map)
    end

    return DataFrame(
        evotraits_edge_id = edge_ids,
        evotraits_parent_node_id = parent_node_id,
        evotraits_child_node_id = child_node_id,
        ape_parent_node_id = ape_parent_node_id,
        ape_child_node_id = ape_child_node_id,
        ape_cladewise_edge_rank = ape_cladewise_edge_rank,
        ape_postorder_edge_rank = ape_postorder_edge_rank,
        branch_length = branch_length,
        tipX = tipX,
        tipY = tipY,
        descendant_signature = descendant_signature,
    )
end

"""
    R_node_table(tree; map = build_phylomap(tree))

Return `phylomap_node_table(tree)` with column names centered on the R-facing
translation use case.
"""
function R_node_table(
    tree::CompactTree;
    map::PhyloMap = build_phylomap(tree),
)
    tbl = phylomap_node_table(tree; map = map)
    return DataFrame(
        R_node_id = tbl.ape_node_id,
        evotraits_node_id = tbl.evotraits_node_id,
        is_tip = tbl.is_tip,
        label = tbl.label,
        tipX = tbl.tipX,
        tipY = tbl.tipY,
    )
end

"""
    R_edge_table(tree; edges = nothing, order = :cladewise, map = build_phylomap(tree))

Return a branch translation table oriented around R/ape edge identities. The
`R_edge_id` column follows the requested `order`.
"""
function R_edge_table(
    tree::CompactTree;
    edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    order::Symbol = :cladewise,
    map::PhyloMap = build_phylomap(tree),
)
    tbl = phylomap_edge_table(tree; edges = edges, map = map)
    R_edge_id =
        order === :cladewise ? tbl.ape_cladewise_edge_rank :
        order === :postorder ? tbl.ape_postorder_edge_rank :
        throw(ArgumentError("Unsupported order=$order; expected :cladewise or :postorder"))
    return DataFrame(
        R_edge_id = R_edge_id,
        evotraits_edge_id = tbl.evotraits_edge_id,
        R_parent_node_id = tbl.ape_parent_node_id,
        R_child_node_id = tbl.ape_child_node_id,
        evotraits_parent_node_id = tbl.evotraits_parent_node_id,
        evotraits_child_node_id = tbl.evotraits_child_node_id,
        branch_length = tbl.branch_length,
        tipX = tbl.tipX,
        tipY = tbl.tipY,
        descendant_signature = tbl.descendant_signature,
    )
end
