"""
    attach_R_node_map(df, tree; node_col = :node_id, map = build_phylomap(tree), prefix = "R_")

Return a copy of `df` with R-facing node translation columns attached for the
EvoTraits node ids stored in `node_col`.
"""
function attach_R_node_map(
    df::AbstractDataFrame,
    tree::CompactTree;
    node_col::Union{Symbol, AbstractString} = :node_id,
    map::PhyloMap = build_phylomap(tree),
    prefix::AbstractString = "R_",
)
    col = Symbol(node_col)
    col in propertynames(df) || throw(ArgumentError("node_col=$node_col is not present in the DataFrame"))
    out = DataFrame(df)
    nodes = Int.(out[!, col])
    attached = R_node_table(tree; map = map)
    lookup = Dict(Int(row.evotraits_node_id) => row for row in eachrow(attached))

    R_node_vals = Vector{Int}(undef, length(nodes))
    is_tip_vals = Vector{Bool}(undef, length(nodes))
    label_vals = Vector{String}(undef, length(nodes))
    tipX_vals = Vector{String}(undef, length(nodes))
    tipY_vals = Vector{String}(undef, length(nodes))
    @inbounds for (i, node) in enumerate(nodes)
        row = get(lookup, node, nothing)
        row === nothing && throw(ArgumentError("node id $node from column $node_col has no mapping in this tree"))
        R_node_vals[i] = row.R_node_id
        is_tip_vals[i] = row.is_tip
        label_vals[i] = row.label
        tipX_vals[i] = row.tipX
        tipY_vals[i] = row.tipY
    end

    out[!, Symbol(prefix, "node_id")] = R_node_vals
    out[!, Symbol(prefix, "is_tip")] = is_tip_vals
    out[!, Symbol(prefix, "label")] = label_vals
    out[!, Symbol(prefix, "tipX")] = tipX_vals
    out[!, Symbol(prefix, "tipY")] = tipY_vals
    return out
end

"""
    attach_R_edge_map(df, tree; edge_col = :edge_id, order = :postorder, map = build_phylomap(tree), prefix = "R_")

Return a copy of `df` with R-facing edge translation columns attached for the
EvoTraits edge ids stored in `edge_col`.
"""
function attach_R_edge_map(
    df::AbstractDataFrame,
    tree::CompactTree;
    edge_col::Union{Symbol, AbstractString} = :edge_id,
    order::Symbol = :postorder,
    map::PhyloMap = build_phylomap(tree),
    prefix::AbstractString = "R_",
)
    col = Symbol(edge_col)
    col in propertynames(df) || throw(ArgumentError("edge_col=$edge_col is not present in the DataFrame"))
    out = DataFrame(df)
    edges = Int.(out[!, col])
    attached = R_edge_table(tree; order = order, map = map)
    lookup = Dict(Int(row.evotraits_edge_id) => row for row in eachrow(attached))

    R_edge_vals = Vector{Int}(undef, length(edges))
    R_parent_vals = Vector{Int}(undef, length(edges))
    R_child_vals = Vector{Int}(undef, length(edges))
    branch_length_vals = Vector{Float64}(undef, length(edges))
    tipX_vals = Vector{String}(undef, length(edges))
    tipY_vals = Vector{String}(undef, length(edges))
    sig_vals = Vector{String}(undef, length(edges))
    @inbounds for (i, edge) in enumerate(edges)
        row = get(lookup, edge, nothing)
        row === nothing && throw(ArgumentError("edge id $edge from column $edge_col has no mapping in this tree"))
        R_edge_vals[i] = row.R_edge_id
        R_parent_vals[i] = row.R_parent_node_id
        R_child_vals[i] = row.R_child_node_id
        branch_length_vals[i] = row.branch_length
        tipX_vals[i] = row.tipX
        tipY_vals[i] = row.tipY
        sig_vals[i] = row.descendant_signature
    end

    out[!, Symbol(prefix, "edge_id")] = R_edge_vals
    out[!, Symbol(prefix, "parent_node_id")] = R_parent_vals
    out[!, Symbol(prefix, "child_node_id")] = R_child_vals
    out[!, :branch_length] = branch_length_vals
    out[!, :tipX] = tipX_vals
    out[!, :tipY] = tipY_vals
    out[!, :descendant_signature] = sig_vals
    return out
end
