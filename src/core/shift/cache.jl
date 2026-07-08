function build_shift_tree_cache(tree::CompactTree)
    nt = tree.ntips
    ne = tree.nedges
    tree_height = maximum(tree.dist_from_root[tree.tip_ids])

    edge_parent = Int.(tree.parent_of_edge)
    edge_child = Int.(tree.child_of_edge)
    edge_length = Float64.(tree.edge_length)
    dist_root = Float64.(tree.dist_from_root)

    tip_order = Int[]
    first_tip = fill(typemax(Int), tree.nnodes)
    last_tip = fill(typemin(Int), tree.nnodes)

    for node in tree.preorder
        if tree.is_tip[node]
            idx = length(tip_order) + 1
            push!(tip_order, Int(node))
            first_tip[node] = idx
            last_tip[node] = idx
        end
    end

    for node in tree.postorder_internal
        ft = typemax(Int)
        lt = typemin(Int)
        for child in tree.children[node]
            ft = min(ft, first_tip[child])
            lt = max(lt, last_tip[child])
        end
        first_tip[node] = ft
        last_tip[node] = lt
    end

    edge_first_tip = Int[first_tip[edge_child[e]] for e in 1:ne]
    edge_last_tip = Int[last_tip[edge_child[e]] for e in 1:ne]

    tip_position_in_tree_order = zeros(Int, tree.nnodes)
    for (i, tip) in enumerate(tree.tip_ids)
        tip_position_in_tree_order[Int(tip)] = i
    end

    descendant_positions = Vector{Int}[]
    sizehint!(descendant_positions, ne)
    for e in 1:ne
        first_pos = edge_first_tip[e]
        last_pos = edge_last_tip[e]
        pos = Vector{Int}(undef, last_pos - first_pos + 1)
        out_idx = 1
        for preorder_tip_pos in first_pos:last_pos
            node = tip_order[preorder_tip_pos]
            pos[out_idx] = tip_position_in_tree_order[node]
            out_idx += 1
        end
        push!(descendant_positions, pos)
    end

    r_node_id = zeros(Int, tree.nnodes)
    @inbounds for (i, tip) in enumerate(tree.tip_ids)
        r_node_id[Int(tip)] = i
    end
    next_internal_id = tree.ntips + 1
    @inbounds for node in tree.preorder
        inode = Int(node)
        if !tree.is_tip[inode]
            r_node_id[inode] = next_internal_id
            next_internal_id += 1
        end
    end
    edge_order = collect(1:ne)
    sort!(edge_order; by = e -> (-r_node_id[edge_parent[e]], e))
    r_postorder_edge_rank = zeros(Int, ne)
    @inbounds for (rank, edge) in enumerate(edge_order)
        r_postorder_edge_rank[edge] = rank
    end

    return OUShiftTreeCache(
        ntips = nt,
        nedges = ne,
        root = tree.root,
        tree_height = tree_height,
        edge_parent = edge_parent,
        edge_child = edge_child,
        edge_length = edge_length,
        dist_from_root = dist_root,
        first_tip = edge_first_tip,
        last_tip = edge_last_tip,
        tip_order = tip_order,
        descendant_tip_positions = descendant_positions,
        postorder = tree.postorder,
        preorder = tree.preorder,
        r_postorder_edge_rank = r_postorder_edge_rank,
    )
end

function filter_candidate_edges(
    cache::OUShiftTreeCache;
    candidate_edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    edge_length_threshold::Float64 = eps(Float64),
    min_descendant_tips::Integer = 1,
    max_descendant_tips::Union{Nothing, Integer} = nothing,
    exclude_root_edges::Bool = true,
    max_candidate_edges::Union{Nothing, Integer} = nothing,
    candidate_sort::Symbol = :descendant_length,
)
    function keep_edge(e::Int)
        exclude_root_edges && cache.edge_parent[e] == cache.root && return false
        cache.edge_length[e] < edge_length_threshold && return false
        ndesc = length(cache.descendant_tip_positions[e])
        ndesc < min_descendant_tips && return false
        max_descendant_tips !== nothing && ndesc > max_descendant_tips && return false
        return true
    end

    if candidate_edges !== nothing
        candidates = sort!(unique(Int.(candidate_edges)))
        filter!(keep_edge, candidates)
        return _limit_candidate_edges(cache, candidates, max_candidate_edges, candidate_sort)
    end

    candidates = Int[]
    for e in 1:cache.nedges
        keep_edge(e) && push!(candidates, e)
    end
    return _limit_candidate_edges(cache, candidates, max_candidate_edges, candidate_sort)
end

function l1ou_default_candidate_edges(cache::OUShiftTreeCache)
    candidates = Int[]
    sizehint!(candidates, max(cache.nedges - 1, 0))
    @inbounds for edge in 1:cache.nedges
        if isempty(cache.r_postorder_edge_rank) || cache.r_postorder_edge_rank[edge] != cache.nedges
            push!(candidates, edge)
        end
    end
    if !isempty(cache.r_postorder_edge_rank)
        sort!(candidates; by = edge -> cache.r_postorder_edge_rank[edge])
    end
    return candidates
end

function sort_edges_l1ou!(cache::OUShiftTreeCache, edges::Vector{Int})
    isempty(cache.r_postorder_edge_rank) ? sort!(edges) : sort!(edges; by = edge -> cache.r_postorder_edge_rank[edge])
    return edges
end

function _limit_candidate_edges(
    cache::OUShiftTreeCache,
    candidates::Vector{Int},
    max_candidate_edges::Union{Nothing, Integer},
    candidate_sort::Symbol,
)
    max_candidate_edges === nothing && return candidates
    maxc = Int(max_candidate_edges)
    maxc >= 1 || throw(ArgumentError("max_candidate_edges must be positive"))
    length(candidates) <= maxc && return candidates
    if candidate_sort === :descendant_length
        sort!(
            candidates;
            by = e -> (
                -log1p(length(cache.descendant_tip_positions[e])) * max(cache.edge_length[e], eps(Float64)),
                e,
            ),
        )
    elseif candidate_sort === :edge_length
        sort!(candidates; by = e -> (-cache.edge_length[e], e))
    elseif candidate_sort === :descendant_tips
        sort!(candidates; by = e -> (-length(cache.descendant_tip_positions[e]), e))
    elseif candidate_sort === :preorder
        sort!(candidates)
    else
        throw(ArgumentError("Unsupported candidate_sort: $candidate_sort"))
    end
    resize!(candidates, maxc)
    sort!(candidates)
    return candidates
end

function correct_shift_configuration(cache::OUShiftTreeCache, shift_edges::AbstractVector{<:Integer})
    edge_by_child = Dict{Int,Int}()
    for edge in Int.(shift_edges)
        cache.edge_parent[edge] == cache.root && continue
        edge_by_child[cache.edge_child[edge]] = edge
    end
    isempty(edge_by_child) && return Int[]

    corrected = Int[]
    covered = falses(cache.ntips)
    ncovered = 0
    for node in cache.postorder
        node == cache.root && continue
        edge = get(edge_by_child, node, 0)
        edge == 0 && continue

        nnew = 0
        @inbounds for tip_pos in cache.descendant_tip_positions[edge]
            if !covered[tip_pos]
                nnew += 1
            end
        end
        nnew == 0 && continue

        push!(corrected, edge)
        @inbounds for tip_pos in cache.descendant_tip_positions[edge]
            if !covered[tip_pos]
                covered[tip_pos] = true
                ncovered += 1
            end
        end
    end

    if ncovered == cache.ntips && !isempty(corrected)
        pop!(corrected)
    end
    return corrected
end

function _order_shift_edges_like(reference::AbstractVector{<:Integer}, order_source::AbstractVector{<:Integer})
    isempty(reference) && return Int[]
    refset = Set{Int}(Int.(reference))
    ordered = Int[]
    sizehint!(ordered, length(reference))
    for edge in Int.(order_source)
        if edge in refset && !(edge in ordered)
            push!(ordered, edge)
        end
    end
    if length(ordered) < length(reference)
        for edge in Int.(reference)
            edge in ordered || push!(ordered, edge)
        end
    end
    return ordered
end

function correct_shift_configuration_ordered(cache::OUShiftTreeCache, shift_edges::AbstractVector{<:Integer})
    corrected = correct_shift_configuration(cache, shift_edges)
    return _order_shift_edges_like(corrected, shift_edges)
end

function correct_shift_configuration_l1ou(cache::OUShiftTreeCache, shift_edges::AbstractVector{<:Integer})
    length(shift_edges) < 2 && return Int.(shift_edges)
    sorted_edges = sort!(unique(Int.(shift_edges)))
    if !isempty(cache.r_postorder_edge_rank)
        sort!(sorted_edges; by = e -> cache.r_postorder_edge_rank[e])
    end
    covered = falses(cache.ntips)
    corrected = Int[]
    sizehint!(corrected, length(sorted_edges))
    identifiable = true
    for edge in sorted_edges
        nnew = 0
        @inbounds for tip_pos in cache.descendant_tip_positions[edge]
            covered[tip_pos] || (nnew += 1)
        end
        if nnew == 0
            identifiable = false
            continue
        end
        push!(corrected, edge)
        @inbounds for tip_pos in cache.descendant_tip_positions[edge]
            covered[tip_pos] = true
        end
    end
    identifiable && return corrected

    while length(corrected) > 1
        fill!(covered, false)
        @inbounds for edge in corrected
            for tip_pos in cache.descendant_tip_positions[edge]
                covered[tip_pos] = true
            end
        end
        if all(covered)
            if isempty(cache.r_postorder_edge_rank)
                deleteat!(corrected, argmax(corrected))
            else
                imax = 1
                maxrank = cache.r_postorder_edge_rank[corrected[1]]
                @inbounds for i in 2:length(corrected)
                    rank = cache.r_postorder_edge_rank[corrected[i]]
                    if rank > maxrank
                        maxrank = rank
                        imax = i
                    end
                end
                deleteat!(corrected, imax)
            end
        else
            break
        end
    end
    return corrected
end

function shift_edges_to_edge_segments(tree::CompactTree, shift_edges::AbstractVector{<:Integer})
    ne = tree.nedges
    shift_set = Set{Int32}(Int32.(shift_edges))
    edge_regime = fill(Int32(0), ne)
    node_regime = fill(Int32(0), tree.nnodes)
    root = Int(tree.root)
    node_regime[root] = Int32(1)
    regime_counter = Int32(1)

    for node in tree.preorder
        tree.is_tip[node] && continue
        for e in Int(tree.first_child_edge[node]):Int(tree.last_child_edge[node])
            edge_regime[e] != Int32(0) && continue
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

    edge_segments = Vector{Vector{SimmapSegment}}(undef, ne)
    for e in 1:ne
        edge_segments[e] = [SimmapSegment(state = edge_regime[e], length = tree.edge_length[e])]
    end
    return edge_segments
end

