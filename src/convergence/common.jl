@inline _shift_edges_key(shift_edges::AbstractVector{<:Integer}) = Tuple(Int.(shift_edges))

function _convergent_criterion(criterion)
    sym =
        criterion isa Symbol ? criterion :
        criterion isa AbstractString ? Symbol(criterion) :
        throw(ArgumentError("criterion must be :AICc or :BIC"))
    sym in (:AICc, :BIC) || throw(ArgumentError("merge_convergent_regimes supports only :AICc or :BIC"))
    return sym
end

function _shift_detection_context(det::OUShiftDetectionResult)
    haskey(det.diagnostics, :source_tree) ||
        throw(ArgumentError("det does not contain source tree; call detect_ou_shifts first and pass its result directly"))
    haskey(det.diagnostics, :source_trait) ||
        throw(ArgumentError("det does not contain source trait; call detect_ou_shifts first and pass its result directly"))
    return det.diagnostics.source_tree, det.diagnostics.source_trait
end

@inline function _without_edge!(
    trial::Vector{Int},
    edges::AbstractVector{<:Integer},
    edge::Integer,
)
    empty!(trial)
    sizehint!(trial, max(length(edges) - 1, 0))
    @inbounds for e in edges
        Int(e) == Int(edge) && continue
        push!(trial, Int(e))
    end
    return trial
end

function _connected_components(edges::Vector{Tuple{Int,Int}}, vertices::Vector{Int})
    adj = Dict{Int,Vector{Int}}()
    for v in vertices
        adj[v] = Int[]
    end
    for (u, v) in edges
        u in vertices && v in vertices || continue
        push!(adj[u], v)
        push!(adj[v], u)
    end
    visited = Set{Int}()
    components = Vector{Int}[]
    for v in vertices
        v in visited && continue
        comp = Int[]
        stack = Int[v]
        while !isempty(stack)
            cur = pop!(stack)
            cur in visited && continue
            push!(visited, cur)
            push!(comp, cur)
            for nb in adj[cur]
                nb in visited || push!(stack, nb)
            end
        end
        push!(components, comp)
    end
    return components
end

function _convergent_vertex_index(vertices::Vector{Int})
    maxv = maximum(vertices)
    index = zeros(Int, maxv + 1)
    @inbounds for (i, v) in enumerate(vertices)
        index[v + 1] = i
    end
    return index
end

function _uf_find!(parent::Vector{Int}, x::Int)
    y = x
    @inbounds while parent[y] != y
        y = parent[y]
    end
    root = y
    y = x
    @inbounds while parent[y] != y
        nxt = parent[y]
        parent[y] = root
        y = nxt
    end
    return root
end

function _uf_union!(
    parent::Vector{Int},
    rank::Vector{UInt8},
    ia::Int,
    ib::Int,
)
    (ia == 0 || ib == 0) && return
    ra = _uf_find!(parent, ia)
    rb = _uf_find!(parent, ib)
    ra == rb && return
    @inbounds if rank[ra] < rank[rb]
        parent[ra] = rb
    elseif rank[ra] > rank[rb]
        parent[rb] = ra
    else
        parent[rb] = ra
        rank[ra] += UInt8(1)
    end
    return
end

function _connected_components_indexed(
    edges::Vector{Tuple{Int,Int}},
    vertices::Vector{Int},
    vertex_index::Vector{Int},
    parent::Vector{Int},
    rank::Vector{UInt8};
    root_slot::Union{Nothing, Vector{Int}} = nothing,
    extra::Union{Nothing, Tuple{Int,Int}} = nothing,
    skip_index::Int = 0,
)
    nv = length(vertices)
    @inbounds for i in 1:nv
        parent[i] = i
        rank[i] = UInt8(0)
    end
    @inbounds for i in eachindex(edges)
        i == skip_index && continue
        u, v = edges[i]
        iu = vertex_index[u + 1]
        iv = vertex_index[v + 1]
        _uf_union!(parent, rank, iu, iv)
    end
    if extra !== nothing
        u, v = extra
        _uf_union!(parent, rank, vertex_index[u + 1], vertex_index[v + 1])
    end

    slots = root_slot === nothing ? zeros(Int, nv) : root_slot
    fill!(slots, 0)
    components = Vector{Int}[]
    @inbounds for (i, v) in enumerate(vertices)
        root = _uf_find!(parent, i)
        slot = slots[root]
        if slot == 0
            push!(components, Int[])
            slot = length(components)
            slots[root] = slot
        end
        push!(components[slot], v)
    end
    return components
end

function _l1ou_convergent_name_vector(
    cache::OUShiftTreeCache,
    shift_edges::AbstractVector{<:Integer},
    components::Vector{Vector{Int}},
)
    ordered_edges = Int.(shift_edges)
    if !isempty(cache.r_postorder_edge_rank)
        sort!(ordered_edges; by = e -> cache.r_postorder_edge_rank[e])
    else
        sort!(ordered_edges)
    end

    names = fill(typemin(Int), length(ordered_edges))
    for comp in components
        shifts = Int[item for item in comp if item != 0]
        isempty(shifts) && continue
        label =
            isempty(cache.r_postorder_edge_rank) ?
            minimum(shifts) :
            minimum(cache.r_postorder_edge_rank[e] for e in shifts)
        @inbounds for (i, edge) in enumerate(ordered_edges)
            edge in shifts && (names[i] = label)
        end
    end
    return names
end

function _l1ou_convergent_order_position(
    cache::OUShiftTreeCache,
    shift_edges::AbstractVector{<:Integer},
)
    ordered_edges = Int.(shift_edges)
    if !isempty(cache.r_postorder_edge_rank)
        sort!(ordered_edges; by = e -> cache.r_postorder_edge_rank[e])
    else
        sort!(ordered_edges)
    end
    max_edge = isempty(ordered_edges) ? 0 : maximum(ordered_edges)
    pos_by_edge = zeros(Int, max_edge)
    @inbounds for (i, edge) in enumerate(ordered_edges)
        pos_by_edge[edge] = i
    end
    return pos_by_edge
end

function _l1ou_convergent_name_vector!(
    names::Vector{Int},
    cache::OUShiftTreeCache,
    components::Vector{Vector{Int}},
    pos_by_edge::Vector{Int},
)
    fill!(names, typemin(Int))
    has_ranks = !isempty(cache.r_postorder_edge_rank)
    for comp in components
        label = typemax(Int)
        has_shift = false
        for item in comp
            item == 0 && continue
            has_shift = true
            value = has_ranks ? cache.r_postorder_edge_rank[item] : item
            value < label && (label = value)
        end
        has_shift || continue
        @inbounds for item in comp
            item == 0 && continue
            1 <= item <= length(pos_by_edge) || continue
            pos = pos_by_edge[item]
            pos == 0 || (names[pos] = label)
        end
    end
    return names
end

function _extract_edge_regimes(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}})
    regimes = Int[]
    for e in 1:tree.nedges
        push!(regimes, Int(edge_segments[e][1].state))
    end
    return regimes
end

function _merge_map_from_edge_groups(
    tree::CompactTree,
    shift_edges::AbstractVector{<:Integer},
    edge_groups::AbstractVector,
)
    shift_set = Set(Int.(shift_edges))
    edge_segments = shift_edges_to_edge_segments(tree, shift_edges)
    merge_map = Dict{Int,Int}()
    for group in edge_groups
        states = Int[]
        for edge0 in group
            edge = Int(edge0)
            edge in shift_set || throw(ArgumentError("convergent regime edge $edge is not in shift_edges"))
            state = Int(edge_segments[edge][1].state)
            state in states || push!(states, state)
        end
        isempty(states) && continue
        canonical = minimum(states)
        for state in states
            state == canonical || (merge_map[state] = canonical)
        end
    end
    return merge_map
end

function _merge_map_from_regime_components(
    tree::CompactTree,
    shift_edges::AbstractVector{<:Integer},
    components::Vector{Vector{Int}},
)
    shift_set = Set(Int.(shift_edges))
    edge_segments = shift_edges_to_edge_segments(tree, shift_edges)
    merge_map = Dict{Int,Int}()
    for comp in components
        states = Int[]
        has_background = 0 in comp
        for item in comp
            item == 0 && continue
            item in shift_set || throw(ArgumentError("convergent component contains non-shift edge $item"))
            state = Int(edge_segments[item][1].state)
            state in states || push!(states, state)
        end
        isempty(states) && continue
        canonical = has_background ? 1 : minimum(states)
        for state in states
            state == canonical || (merge_map[state] = canonical)
        end
    end
    return merge_map
end

function _shift_state_by_edge(
    tree::CompactTree,
    shift_edges::AbstractVector{<:Integer},
)
    edge_segments = shift_edges_to_edge_segments(tree, shift_edges)
    state_by_edge = zeros(Int, tree.nedges)
    @inbounds for edge in shift_edges
        state_by_edge[Int(edge)] = Int(edge_segments[Int(edge)][1].state)
    end
    return state_by_edge
end

function _merge_map_from_regime_components(
    shift_edges::AbstractVector{<:Integer},
    components::Vector{Vector{Int}},
    state_by_edge::Vector{Int},
)
    merge_map = Dict{Int,Int}()
    for comp in components
        states = Int[]
        has_background = 0 in comp
        for item in comp
            item == 0 && continue
            state = 1 <= item <= length(state_by_edge) ? state_by_edge[item] : 0
            state == 0 && throw(ArgumentError("convergent component contains non-shift edge $item"))
            state in states || push!(states, state)
        end
        isempty(states) && continue
        canonical = has_background ? 1 : minimum(states)
        for state in states
            state == canonical || (merge_map[state] = canonical)
        end
    end
    return merge_map
end

