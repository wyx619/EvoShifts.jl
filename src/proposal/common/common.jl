@inline function _soft_threshold(z::Float64, lambda::Float64)
    if z > lambda
        return z - lambda
    elseif z < -lambda
        return z + lambda
    end
    return 0.0
end

function _intercept_projection_vector(intercept_w::AbstractVector{<:Real})
    iw = Float64.(intercept_w)
    denom = LinearAlgebra.dot(iw, iw)
    return iw, max(denom, eps(Float64))
end

function _remove_intercept_component!(
    v::AbstractVector{<:Real},
    intercept_w::AbstractVector{<:Real},
    intercept_norm2::Float64,
)
    coef = LinearAlgebra.dot(intercept_w, v) / intercept_norm2
    LinearAlgebra.axpy!(-coef, intercept_w, v)
    return v
end

struct _ScreeningState
    mask::UInt64
    last::Int
    sum::Int
end

@inline _screening_state(pos::Integer) = _ScreeningState(UInt64(1) << (Int(pos) - 1), Int(pos), Int(pos))

@inline function _extend_screening_state(state::_ScreeningState, pos::Integer)
    p = Int(pos)
    return _ScreeningState(state.mask | (UInt64(1) << (p - 1)), p, state.sum + p)
end

@inline function _ndigits10(x::Int)
    n = x
    d = 1
    while n >= 10
        n ÷= 10
        d += 1
    end
    return d
end

@inline function _pow10_for_digits(d::Int)
    p = 1
    for _ in 2:d
        p *= 10
    end
    return p
end

function _decimal_string_less(x::Int, y::Int)
    x == y && return false
    dx = _ndigits10(x)
    dy = _ndigits10(y)
    px = _pow10_for_digits(dx)
    py = _pow10_for_digits(dy)
    while px > 0 && py > 0
        xd = (x ÷ px) % 10
        yd = (y ÷ py) % 10
        xd != yd && return xd < yd
        px ÷= 10
        py ÷= 10
    end
    return dx < dy
end

function _screening_state_join_less(a::_ScreeningState, b::_ScreeningState)
    ma = a.mask
    mb = b.mask
    while ma != 0 && mb != 0
        pa = trailing_zeros(ma) + 1
        pb = trailing_zeros(mb) + 1
        pa != pb && return _decimal_string_less(pa, pb)
        ma &= ma - one(UInt64)
        mb &= mb - one(UInt64)
    end
    return ma == 0 && mb != 0
end

function _screening_state_less(a::_ScreeningState, b::_ScreeningState)
    a.sum != b.sum && return a.sum < b.sum
    a.last != b.last && return a.last < b.last
    return _screening_state_join_less(a, b)
end

function _state_edges!(
    out::Vector{Int},
    pool::AbstractVector{<:Integer},
    state::_ScreeningState,
)
    empty!(out)
    mask = state.mask
    while mask != 0
        pos = trailing_zeros(mask) + 1
        push!(out, Int(pool[pos]))
        mask &= mask - one(UInt64)
    end
    return out
end

function _push_screening_candidate!(
    configs::Vector{Vector{Int}},
    seen::Set{Vector{Int}},
    cache::OUShiftTreeCache,
    raw_edges::AbstractVector{<:Integer};
    max_shifts::Integer = typemax(Int),
)
    corrected = correct_shift_configuration_l1ou(cache, raw_edges)
    corrected = _order_shift_edges_like(corrected, raw_edges)
    isempty(corrected) && return false
    length(corrected) > max_shifts && return false
    key = sort!(copy(corrected))
    key in seen && return false
    push!(seen, key)
    push!(configs, corrected)
    return true
end

function _screening_candidate_limits(nranked::Integer, max_shifts::Integer)
    maxs = max(Int(max_shifts), 0)
    maxs == 0 && return (pool_width = 0, beam_width = 0, max_configs = 0)
    pool_width = min(Int(nranked), max(12, min(64, 4 * maxs)))
    beam_width = max(24, min(128, 8 * maxs))
    max_configs = max(64, min(2000, 24 * maxs))
    return (pool_width = pool_width, beam_width = beam_width, max_configs = max_configs)
end

function _valid_screening_ranked_edges(
    cache::OUShiftTreeCache,
    ranked_edges::AbstractVector{<:Integer},
)
    edges = Int[]
    seen = Set{Int}()
    sizehint!(edges, length(ranked_edges))
    @inbounds for edge0 in ranked_edges
        edge = Int(edge0)
        1 <= edge <= cache.nedges || continue
        Int(cache.edge_parent[edge]) == Int(cache.root) && continue
        edge in seen && continue
        push!(seen, edge)
        push!(edges, edge)
    end
    return edges
end

function _build_screening_candidate_family(
    cache::OUShiftTreeCache,
    ranked_edges::AbstractVector{<:Integer};
    max_shifts::Integer = typemax(Int),
)
    maxs = Int(max_shifts)
    maxs <= 0 && return Vector{Vector{Int}}()

    ranked = _valid_screening_ranked_edges(cache, ranked_edges)
    isempty(ranked) && return Vector{Vector{Int}}()

    limits = _screening_candidate_limits(length(ranked), maxs)
    pool = ranked[1:limits.pool_width]
    max_depth = min(maxs, length(pool))

    configs = Vector{Vector{Int}}()
    seen_configs = Set{Vector{Int}}()

    raw_prefix = Int[]
    sizehint!(raw_prefix, min(maxs, length(ranked)))
    largest_prefix_size = 0
    @inbounds for edge in ranked
        push!(raw_prefix, edge)
        if _push_screening_candidate!(configs, seen_configs, cache, raw_prefix; max_shifts = maxs)
            largest_prefix_size = max(largest_prefix_size, length(configs[end]))
        end
        (largest_prefix_size >= maxs || length(configs) >= limits.max_configs) && break
    end
    length(configs) >= limits.max_configs && return configs

    raw_state = Int[]
    sizehint!(raw_state, max_depth)

    states = Vector{_ScreeningState}()
    sizehint!(states, length(pool))
    for pos in 1:length(pool)
        state = _screening_state(pos)
        push!(states, state)
        _state_edges!(raw_state, pool, state)
        _push_screening_candidate!(configs, seen_configs, cache, raw_state; max_shifts = maxs)
        length(configs) >= limits.max_configs && return configs
    end

    for depth in 2:max_depth
        proposals = Vector{_ScreeningState}()
        sizehint!(proposals, limits.beam_width * 2)
        for state in states
            start_pos = state.last + 1
            start_pos > length(pool) && continue
            for pos in start_pos:length(pool)
                push!(proposals, _extend_screening_state(state, pos))
            end
        end
        isempty(proposals) && break
        sort!(proposals; lt = _screening_state_less)
        length(proposals) > limits.beam_width && resize!(proposals, limits.beam_width)

        next_states = Vector{_ScreeningState}()
        sizehint!(next_states, length(proposals))
        for state in proposals
            _state_edges!(raw_state, pool, state)
            _push_screening_candidate!(configs, seen_configs, cache, raw_state; max_shifts = maxs)
            push!(next_states, state)
            length(configs) >= limits.max_configs && return configs
        end
        states = next_states
    end

    return configs
end

function _build_screening_prefix_anchor_family(
    cache::OUShiftTreeCache,
    ranked_edges::AbstractVector{<:Integer};
    max_shifts::Integer = typemax(Int),
    max_prefix_edges::Integer = typemax(Int),
)
    maxs = Int(max_shifts)
    maxs <= 0 && return Vector{Vector{Int}}()

    ranked = _valid_screening_ranked_edges(cache, ranked_edges)
    isempty(ranked) && return Vector{Vector{Int}}()

    prefix_limit = min(length(ranked), Int(max_prefix_edges))
    configs = Vector{Vector{Int}}()
    seen_configs = Set{Vector{Int}}()
    raw_prefix = Int[]
    sizehint!(raw_prefix, min(prefix_limit, maxs))

    @inbounds for i in 1:prefix_limit
        push!(raw_prefix, ranked[i])
        corrected = correct_shift_configuration_l1ou(cache, raw_prefix)
        corrected = _order_shift_edges_like(corrected, raw_prefix)
        isempty(corrected) && continue
        length(corrected) > maxs && break
        key = sort!(copy(corrected))
        key in seen_configs && continue
        push!(seen_configs, key)
        push!(configs, corrected)
    end
    return configs
end

