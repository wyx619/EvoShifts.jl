const _OU_SHIFT_GLOBAL_DF = 2

function _theta_df_from_nshifts(n_shifts::Integer)
    return Int(n_shifts) + 1
end

function _ou_shift_parameter_df(n_shifts::Integer; ntraits::Integer = 1, shared_shift_edges::Bool = true)
    theta_df = _theta_df_from_nshifts(n_shifts)
    return Int(ntraits) * (theta_df + _OU_SHIFT_GLOBAL_DF)
end

function _compute_aicc(loglik::Float64, nparams::Integer, n::Integer)
    k = nparams
    penalty = 2.0 * k + 2.0 * k * (k + 1.0) / max(n - k - 1.0, 1.0)
    return -2.0 * loglik + penalty
end

function _compute_bic(loglik::Float64, nparams::Integer, n::Integer)
    return -2.0 * loglik + nparams * log(n)
end

function _convergent_nshiftvals(merge_map::Dict{Int,Int}, shift_edges::AbstractVector{<:Integer})
    isempty(merge_map) && return length(shift_edges)
    states = Set{Int}()
    for state in 2:(length(shift_edges) + 1)
        push!(states, get(merge_map, state, state))
    end
    return length(states)
end

function _convergent_parameter_df(
    n_shift_edges::Integer,
    n_shift_values::Integer;
    ntraits::Integer = 1,
)
    return Int(n_shift_values) + Int(ntraits) * (Int(n_shift_edges) + _OU_SHIFT_GLOBAL_DF + 1)
end

function _l1ou_mbic_penalty_parts(
    cache::OUShiftTreeCache,
    shift_edges::AbstractVector{<:Integer},
    covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    edges_workspace::Union{Nothing, Vector{Int}} = nothing,
)
    edges =
        edges_workspace === nothing ?
        Int.(shift_edges) :
        edges_workspace
    if edges_workspace !== nothing
        empty!(edges)
        sizehint!(edges, length(shift_edges))
        @inbounds for edge in shift_edges
            push!(edges, Int(edge))
        end
    end
    sort_edges_l1ou!(cache, edges)
    n = cache.ntips
    isempty(edges) && return (df1 = 0.0, df2 = 3.0 * log(n))

    df1 = (2.0 * length(edges) - 1.0) * log(n)
    df2 = 3.0 * log(n)
    covered =
        covered_workspace === nothing || length(covered_workspace) != n ?
        falses(n) :
        covered_workspace
    fill!(covered, false)
    ncovered = 0
    @inbounds for edge in edges
        nnew = 0
        for tip_pos in cache.descendant_tip_positions[edge]
            if !covered[tip_pos]
                covered[tip_pos] = true
                nnew += 1
                ncovered += 1
            end
        end
        nnew > 0 || throw(ArgumentError("shift configuration is not parsimonious under l1ou mBIC"))
        df2 += log(nnew)
    end
    nuncovered = n - ncovered
    nuncovered > 0 || throw(ArgumentError("shift configuration covers all tips under l1ou mBIC"))
    df2 += log(nuncovered)
    return (df1 = df1, df2 = df2)
end

function _compute_l1ou_mbic(
    loglik::Float64,
    cache::OUShiftTreeCache,
    shift_edges::AbstractVector{<:Integer};
    ntraits::Integer = 1,
)
    parts = try
        _l1ou_mbic_penalty_parts(cache, shift_edges)
    catch err
        err isa ArgumentError || rethrow()
        return Inf
    end
    return parts.df1 - 2.0 * Float64(loglik) + Int(ntraits) * parts.df2
end

