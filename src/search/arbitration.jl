function _score_configuration_full(
    cache::OUShiftTreeCache,
    loglik::Float64,
    n_shifts::Integer,
    shift_edges::AbstractVector{<:Integer},
    n::Integer,
;
    criterion::Symbol = :mBIC,
    merge_map::Dict{Int,Int} = Dict{Int,Int}(),
)
    nparams =
        isempty(merge_map) ? _ou_shift_parameter_df(n_shifts) :
        _convergent_parameter_df(length(shift_edges), _convergent_nshiftvals(merge_map, shift_edges))
    if criterion === :AICc
        return _compute_aicc(loglik, nparams, n)
    elseif criterion === :BIC
        return _compute_bic(loglik, nparams, n)
    elseif criterion === :mBIC
        return _compute_l1ou_mbic(loglik, cache, shift_edges; ntraits = 1)
    end
    throw(ArgumentError("Unsupported criterion: $criterion"))
end

