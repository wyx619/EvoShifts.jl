function _shift_edge_age(tree::CompactTree, edge::Integer)
    parent = Int(tree.parent_of_edge[Int(edge)])
    height = maximum(tree.dist_from_root[tree.tip_ids])
    return height - tree.dist_from_root[parent]
end

@inline function _shift_design_weight(age::Float64, alpha::Float64, mode::Symbol)
    if mode === :simp
        return 1.0
    elseif mode === :bm_approx
        return age
    elseif mode === :ou
        alpha > 0.0 || return age
        return 1.0 - exp(-alpha * age)
    end
    throw(ArgumentError("Unsupported shift design mode: $mode"))
end

function _precompute_design_weights(
    tree::CompactTree,
    candidates::AbstractVector{<:Integer},
    alpha::Float64;
    mode::Union{Nothing, Symbol} = nothing,
)
    design_mode = mode === nothing ? (alpha <= 0.0 ? :bm_approx : :ou) : mode
    weights = Vector{Float64}(undef, length(candidates))
    @inbounds for (j, e) in enumerate(candidates)
        age = _shift_edge_age(tree, e)
        weights[j] = _shift_design_weight(age, alpha, design_mode)
    end
    return weights
end

@inline function _get_descendant_tips_positions(cache::OUShiftTreeCache, edge::Integer)
    return cache.descendant_tip_positions[Int(edge)]
end

