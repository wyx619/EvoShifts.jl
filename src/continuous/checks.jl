function _validate_univariate_trait(tree::CompactTree, trait::AbstractVector{<:Real})
    length(trait) == tree.ntips || throw(ArgumentError("Expected $(tree.ntips) trait values, got $(length(trait))"))
    any(x -> !isfinite(x), trait) && throw(ArgumentError("Trait vector contains non-finite values"))
    return Float64.(trait)
end

function _validate_univariate_trait_allow_missing(tree::CompactTree, trait::AbstractVector{<:Real})
    length(trait) == tree.ntips || throw(ArgumentError("Expected $(tree.ntips) trait values, got $(length(trait))"))
    data = Float64.(trait)
    any(x -> isinf(x), data) && throw(ArgumentError("Trait vector contains infinite values"))
    all(isnan, data) && throw(ArgumentError("Trait vector contains no observed values"))
    return data
end

function _validate_binary_tree(tree::CompactTree)
    for node in tree.postorder_internal
        length(tree.children[node]) == 2 || throw(ArgumentError("Current continuous tree-pruning models require a bifurcating tree"))
    end
    return true
end

function _validate_ultrametric_tree(tree::CompactTree; atol::Float64 = 1e-5)
    tip_heights = tree.dist_from_root[tree.tip_ids]
    maximum(abs.(tip_heights .- first(tip_heights))) <= atol || throw(ArgumentError("Current OU1/EB tree-pruning models require an ultrametric tree"))
    return true
end
