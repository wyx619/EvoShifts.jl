function _linear_gaussian_loglik(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_a::AbstractVector{<:Real},
    edge_b::AbstractVector{<:Real},
    edge_v::AbstractVector{<:Real};
    root_prior_mean::Float64 = 0.0,
    root_prior_var::Float64 = Inf,
    profile_root::Bool = true,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    length(edge_a) == tree.nedges || throw(ArgumentError("edge_a must have $(tree.nedges) entries"))
    length(edge_b) == tree.nedges || throw(ArgumentError("edge_b must have $(tree.nedges) entries"))
    length(edge_v) == tree.nedges || throw(ArgumentError("edge_v must have $(tree.nedges) entries"))

    a = Float64.(edge_a)
    b = Float64.(edge_b)
    v = Float64.(edge_v)
    any(x -> !isfinite(x) || x <= 0.0, a) && return (success = false, loglik = -Inf, root_state = NaN)
    any(x -> !isfinite(x) || x < 0.0, v) && return (success = false, loglik = -Inf, root_state = NaN)

    precision = zeros(Float64, tree.nnodes)
    linear = zeros(Float64, tree.nnodes)
    logconst = zeros(Float64, tree.nnodes)

    tip_index = zeros(Int, tree.nnodes)
    for (i, node) in enumerate(tree.tip_ids)
        tip_index[node] = i
    end

    for node in tree.postorder_internal
        precision[node] = 0.0
        linear[node] = 0.0
        logconst[node] = 0.0

        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = tree.child_of_edge[edge]
            msg =
                if tree.is_tip[child]
                    _scalar_observation_info_to_parent(tr[tip_index[child]], a[edge], b[edge], v[edge])
                else
                    _scalar_info_to_parent(precision[child], linear[child], logconst[child], a[edge], b[edge], v[edge])
                end
            msg.success || return (success = false, loglik = -Inf, root_state = NaN)
            precision[node] += msg.precision
            linear[node] += msg.linear
            logconst[node] += msg.logconst
        end
    end

    root = Int(tree.root)
    return _scalar_root_info_loglik(
        precision[root],
        linear[root],
        logconst[root],
        root_prior_mean,
        root_prior_var,
        profile_root,
    )
end
