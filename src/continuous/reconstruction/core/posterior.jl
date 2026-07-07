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

function _linear_gaussian_asr(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_a::AbstractVector{<:Real},
    edge_b::AbstractVector{<:Real},
    edge_v::AbstractVector{<:Real};
    root_prior_mean::Float64 = 0.0,
    root_prior_var::Float64 = Inf,
    model::Symbol = :unknown,
)
    cache = _linear_gaussian_posterior_cache(
        tree,
        trait,
        edge_a,
        edge_b,
        edge_v;
        root_prior_mean = root_prior_mean,
        root_prior_var = root_prior_var,
    )
    cache.success || return ContinuousASRResult(model = model, success = false)

    internal_ids = Int32[node for node in 1:tree.nnodes if !tree.is_tip[node]]
    estimates = cache.full_mean[internal_ids]
    variances = cache.full_var[internal_ids]
    any(x -> !isfinite(x) || x < 0.0, variances) && return ContinuousASRResult(model = model, success = false)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    tip_ids = Int32.(tree.tip_ids)
    tip_estimates = cache.full_mean[tip_ids]
    tip_variances = cache.full_var[tip_ids]
    node_time_from_root = tree.dist_from_root[internal_ids]
    tree_height = maximum(tree.dist_from_root[tree.tip_ids])
    node_time_before_present = tree_height .- node_time_from_root

    return ContinuousASRResult(
        model = model,
        success = true,
        node_ids = internal_ids,
        node_labels = tree.node_labels[internal_ids],
        time_from_root = node_time_from_root,
        time_before_present = node_time_before_present,
        estimates = estimates,
        variances = variances,
        se = sqrt.(variances),
        all_node_estimates = cache.full_mean,
        all_node_variances = cache.full_var,
        tip_ids = tip_ids,
        tip_labels = tree.node_labels[tip_ids],
        tip_estimates = tip_estimates,
        tip_variances = tip_variances,
        tip_se = sqrt.(tip_variances),
        tip_observed = .!isnan.(tr),
    )
end

function _linear_gaussian_posterior_cache(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_a::AbstractVector{<:Real},
    edge_b::AbstractVector{<:Real},
    edge_v::AbstractVector{<:Real};
    root_prior_mean::Float64 = 0.0,
    root_prior_var::Float64 = Inf,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    length(edge_a) == tree.nedges || throw(ArgumentError("edge_a must have $(tree.nedges) entries"))
    length(edge_b) == tree.nedges || throw(ArgumentError("edge_b must have $(tree.nedges) entries"))
    length(edge_v) == tree.nedges || throw(ArgumentError("edge_v must have $(tree.nedges) entries"))

    a = Float64.(edge_a)
    b = Float64.(edge_b)
    v = Float64.(edge_v)
    any(x -> !isfinite(x) || x <= 0.0, a) && throw(ArgumentError("edge_a must be positive and finite"))
    any(x -> !isfinite(x) || x < 0.0, v) && throw(ArgumentError("edge_v must be non-negative and finite"))

    desc_mean = zeros(Float64, tree.nnodes)
    desc_var = fill(Inf, tree.nnodes)
    outside_mean = zeros(Float64, tree.nnodes)
    outside_var = fill(Inf, tree.nnodes)
    full_mean = zeros(Float64, tree.nnodes)
    full_var = fill(Inf, tree.nnodes)
    edge_context_mean = zeros(Float64, tree.nedges)
    edge_context_var = fill(Inf, tree.nedges)

    for (i, node) in enumerate(tree.tip_ids)
        if isnan(tr[i])
            desc_mean[node] = 0.0
            desc_var[node] = Inf
        else
            desc_mean[node] = tr[i]
            desc_var[node] = 0.0
        end
    end

    for node in tree.postorder_internal
        child1 = tree.children[node][1]
        child2 = tree.children[node][2]
        edge1 = tree.first_child_edge[node]
        edge2 = tree.last_child_edge[node]

        msg1 = _edge_message_to_parent(desc_mean[child1], desc_var[child1], a[edge1], b[edge1], v[edge1])
        msg2 = _edge_message_to_parent(desc_mean[child2], desc_var[child2], a[edge2], b[edge2], v[edge2])
        combined = _gaussian_product(msg1.mean, msg1.var, msg2.mean, msg2.var)
        isfinite(combined.mean) || return (success = false,)
        desc_mean[node] = combined.mean
        desc_var[node] = combined.var
    end

    root = Int(tree.root)
    outside_mean[root] = root_prior_mean
    outside_var[root] = root_prior_var

    for node in tree.preorder
        if tree.is_tip[node]
            full = _gaussian_product(desc_mean[node], desc_var[node], outside_mean[node], outside_var[node])
            full_mean[node] = full.mean
            full_var[node] = full.var
            continue
        end

        full = _gaussian_product(desc_mean[node], desc_var[node], outside_mean[node], outside_var[node])
        full_mean[node] = full.mean
        full_var[node] = full.var

        child1 = tree.children[node][1]
        child2 = tree.children[node][2]
        edge1 = tree.first_child_edge[node]
        edge2 = tree.last_child_edge[node]

        msg1 = _edge_message_to_parent(desc_mean[child1], desc_var[child1], a[edge1], b[edge1], v[edge1])
        msg2 = _edge_message_to_parent(desc_mean[child2], desc_var[child2], a[edge2], b[edge2], v[edge2])

        parent_for_child1 = _gaussian_product(outside_mean[node], outside_var[node], msg2.mean, msg2.var)
        parent_for_child2 = _gaussian_product(outside_mean[node], outside_var[node], msg1.mean, msg1.var)
        edge_context_mean[edge1] = parent_for_child1.mean
        edge_context_var[edge1] = parent_for_child1.var
        edge_context_mean[edge2] = parent_for_child2.mean
        edge_context_var[edge2] = parent_for_child2.var

        pred1 = _edge_predict_to_child(parent_for_child1.mean, parent_for_child1.var, a[edge1], b[edge1], v[edge1])
        pred2 = _edge_predict_to_child(parent_for_child2.mean, parent_for_child2.var, a[edge2], b[edge2], v[edge2])

        outside_mean[child1] = pred1.mean
        outside_var[child1] = pred1.var
        outside_mean[child2] = pred2.mean
        outside_var[child2] = pred2.var
    end

    return (
        success = true,
        desc_mean = desc_mean,
        desc_var = desc_var,
        outside_mean = outside_mean,
        outside_var = outside_var,
        full_mean = full_mean,
        full_var = full_var,
        edge_context_mean = edge_context_mean,
        edge_context_var = edge_context_var,
        edge_a = a,
        edge_b = b,
        edge_v = v,
    )
end
