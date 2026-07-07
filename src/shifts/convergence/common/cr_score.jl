function _cr_supports(
    cache::OUShiftTreeCache,
    shift_edges::AbstractVector{<:Integer},
)
    supports = Vector{Vector{Int}}(undef, length(shift_edges) + 1)
    supports[1] = collect(1:cache.ntips)
    @inbounds for (j, edge) in enumerate(shift_edges)
        supports[j + 1] = sort!(copy(cache.descendant_tip_positions[Int(edge)]))
    end
    return supports
end

struct _CRDesignContext
    n::Int
    d::Int
    shift_edges::Vector{Int}
    edge_to_col::Vector{Int}
    parent_ages::Vector{Float64}
    support_subset::BitMatrix
end

mutable struct _CRTraitWorkspace
    z::Matrix{Float64}
    cross_workspace::_ShiftCrossproductWorkspace
    component_xy::Vector{Float64}
    component_beta::Vector{Float64}
    component_xx::Matrix{Float64}
    component_factor::Matrix{Float64}
end

struct _CRComponentColumns
    group_of_col::Vector{Int}
    n_groups::Int
end

function _cr_design_context(
    cache::OUShiftTreeCache,
    shift_edges::AbstractVector{<:Integer},
)
    edges = Int.(shift_edges)
    d = length(edges) + 1
    max_edge = isempty(edges) ? 0 : maximum(edges)
    edge_to_col = zeros(Int, max_edge)
    parent_ages = Vector{Float64}(undef, length(edges))
    @inbounds for (j, edge) in enumerate(edges)
        edge_to_col[edge] = j + 1
        parent_ages[j] = cache.tree_height - cache.dist_from_root[cache.edge_parent[edge]]
    end
    supports = _cr_supports(cache, edges)
    support_subset = falses(d, d)
    @inbounds for i in 1:d
        set1 = supports[i]
        for j in 1:d
            i == j && continue
            support_subset[i, j] = _is_subset_sorted(set1, supports[j])
        end
    end
    return _CRDesignContext(cache.ntips, d, edges, edge_to_col, parent_ages, support_subset)
end

function _cr_trait_workspace(tree::CompactTree, ctx::_CRDesignContext)
    k = ctx.d + 1
    return _CRTraitWorkspace(
        Matrix{Float64}(undef, tree.ntips, k),
        _shift_crossproduct_workspace(tree, k),
        Vector{Float64}(undef, ctx.d),
        Vector{Float64}(undef, ctx.d),
        Matrix{Float64}(undef, ctx.d, ctx.d),
        Matrix{Float64}(undef, ctx.d, ctx.d),
    )
end

function _fill_cr_design_z!(
    z::Matrix{Float64},
    trait::AbstractVector{<:Real},
    cache::OUShiftTreeCache,
    ctx::_CRDesignContext,
    alpha::Float64,
)
    n = ctx.n
    d = ctx.d
    @inbounds for i in 1:n
        z[i, 1] = Float64(trait[i])
        z[i, 2] = 1.0
        for j in 3:(d + 1)
            z[i, j] = 0.0
        end
    end

    @inbounds for (j, edge) in enumerate(ctx.shift_edges)
        age = ctx.parent_ages[j]
        weight = alpha <= 0.0 ? age : 1.0 - exp(-alpha * age)
        col = j + 2
        for tip_pos in cache.descendant_tip_positions[edge]
            z[tip_pos, col] = weight
        end
    end

    @inbounds for i in 1:d
        zi = i + 1
        for j in 1:d
            i == j && continue
            ctx.support_subset[i, j] || continue
            zj = j + 1
            for r in 1:n
                z[r, zj] -= z[r, zi]
            end
        end
    end
    return z
end

function _is_subset_sorted(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    ia = firstindex(a)
    ib = firstindex(b)
    enda = lastindex(a)
    endb = lastindex(b)
    while ia <= enda && ib <= endb
        av = Int(a[ia])
        bv = Int(b[ib])
        if av == bv
            ia += 1
            ib += 1
        elseif av > bv
            ib += 1
        else
            return false
        end
    end
    return ia > enda
end

function _cr_profile_loglik_for_alpha!(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    ctx::_CRDesignContext,
    workspace::_CRTraitWorkspace,
    component_cols::Vector{Vector{Int}},
    design_alpha::Float64,
    cov_alpha::Float64,
)
    edge_a, edge_v = _fill_shift_screening_edges!(
        workspace.cross_workspace.edge_a,
        workspace.cross_workspace.edge_v,
        tree,
        cov_alpha,
    )
    z = _fill_cr_design_z!(workspace.z, trait, cache, ctx, design_alpha)
    cross = _shift_tip_crossproducts_root_fixed(tree, z, edge_a, edge_v, workspace.cross_workspace)
    cross.success || return -Inf
    f0 = Float64(cross.logdet) + tree.ntips * log(2.0 * pi)
    return _cr_profile_loglik_from_cross!(
        cross.cross,
        f0,
        component_cols,
        tree.ntips,
        workspace,
    )
end

function _cr_profile_loglik_for_alpha!(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    ctx::_CRDesignContext,
    workspace::_CRTraitWorkspace,
    component_cols::_CRComponentColumns,
    design_alpha::Float64,
    cov_alpha::Float64,
)
    edge_a, edge_v = _fill_shift_screening_edges!(
        workspace.cross_workspace.edge_a,
        workspace.cross_workspace.edge_v,
        tree,
        cov_alpha,
    )
    z = _fill_cr_design_z!(workspace.z, trait, cache, ctx, design_alpha)
    cross = _shift_tip_crossproducts_root_fixed(tree, z, edge_a, edge_v, workspace.cross_workspace)
    cross.success || return -Inf
    f0 = Float64(cross.logdet) + tree.ntips * log(2.0 * pi)
    return _cr_profile_loglik_from_cross!(
        cross.cross,
        f0,
        component_cols,
        tree.ntips,
        workspace,
    )
end

function _cr_component_column_groups!(
    group_of_col::Vector{Int},
    ctx::_CRDesignContext,
    components::Vector{Vector{Int}},
)
    fill!(group_of_col, 0)
    for (g, comp) in enumerate(components)
        for item in comp
            col =
                item == 0 ? 1 :
                1 <= item <= length(ctx.edge_to_col) ? ctx.edge_to_col[item] :
                0
            col == 0 && throw(ArgumentError("CR component contains non-shift edge $item"))
            group_of_col[col] = g
        end
    end
    return _CRComponentColumns(group_of_col, length(components))
end

function _cr_profile_loglik_from_cross!(
    cross::AbstractMatrix{<:Real},
    f0::Float64,
    component_cols::Vector{Vector{Int}},
    n::Integer,
    workspace::_CRTraitWorkspace,
)
    d = length(component_cols)
    Xy = workspace.component_xy
    beta = workspace.component_beta
    XX = workspace.component_xx
    XXfactor = workspace.component_factor
    @inbounds for g in 1:d
        xyg = 0.0
        for c in component_cols[g]
            xyg += cross[c + 1, 1]
        end
        Xy[g] = xyg
        for h in 1:g
            xx = 0.0
            for c in component_cols[g]
                for c2 in component_cols[h]
                    xx += cross[c + 1, c2 + 1]
                end
            end
            XX[g, h] = xx
            XX[h, g] = xx
            XXfactor[g, h] = xx
            XXfactor[h, g] = xx
        end
    end

    try
        chol = cholesky!(Symmetric(@view(XXfactor[1:d, 1:d])), check = false)
        if issuccess(chol)
            ldiv!(@view(beta[1:d]), chol, @view(Xy[1:d]))
        else
            beta[1:d] .= qr(@view(XX[1:d, 1:d]), ColumnNorm()) \ @view(Xy[1:d])
        end
    catch
        try
            beta[1:d] .= svd(@view(XX[1:d, 1:d])) \ @view(Xy[1:d])
        catch
            return -Inf
        end
    end

    rss = Float64(cross[1, 1])
    @inbounds for i in 1:d
        rss -= beta[i] * Xy[i]
    end
    rss > 0.0 && isfinite(rss) || return -Inf
    return -0.5 * (f0 + Int(n) * log(rss / Int(n)) + Int(n))
end

function _cr_profile_loglik_from_cross!(
    cross::AbstractMatrix{<:Real},
    f0::Float64,
    component_cols::_CRComponentColumns,
    n::Integer,
    workspace::_CRTraitWorkspace,
)
    d = component_cols.n_groups
    group_of_col = component_cols.group_of_col
    Xy = workspace.component_xy
    beta = workspace.component_beta
    XX = workspace.component_xx
    XXfactor = workspace.component_factor
    @inbounds for g in 1:d
        Xy[g] = 0.0
        for h in 1:d
            XX[g, h] = 0.0
            XXfactor[g, h] = 0.0
        end
    end
    @inbounds for c in eachindex(group_of_col)
        g = group_of_col[c]
        g == 0 && continue
        Xy[g] += cross[c + 1, 1]
        for c2 in eachindex(group_of_col)
            h = group_of_col[c2]
            h == 0 && continue
            XX[g, h] += cross[c + 1, c2 + 1]
        end
    end
    @inbounds for g in 1:d
        for h in 1:d
            XXfactor[g, h] = XX[g, h]
        end
    end

    try
        chol = cholesky!(Symmetric(@view(XXfactor[1:d, 1:d])), check = false)
        if issuccess(chol)
            ldiv!(@view(beta[1:d]), chol, @view(Xy[1:d]))
        else
            beta[1:d] .= qr(@view(XX[1:d, 1:d]), ColumnNorm()) \ @view(Xy[1:d])
        end
    catch
        try
            beta[1:d] .= svd(@view(XX[1:d, 1:d])) \ @view(Xy[1:d])
        catch
            return -Inf
        end
    end

    rss = Float64(cross[1, 1])
    @inbounds for i in 1:d
        rss -= beta[i] * Xy[i]
    end
    rss > 0.0 && isfinite(rss) || return -Inf
    return -0.5 * (f0 + Int(n) * log(rss / Int(n)) + Int(n))
end

function _l1ou_alpha_upper_bound(tree::CompactTree)
    min_terminal = Inf
    @inbounds for e in 1:tree.nedges
        child = Int(tree.child_of_edge[e])
        tree.is_tip[child] && (min_terminal = min(min_terminal, Float64(tree.edge_length[e])))
    end
    isfinite(min_terminal) && min_terminal > 0.0 || return 1e8
    return log(2.0) / min_terminal
end

function _cr_profile_fit_components_l1ou_optim(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    shift_edges::AbstractVector{<:Integer},
    components::Vector{Vector{Int}},
    start_alpha::Float64;
    max_iterations::Integer = 50,
    ctx::Union{Nothing, _CRDesignContext} = nothing,
    workspace::Union{Nothing, _CRTraitWorkspace} = nothing,
    group_of_col::Union{Nothing, Vector{Int}} = nothing,
)
    alpha0 = isfinite(start_alpha) && start_alpha > 0.0 ? start_alpha : max(log(2.0) / max(cache.tree_height, eps(Float64)), 1e-7)
    lower = max(alpha0 / 100.0, eps(Float64))
    upper = max(alpha0, _l1ou_alpha_upper_bound(tree) + eps(Float64))
    lower < upper || (upper = lower * 100.0)
    local_ctx = ctx === nothing ? _cr_design_context(cache, shift_edges) : ctx
    local_workspace = workspace === nothing ? _cr_trait_workspace(tree, local_ctx) : workspace
    comps = [sort!(copy(c)) for c in components]
    sort!(comps; by = c -> (minimum(c), length(c)))
    local_group_of_col = group_of_col === nothing ? Vector{Int}(undef, local_ctx.d) : group_of_col
    component_cols = _cr_component_column_groups!(local_group_of_col, local_ctx, comps)
    objective = log_alpha -> begin
        alpha = exp(Float64(log_alpha))
        ll = _cr_profile_loglik_for_alpha!(
            tree, cache, trait, local_ctx, local_workspace, component_cols, alpha, alpha
        )
        isfinite(ll) ? -ll : Inf
    end
    result = Optim.optimize(
        objective,
        log(lower),
        log(upper),
        Optim.Brent();
        iterations = Int(max_iterations),
    )
    alpha_hat = exp(Optim.minimizer(result))
    ll = _cr_profile_loglik_for_alpha!(
        tree, cache, trait, local_ctx, local_workspace, component_cols, alpha0, alpha_hat
    )
    return (success = isfinite(ll), loglik = ll, alpha = alpha_hat)
end

function _score_cr_components_l1ou_optim(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    shift_edges::AbstractVector{<:Integer},
    components::Vector{Vector{Int}},
    start_alpha::Float64;
    criterion::Symbol = :BIC,
    max_iterations::Integer = 50,
    ctx::Union{Nothing, _CRDesignContext} = nothing,
    workspace::Union{Nothing, _CRTraitWorkspace} = nothing,
    group_of_col::Union{Nothing, Vector{Int}} = nothing,
)
    fit = _cr_profile_fit_components_l1ou_optim(
        tree, cache, trait, shift_edges, components, start_alpha;
        max_iterations = max_iterations,
        ctx = ctx,
        workspace = workspace,
        group_of_col = group_of_col,
    )
    fit.success || return Inf
    n_shifts = length(shift_edges)
    n_shift_values = max(length(components) - 1, 0)
    if criterion === :AICc
        return _compute_aicc(fit.loglik, n_shifts + n_shift_values + 3, tree.ntips)
    elseif criterion === :BIC
        return _compute_bic(fit.loglik, n_shift_values + n_shifts + 3, tree.ntips)
    end
    throw(ArgumentError("Unsupported criterion: $criterion"))
end

function _score_cr_components_l1ou_optim_mv(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    shift_edges::AbstractVector{<:Integer},
    components::Vector{Vector{Int}},
    start_alpha::AbstractVector{<:Real};
    criterion::Symbol = :BIC,
    max_iterations::Integer = 50,
    ctx::Union{Nothing, _CRDesignContext} = nothing,
    workspaces::Union{Nothing, Vector{_CRTraitWorkspace}} = nothing,
    group_buffers::Union{Nothing, Vector{Vector{Int}}} = nothing,
)
    total_ll = 0.0
    m = size(trait_mat, 2)
    local_ctx = ctx === nothing ? _cr_design_context(cache, shift_edges) : ctx
    local_workspaces =
        workspaces === nothing || length(workspaces) != m ?
        [_cr_trait_workspace(tree, local_ctx) for _ in 1:m] :
        workspaces
    local_group_buffers =
        group_buffers === nothing || length(group_buffers) != m ?
        [Vector{Int}(undef, local_ctx.d) for _ in 1:m] :
        group_buffers
    @inbounds for j in 1:m
        a0 = j <= length(start_alpha) ? Float64(start_alpha[j]) : 0.0
        fit = _cr_profile_fit_components_l1ou_optim(
            tree, cache, @view(trait_mat[:, j]), shift_edges, components, a0;
            max_iterations = max_iterations,
            ctx = local_ctx,
            workspace = local_workspaces[j],
            group_of_col = local_group_buffers[j],
        )
        fit.success || return Inf
        total_ll += fit.loglik
    end
    n_shifts = length(shift_edges)
    n_shift_values = max(length(components) - 1, 0)
    if criterion === :AICc
        p = n_shifts + (n_shift_values + 3) * m
        return _compute_aicc(total_ll, p, tree.ntips * m)
    elseif criterion === :BIC
        p = n_shift_values + m * (n_shifts + 3)
        return -2.0 * total_ll + p * log(tree.ntips)
    end
    throw(ArgumentError("Unsupported criterion: $criterion"))
end

