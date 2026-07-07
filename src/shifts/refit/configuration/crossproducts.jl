
@inline function _zero_shift_cross_profile!(cross::AbstractMatrix{Float64}, k::Integer)
    kk = Int(k)
    @inbounds begin
        cross[1, 1] = 0.0
        for j in 2:kk
            cross[j, 1] = 0.0
        end
        for j in 2:kk
            for h in j:kk
                cross[j, h] = 0.0
            end
        end
    end
    return cross
end

@inline function _accumulate_shift_cross_profile!(
    cross::AbstractMatrix{Float64},
    contrast::AbstractVector{Float64},
    k::Integer,
)
    kk = Int(k)
    c1 = contrast[1]
    @inbounds begin
        cross[1, 1] += c1 * c1
        for j in 2:kk
            cross[j, 1] += contrast[j] * c1
        end
        for j in 2:kk
            cj = contrast[j]
            for h in j:kk
                cross[j, h] += cj * contrast[h]
            end
        end
    end
    return cross
end

@inline function _accumulate_shift_cross_full!(
    cross::AbstractMatrix{Float64},
    contrast::AbstractVector{Float64},
    k::Integer,
)
    kk = Int(k)
    @inbounds for j in 1:kk
        cj = contrast[j]
        for h in 1:j
            val = cj * contrast[h]
            cross[j, h] += val
            j == h || (cross[h, j] += val)
        end
    end
    return cross
end

function _fill_profile_xx_fallback!(
    XX::AbstractMatrix{Float64},
    cross::AbstractMatrix{Float64},
    p::Integer,
)
    pp = Int(p)
    @inbounds for j in 1:pp
        for h in 1:j - 1
            XX[j, h] = cross[h + 1, j + 1]
        end
        for h in j:pp
            XX[j, h] = cross[j + 1, h + 1]
        end
    end
    return XX
end

function _profile_cholesky_solve!(
    theta::AbstractVector{Float64},
    factor::AbstractMatrix{Float64},
    rhs::AbstractVector{Float64},
)
    F, info = LAPACK.potrf!('U', factor)
    info == 0 || return false
    theta .= rhs
    LAPACK.potrs!('U', F, theta)
    return all(isfinite, theta)
end

function _shift_tip_crossproducts_root_fixed(
    tree::CompactTree,
    z::AbstractMatrix{<:Real},
    edge_a::AbstractVector{<:Real},
    edge_v::AbstractVector{<:Real},
    workspace::Union{Nothing, _ShiftCrossproductWorkspace} = nothing,
    root_model::Symbol = :OUfixedRoot,
    alpha::Float64 = 0.0,
    ;
    profile_only::Bool = false,
)
    root_model = _normalize_ou_root_model(root_model)
    size(z, 1) == tree.ntips || throw(ArgumentError("z must have one row per tip"))
    length(edge_a) == tree.nedges || throw(ArgumentError("edge_a must have $(tree.nedges) entries"))
    length(edge_v) == tree.nedges || throw(ArgumentError("edge_v must have $(tree.nedges) entries"))

    _validate_binary_tree(tree)
    k = size(z, 2)
    any(x -> !isfinite(x), z) && return (success = false, cross = zeros(k, k), logdet = Inf)

    ws = workspace === nothing ? _shift_crossproduct_workspace(tree, k) : workspace
    size(ws.means, 2) == k || throw(ArgumentError("workspace has wrong column count"))
    spreads = ws.spreads
    precision = ws.precision
    means = ws.means
    contrast = ws.contrast
    profile_only ? _zero_shift_cross_profile!(ws.cross, k) : fill!(ws.cross, 0.0)
    logdet_info = 0.0

    @inbounds for (i, node0) in enumerate(tree.tip_ids)
        node = Int(node0)
        spreads[node] = 0.0
        precision[node] = 0.0
        for j in 1:k
            means[node, j] = Float64(z[i, j])
        end
    end

    @inbounds for node0 in tree.postorder_internal
        node = Int(node0)
        length(tree.children[node]) == 2 ||
            return (success = false, cross = zeros(k, k), logdet = Inf)
        child1 = Int(tree.children[node][1])
        child2 = Int(tree.children[node][2])
        edge1 = Int(tree.first_child_edge[node])
        edge2 = Int(tree.last_child_edge[node])
        a1 = Float64(edge_a[edge1])
        a2 = Float64(edge_a[edge2])
        v1e = Float64(edge_v[edge1])
        v2e = Float64(edge_v[edge2])
        (isfinite(a1) && a1 > 0.0 && isfinite(a2) && a2 > 0.0 &&
         isfinite(v1e) && v1e >= 0.0 && isfinite(v2e) && v2e >= 0.0) ||
            return (success = false, cross = zeros(k, k), logdet = Inf)

        v1 = (spreads[child1] + max(v1e, 1e-12)) / (a1 * a1)
        v2 = (spreads[child2] + max(v2e, 1e-12)) / (a2 * a2)
        denom = v1 + v2
        (isfinite(denom) && denom > 0.0) ||
            return (success = false, cross = zeros(k, k), logdet = Inf)
        invsqrt = 1.0 / sqrt(denom)
        invdenom = 1.0 / denom

        p1 =
            if tree.is_tip[child1]
                a1 * a1 / max(v1e, 1e-12)
            else
                denom1 = 1.0 + max(v1e, 1e-12) * precision[child1]
                (isfinite(denom1) && denom1 > 0.0) ||
                    return (success = false, cross = zeros(k, k), logdet = Inf)
                logdet_info += log(denom1)
                a1 * a1 * precision[child1] / denom1
            end
        p2 =
            if tree.is_tip[child2]
                a2 * a2 / max(v2e, 1e-12)
            else
                denom2 = 1.0 + max(v2e, 1e-12) * precision[child2]
                (isfinite(denom2) && denom2 > 0.0) ||
                    return (success = false, cross = zeros(k, k), logdet = Inf)
                logdet_info += log(denom2)
                a2 * a2 * precision[child2] / denom2
            end
        tree.is_tip[child1] && (logdet_info += log(max(v1e, 1e-12)))
        tree.is_tip[child2] && (logdet_info += log(max(v2e, 1e-12)))
        precision[node] = p1 + p2

        for j in 1:k
            contrast[j] = (means[child1, j] / a1 - means[child2, j] / a2) * invsqrt
        end
        profile_only ?
            _accumulate_shift_cross_profile!(ws.cross, contrast, k) :
            _accumulate_shift_cross_full!(ws.cross, contrast, k)
        for j in 1:k
            m1 = means[child1, j] / a1
            m2 = means[child2, j] / a2
            means[node, j] = (m1 * v2 + m2 * v1) * invdenom
        end
        spreads[node] = (v1 * v2) * invdenom
    end

    root = Int(tree.root)
    root_var = spreads[root]
    if root_model === :OUrandomRoot && alpha > 0.0
        random_root_var = 1.0 / (2.0 * alpha)
        root_var_random = root_var + random_root_var
        if isfinite(root_var) && root_var > 0.0 && isfinite(root_var_random) && root_var_random > 0.0
            logdet_info += log(root_var_random / root_var)
            root_var = root_var_random
        else
            return (success = false, cross = zeros(k, k), logdet = Inf)
        end
    end
    (isfinite(root_var) && root_var > 0.0) ||
        return (success = false, cross = zeros(k, k), logdet = Inf)
    inv_root_var = 1.0 / root_var
    @inbounds for j in 1:k
        contrast[j] = means[root, j] * sqrt(inv_root_var)
    end
    profile_only ?
        _accumulate_shift_cross_profile!(ws.cross, contrast, k) :
        _accumulate_shift_cross_full!(ws.cross, contrast, k)
    return (success = isfinite(logdet_info), cross = ws.cross, logdet = logdet_info)
end

