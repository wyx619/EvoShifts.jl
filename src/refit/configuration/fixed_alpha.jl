function _profile_oum_fixed_alpha(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    cache::OUMEdgeSegmentCache,
    alpha::Float64,
    cross_workspace::Union{Nothing, _ShiftCrossproductWorkspace} = nothing;
    keep_theta::Bool = true,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    isfinite(alpha) && alpha > 0.0 || return (success = false, loglik = -Inf, alpha = alpha, sigma2 = NaN, theta = Float64[])
    n = tree.ntips
    edge_a, edge_v =
        cross_workspace === nothing ?
        _shift_screening_edges(tree, alpha, 1.0) :
        _fill_shift_screening_edges!(cross_workspace.edge_a, cross_workspace.edge_v, tree, alpha)
    z =
        cross_workspace === nothing ?
        _fill_oum_profile_z!(
            Matrix{Float64}(undef, n, cache.nregimes + 1),
            tree,
            cache,
            trait,
            alpha,
            Matrix{Float64}(undef, tree.nnodes, cache.nregimes),
        ) :
        _fill_oum_profile_z!(
            cross_workspace.z,
            tree,
            cache,
            trait,
            alpha,
            @view(cross_workspace.means[:, 1:cache.nregimes]),
        )
    cross = _shift_tip_crossproducts_root_fixed(tree, z, edge_a, edge_v, cross_workspace, root_model, alpha; profile_only = true)
    cross.success || return (success = false, loglik = -Inf, alpha = alpha, sigma2 = NaN, theta = Float64[])

    yy = cross.cross[1, 1]
    p = cache.nregimes
    Xy = cross_workspace === nothing ? Vector{Float64}(undef, p) : cross_workspace.xy
    XX = cross_workspace === nothing ? Matrix{Float64}(undef, p, p) : cross_workspace.xx
    XXfactor = cross_workspace === nothing ? Matrix{Float64}(undef, p, p) : cross_workspace.xx_factor
    theta_buf = cross_workspace === nothing ? Vector{Float64}(undef, p) : cross_workspace.theta
    @inbounds for j in 1:p
        Xy[j] = cross.cross[j + 1, 1]
        for h in j:p
            XXfactor[j, h] = cross.cross[j + 1, h + 1]
        end
    end

    theta =
        try
            if !_profile_cholesky_solve!(theta_buf, XXfactor, Xy)
                _fill_profile_xx_fallback!(XX, cross.cross, p)
                theta_buf .= qr(XX, ColumnNorm()) \ Xy
            end
            theta_buf
        catch
            try
                _fill_profile_xx_fallback!(XX, cross.cross, p)
                theta_buf .= svd(XX) \ Xy
                theta_buf
            catch
                return (success = false, loglik = -Inf, alpha = alpha, sigma2 = NaN, theta = Float64[])
            end
        end
    rss = yy - dot(theta, Xy)
    rss > 0.0 && isfinite(rss) || return (success = false, loglik = -Inf, alpha = alpha, sigma2 = NaN, theta = Float64[])
    sigma2 = rss / n
    f0 = Float64(cross.logdet) + n * log(2π)
    loglik = -0.5 * (f0 + n * log(sigma2) + n)
    return (
        success = isfinite(loglik),
        loglik = loglik,
        alpha = alpha,
        sigma2 = sigma2,
        theta = keep_theta ? Vector{Float64}(theta) : Float64[],
    )
end

