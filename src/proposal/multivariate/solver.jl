@inline function _pruning_group_path_step!(
    beta_new::AbstractVector{Float64},
    coef::AbstractVector{Float64},
    grad::AbstractVector{Float64},
    lambda::Float64,
    hessian::Float64,
    group_size::Integer,
)
    border = lambda * sqrt(Float64(group_size))
    cond_norm2 = 0.0
    @inbounds for i in eachindex(coef)
        c = -grad[i] + hessian * coef[i]
        beta_new[i] = c
        cond_norm2 += c * c
    end
    if cond_norm2 > border * border
        cond_norm = sqrt(cond_norm2)
        @inbounds for i in eachindex(coef)
            beta_new[i] = coef[i] + (-grad[i] - border * beta_new[i] / cond_norm) / hessian
        end
    else
        fill!(beta_new, 0.0)
    end
    return beta_new
end

function _pruning_group_path_penalty_value(beta::AbstractMatrix{<:Real}, lambda::Float64)
    ntraits, ncandidates = size(beta)
    group_sizes = fill(ntraits, ncandidates)
    return _pruning_group_path_penalty_value(beta, lambda, group_sizes)
end

function _pruning_group_path_penalty_value(beta::AbstractMatrix{<:Real}, lambda::Float64, group_sizes::AbstractVector{<:Integer})
    ntraits, ncandidates = size(beta)
    total = 0.0
    @inbounds for j in 1:ncandidates
        s = 0.0
        for i in 1:ntraits
            s += beta[i, j] * beta[i, j]
        end
        scale = lambda * sqrt(Float64(group_sizes[j]))
        total += scale * sqrt(s)
    end
    return total
end

function _pruning_group_path_loss_value(
    y_list::Vector{Vector{Float64}},
    eta_list::Vector{Vector{Float64}},
)
    total = 0.0
    @inbounds for i in eachindex(y_list)
        y = y_list[i]
        eta = eta_list[i]
        for r in eachindex(y)
            d = y[r] - eta[r]
            total += d * d
        end
    end
    return total
end

@inline function _pruning_group_path_hessian_col(x::AbstractVector{Float64})
    return max(2.0 * LinearAlgebra.dot(x, x), 0.01)
end

function _pruning_group_path_fit_operator!(
    beta::Matrix{Float64},
    y_list::Vector{Vector{Float64}},
    ncandidates::Integer,
    lambdas::AbstractVector{<:Real};
    get_group_cols!::Function,
    hessian::AbstractVector{<:Real},
    max_iterations::Integer = 500,
    tol::Float64 = 1e-6,
    group_sizes::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    on_solution::Union{Nothing, Function} = nothing,
)
    ntraits, p = size(beta)
    Int(ncandidates) == p || throw(ArgumentError("operator candidate count does not match beta"))
    length(hessian) == p || throw(ArgumentError("hessian length does not match candidate count"))
    gsize = group_sizes === nothing ? fill(ntraits, p) : Int.(group_sizes)
    eta_list = [zeros(Float64, length(y)) for y in y_list]
    residual_list = [copy(y) for y in y_list]
    nH = Float64.(hessian)

    converged = trues(length(lambdas))
    beta_old = similar(beta)
    beta_new = zeros(Float64, ntraits)
    dvec = zeros(Float64, ntraits)
    grad = zeros(Float64, ntraits)
    xjd_buf = [Vector{Float64}(undef, length(y_list[i])) for i in 1:ntraits]
    xjd_active = falses(ntraits)
    norms = zeros(Float64, p)

    for (pos, lambda0) in enumerate(lambdas)
        lambda = Float64(lambda0)
        loss_current = _pruning_group_path_loss_value(y_list, eta_list)
        fn_val = loss_current + _pruning_group_path_penalty_value(beta, lambda, gsize)
        d_fn = 1.0
        d_par = 1.0
        do_all = false
        counter = 1
        iter_count = 0
        fill!(norms, 0.0)
        @inbounds for j in 1:p
            s = 0.0
            for i in 1:ntraits
                s += beta[i, j] * beta[i, j]
            end
            norms[j] = sqrt(s)
        end

        while d_fn > tol || d_par > sqrt(tol) || !do_all
            if iter_count >= max_iterations
                converged[pos] = false
                break
            end
            fn_old = fn_val
            copyto!(beta_old, beta)
            guessed_active =
                if counter == 0 || counter > 10
                    do_all = true
                    counter = 1
                    1:p
                else
                    inds = findall(!iszero, norms)
                    if isempty(inds)
                        do_all = true
                        1:p
                    else
                        do_all = false
                        counter += 1
                        inds
                    end
                end
            do_all && (iter_count += 1)

            for j in guessed_active
                cols = get_group_cols!(j)
                fill!(grad, 0.0)
                @inbounds for i in 1:ntraits
                    col = cols[i]
                    if col !== nothing
                        s = 0.0
                        resid = residual_list[i]
                        for r in eachindex(col)
                            s += col[r] * resid[r]
                        end
                        grad[i] = -2.0 * s
                    end
                end

                hj = nH[j]
                cond_norm2 = 0.0
                @inbounds for i in 1:ntraits
                    c = -grad[i] + hj * beta[i, j]
                    beta_new[i] = c
                    cond_norm2 += c * c
                end
                border = lambda * sqrt(Float64(gsize[j]))
                if cond_norm2 > border * border
                    cond_norm = sqrt(cond_norm2)
                    @inbounds for i in 1:ntraits
                        dvec[i] = (-grad[i] - border * beta_new[i] / cond_norm) / hj
                    end
                else
                    @inbounds for i in 1:ntraits
                        dvec[i] = -beta[i, j]
                    end
                end

                allzero = true
                @inbounds for i in 1:ntraits
                    if dvec[i] != 0.0
                        allzero = false
                        break
                    end
                end
                allzero && continue

                scale = 1.0
                old_norm = norms[j]
                @inbounds for i in 1:ntraits
                    col = cols[i]
                    if col === nothing
                        xjd_active[i] = false
                    else
                        v = xjd_buf[i]
                        di = dvec[i]
                        for r in eachindex(col)
                            v[r] = col[r] * di
                        end
                        xjd_active[i] = true
                    end
                end

                accepted = false
                loss0 = loss_current
                accepted_loss = loss_current
                while scale > 1e-30
                    new_norm2 = 0.0
                    full_step_norm2 = 0.0
                    @inbounds for i in 1:ntraits
                        v = beta[i, j] + scale * dvec[i]
                        new_norm2 += v * v
                        vf = beta[i, j] + dvec[i]
                        full_step_norm2 += vf * vf
                    end
                    new_norm = sqrt(new_norm2)
                    full_step_norm = sqrt(full_step_norm2)
                    qh = 0.0
                    @inbounds for i in 1:ntraits
                        qh += grad[i] * dvec[i]
                    end
                    qh += lambda * sqrt(Float64(gsize[j])) * (full_step_norm - old_norm)

                    loss_test = 0.0
                    @inbounds for i in 1:ntraits
                        resid = residual_list[i]
                        if !xjd_active[i]
                            for r in eachindex(resid)
                                d = resid[r]
                                loss_test += d * d
                            end
                        else
                            dx = xjd_buf[i]
                            for r in eachindex(resid)
                                d = resid[r] - scale * dx[r]
                                loss_test += d * d
                            end
                        end
                    end
                    left = loss_test + lambda * sqrt(Float64(gsize[j])) * new_norm
                    right = loss0 + lambda * sqrt(Float64(gsize[j])) * old_norm + 0.1 * scale * qh
                    left <= right && (accepted = true; accepted_loss = loss_test; break)
                    scale *= 0.5
                end

                accepted || continue
                loss_current = accepted_loss
                @inbounds for i in 1:ntraits
                    beta[i, j] += scale * dvec[i]
                    xjd_active[i] || continue
                    dx = xjd_buf[i]
                    eta = eta_list[i]
                    resid = residual_list[i]
                    for r in eachindex(eta)
                        eta[r] += scale * dx[r]
                        resid[r] -= scale * dx[r]
                    end
                end
                s2 = 0.0
                @inbounds for i in 1:ntraits
                    s2 += beta[i, j] * beta[i, j]
                end
                norms[j] = sqrt(s2)
            end

            fn_val = loss_current + _pruning_group_path_penalty_value(beta, lambda, gsize)
            maxrel = 0.0
            @inbounds for idx in eachindex(beta)
                denom = 1.0 + abs(beta[idx])
                maxrel = max(maxrel, abs(beta[idx] - beta_old[idx]) / denom)
            end
            d_par = maxrel
            d_fn = abs(fn_old - fn_val) / (1.0 + abs(fn_val))
            if d_fn <= tol && d_par <= sqrt(tol)
                counter = 0
            end
        end
        on_solution !== nothing && on_solution(pos, beta)
    end
    return converged
end

function _pruning_group_path_fit_cached!(
    beta::Matrix{Float64},
    y_list::Vector{Vector{Float64}},
    x_cols::Vector{Vector{Union{Nothing, Vector{Float64}}}},
    lambdas::AbstractVector{<:Real};
    max_iterations::Integer = 500,
    tol::Float64 = 1e-6,
    group_sizes::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    on_solution::Union{Nothing, Function} = nothing,
)
    ntraits, ncandidates = size(beta)
    gsize = group_sizes === nothing ? fill(ntraits, ncandidates) : Int.(group_sizes)
    eta_list = [zeros(Float64, length(y)) for y in y_list]
    residual_list = [copy(y) for y in y_list]
    nH = fill(0.01, ncandidates)
    for j in 1:ncandidates
        hj = 0.01
        cols = x_cols[j]
        @inbounds for i in 1:ntraits
            col = cols[i]
            col === nothing && continue
            hj = max(hj, _pruning_group_path_hessian_col(col))
        end
        nH[j] = hj
    end

    converged = trues(length(lambdas))
    beta_old = similar(beta)
    beta_new = zeros(Float64, ntraits)
    dvec = zeros(Float64, ntraits)
    grad = zeros(Float64, ntraits)
    xjd_buf = [Vector{Float64}(undef, length(y_list[i])) for i in 1:ntraits]
    xjd_active = falses(ntraits)
    norms = zeros(Float64, ncandidates)

    for (pos, lambda0) in enumerate(lambdas)
        lambda = Float64(lambda0)
        loss_current = _pruning_group_path_loss_value(y_list, eta_list)
        fn_val = loss_current + _pruning_group_path_penalty_value(beta, lambda, gsize)
        d_fn = 1.0
        d_par = 1.0
        do_all = false
        counter = 1
        iter_count = 0
        fill!(norms, 0.0)
        @inbounds for j in 1:ncandidates
            s = 0.0
            for i in 1:ntraits
                s += beta[i, j] * beta[i, j]
            end
            norms[j] = sqrt(s)
        end

        while d_fn > tol || d_par > sqrt(tol) || !do_all
            if iter_count >= max_iterations
                converged[pos] = false
                break
            end
            fn_old = fn_val
            copyto!(beta_old, beta)
            guessed_active =
                if counter == 0 || counter > 10
                    do_all = true
                    counter = 1
                    1:ncandidates
                else
                    inds = findall(!iszero, norms)
                    if isempty(inds)
                        do_all = true
                        1:ncandidates
                    else
                        do_all = false
                        counter += 1
                        inds
                    end
                end
            do_all && (iter_count += 1)

            for j in guessed_active
                cols = x_cols[j]
                fill!(grad, 0.0)
                @inbounds for i in 1:ntraits
                    col = cols[i]
                    if col !== nothing
                        s = 0.0
                        resid = residual_list[i]
                        for r in eachindex(col)
                            s += col[r] * resid[r]
                        end
                        grad[i] = -2.0 * s
                    end
                end

                hj = nH[j]
                cond_norm2 = 0.0
                @inbounds for i in 1:ntraits
                    c = -grad[i] + hj * beta[i, j]
                    beta_new[i] = c
                    cond_norm2 += c * c
                end
                border = lambda * sqrt(Float64(gsize[j]))
                if cond_norm2 > border * border
                    cond_norm = sqrt(cond_norm2)
                    @inbounds for i in 1:ntraits
                        dvec[i] = (-grad[i] - border * beta_new[i] / cond_norm) / hj
                    end
                else
                    @inbounds for i in 1:ntraits
                        dvec[i] = -beta[i, j]
                    end
                end

                allzero = true
                @inbounds for i in 1:ntraits
                    if dvec[i] != 0.0
                        allzero = false
                        break
                    end
                end
                allzero && continue

                scale = 1.0
                old_norm = norms[j]
                @inbounds for i in 1:ntraits
                    col = cols[i]
                    if col === nothing
                        xjd_active[i] = false
                    else
                        v = xjd_buf[i]
                        di = dvec[i]
                        for r in eachindex(col)
                            v[r] = col[r] * di
                        end
                        xjd_active[i] = true
                    end
                end

                accepted = false
                loss0 = loss_current
                accepted_loss = loss_current
                while scale > 1e-30
                    new_norm2 = 0.0
                    full_step_norm2 = 0.0
                    @inbounds for i in 1:ntraits
                        v = beta[i, j] + scale * dvec[i]
                        new_norm2 += v * v
                        vf = beta[i, j] + dvec[i]
                        full_step_norm2 += vf * vf
                    end
                    new_norm = sqrt(new_norm2)
                    full_step_norm = sqrt(full_step_norm2)
                    qh = 0.0
                    @inbounds for i in 1:ntraits
                        qh += grad[i] * dvec[i]
                    end
                    qh += lambda * sqrt(Float64(gsize[j])) * (full_step_norm - old_norm)

                    loss_test = 0.0
                    @inbounds for i in 1:ntraits
                        resid = residual_list[i]
                        if !xjd_active[i]
                            for r in eachindex(resid)
                                d = resid[r]
                                loss_test += d * d
                            end
                        else
                            dx = xjd_buf[i]
                            for r in eachindex(resid)
                                d = resid[r] - scale * dx[r]
                                loss_test += d * d
                            end
                        end
                    end
                    left = loss_test + lambda * sqrt(Float64(gsize[j])) * new_norm
                    right = loss0 + lambda * sqrt(Float64(gsize[j])) * old_norm + 0.1 * scale * qh
                    left <= right && (accepted = true; accepted_loss = loss_test; break)
                    scale *= 0.5
                end

                accepted || continue
                loss_current = accepted_loss
                @inbounds for i in 1:ntraits
                    beta[i, j] += scale * dvec[i]
                    xjd_active[i] || continue
                    dx = xjd_buf[i]
                    eta = eta_list[i]
                    resid = residual_list[i]
                    for r in eachindex(eta)
                        eta[r] += scale * dx[r]
                        resid[r] -= scale * dx[r]
                    end
                end
                s2 = 0.0
                @inbounds for i in 1:ntraits
                    s2 += beta[i, j] * beta[i, j]
                end
                norms[j] = sqrt(s2)
            end

            fn_val = loss_current + _pruning_group_path_penalty_value(beta, lambda, gsize)
            maxrel = 0.0
            @inbounds for idx in eachindex(beta)
                denom = 1.0 + abs(beta[idx])
                maxrel = max(maxrel, abs(beta[idx] - beta_old[idx]) / denom)
            end
            d_par = maxrel
            d_fn = abs(fn_old - fn_val) / (1.0 + abs(fn_val))
            if d_fn <= tol && d_par <= sqrt(tol)
                counter = 0
            end
        end
        on_solution !== nothing && on_solution(pos, beta)
    end
    return converged
end

function _pruning_group_path_df_count(beta::AbstractMatrix{<:Real})
    ntraits, ncandidates = size(beta)
    nactive = 0
    @inbounds for j in 1:ncandidates
        c = 0
        for i in 1:ntraits
            beta[i, j] != 0.0 && (c += 1)
        end
        c > ntraits - 1 && (nactive += 1)
    end
    return nactive
end

function _pruning_group_path_adaptive_lambda_base_seq(max_shifts::Integer, lambda_max::Float64, fit_for_base_seq)
    delta = 1.0 / 16.0
    seq_ub = 5.0
    base_seq = collect(0.0:delta:seq_ub)
    lmax = 1.2 * lambda_max + 1.0
    for _ in 1:7
        df_vec, _ = fit_for_base_seq(base_seq, lmax)
        missing_df = setdiff(collect(0:(Int(max_shifts) + 1)), df_vec)
        if isempty(missing_df)
            over = findfirst(>(Int(max_shifts) + 4), df_vec)
            over !== nothing && (base_seq = base_seq[1:over])
            return base_seq, lmax
        end

        tmp = Float64[]
        idx = 1
        prev_idx = 1
        cut_extra = true
        for mdf in missing_df
            greater = findall(>(mdf), df_vec)
            if !isempty(greater)
                idx = minimum(greater)
                if idx == 1
                    lmax += 2.0
                    continue
                end
                idx == prev_idx && continue
                lower = base_seq[idx - 1] + delta / 4.0
                upper = base_seq[idx]
                append!(tmp, base_seq[prev_idx:idx])
                append!(tmp, collect(lower:(delta / 4.0):upper))
            else
                lower = isempty(tmp) ? 0.0 : tmp[end] + delta
                upper = max(lower, maximum(base_seq)) + 1.0
                append!(tmp, collect(lower:delta:upper))
                cut_extra = false
            end
            prev_idx = idx
        end
        if cut_extra
            over = findfirst(>(Int(max_shifts) + 4), df_vec)
            upper_idx = over === nothing ? length(base_seq) : over
            idx < upper_idx && append!(tmp, base_seq[idx:upper_idx])
        end
        delta /= 4.0
        base_seq = sort!(unique(tmp))
        isempty(base_seq) && return [0.0], lmax
    end
    return base_seq, lmax
end

