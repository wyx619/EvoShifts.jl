@inline function _gaussian_product(mean1::Float64, var1::Float64, mean2::Float64, var2::Float64)
    if isinf(var1)
        return (mean = mean2, var = var2)
    elseif isinf(var2)
        return (mean = mean1, var = var1)
    elseif var1 == 0.0 && var2 == 0.0
        abs(mean1 - mean2) <= 1e-8 || return (mean = NaN, var = NaN)
        return (mean = 0.5 * (mean1 + mean2), var = 0.0)
    elseif var1 == 0.0
        return (mean = mean1, var = 0.0)
    elseif var2 == 0.0
        return (mean = mean2, var = 0.0)
    end

    invvar = 1.0 / var1 + 1.0 / var2
    var = 1.0 / invvar
    mean = var * (mean1 / var1 + mean2 / var2)
    return (mean = mean, var = var)
end

@inline function _edge_message_to_parent(child_mean::Float64, child_var::Float64, a::Float64, b::Float64, v::Float64)
    (a > 0.0 && isfinite(v) && v >= 0.0) || return (mean = NaN, var = NaN)
    if a < 1.0e-150
        return (mean = 0.0, var = Inf)
    end
    total_var = child_var + v
    parent_var = total_var / (a * a)
    isfinite(parent_var) || return (mean = 0.0, var = Inf)
    return (mean = (child_mean - b) / a, var = parent_var)
end

@inline function _edge_predict_to_child(parent_mean::Float64, parent_var::Float64, a::Float64, b::Float64, v::Float64)
    (a > 0.0 && isfinite(v) && v >= 0.0) || return (mean = NaN, var = NaN)
    if isinf(parent_var)
        return (mean = 0.0, var = Inf)
    end
    return (mean = a * parent_mean + b, var = a * a * parent_var + v)
end

@inline function _scalar_observation_info_to_parent(y::Float64, a::Float64, b::Float64, v::Float64)
    (a > 0.0 && isfinite(v) && v >= 0.0) || return (success = false, precision = 0.0, linear = 0.0, logconst = -Inf)
    isnan(y) && return (success = true, precision = 0.0, linear = 0.0, logconst = 0.0)
    vv = max(v, 1e-12)
    resid = y - b
    invv = 1.0 / vv
    return (
        success = true,
        precision = a * a * invv,
        linear = a * resid * invv,
        logconst = -0.5 * (resid * resid * invv + log(vv) + log(2 * pi)),
    )
end

@inline function _scalar_info_to_parent(
    child_precision::Float64,
    child_linear::Float64,
    child_logconst::Float64,
    a::Float64,
    b::Float64,
    v::Float64,
)
    (a > 0.0 && isfinite(v) && v >= 0.0 && child_precision >= 0.0) ||
        return (success = false, precision = 0.0, linear = 0.0, logconst = -Inf)
    vv = max(v, 1e-12)
    q = 1.0 / vv
    T = child_precision + q
    (isfinite(T) && T > 0.0) || return (success = false, precision = 0.0, linear = 0.0, logconst = -Inf)
    r = child_linear + q * b
    precision = a * a * (q - q * q / T)
    linear = a * q * (r / T - b)
    logconst = child_logconst - 0.5 * log(vv) - 0.5 * log(T) + 0.5 * r * r / T - 0.5 * q * b * b
    return (success = true, precision = max(precision, 0.0), linear = linear, logconst = logconst)
end

@inline function _scalar_root_info_loglik(
    precision::Float64,
    linear::Float64,
    logconst::Float64,
    root_prior_mean::Float64,
    root_prior_var::Float64,
    profile_root::Bool,
)
    if profile_root || isinf(root_prior_var)
        precision > 0.0 || return (success = false, loglik = -Inf, root_state = NaN)
        root_state = linear / precision
        loglik = logconst + 0.5 * linear * root_state + 0.5 * log(2 * pi / precision)
        return (success = isfinite(loglik), loglik = loglik, root_state = root_state)
    elseif root_prior_var <= 1e-12
        root_state = root_prior_mean
        loglik = logconst - 0.5 * precision * root_state * root_state + linear * root_state
        return (success = isfinite(loglik), loglik = loglik, root_state = root_state)
    else
        q = 1.0 / root_prior_var
        T = precision + q
        T > 0.0 || return (success = false, loglik = -Inf, root_state = NaN)
        b = linear + q * root_prior_mean
        root_state = b / T
        loglik =
            logconst - 0.5 * root_prior_mean * root_prior_mean * q -
            0.5 * log(root_prior_var) - 0.5 * log(T) + 0.5 * b * root_state
        return (success = isfinite(loglik), loglik = loglik, root_state = root_state)
    end
end
