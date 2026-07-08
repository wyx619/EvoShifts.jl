@inline function compute_aic(loglik::Real, nparams::Integer)
    return 2.0 * nparams - 2.0 * Float64(loglik)
end

@inline function compute_aicc(loglik::Real, nparams::Integer, n::Integer)
    aic = compute_aic(loglik, nparams)
    denominator = n - nparams - 1
    denominator > 0 || return Inf
    return aic + (2.0 * nparams * (nparams + 1)) / denominator
end

@inline function compute_bic(loglik::Real, nparams::Integer, n::Integer)
    n > 0 || throw(ArgumentError("n must be positive for BIC"))
    return -2.0 * Float64(loglik) + nparams * log(Float64(n))
end
