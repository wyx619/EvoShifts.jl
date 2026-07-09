Base.@kwdef struct ContinuousNLoptResult
    minimizer::Vector{Float64} = Float64[]
    minimum::Float64 = Inf
    ret::Any = nothing
    iterations::Int = 0
end

Base.@kwdef struct ContinuousCompositeOptResult
    minimizer::Vector{Float64} = Float64[]
    minimum::Float64 = Inf
    converged::Bool = false
    iterations::Int = 0
    f_calls::Int = 0
end

resultminimizer(result::ContinuousNLoptResult) = result.minimizer
resultminimizer(result::ContinuousCompositeOptResult) = result.minimizer
resultminimizer(result) = Optim.minimizer(result)

resultminimum(result::ContinuousNLoptResult) = result.minimum
resultminimum(result::ContinuousCompositeOptResult) = result.minimum
resultminimum(result) = Optim.minimum(result)

resultconverged(result::ContinuousNLoptResult) =
    result.ret in (NLopt.SUCCESS, NLopt.STOPVAL_REACHED, NLopt.FTOL_REACHED, NLopt.XTOL_REACHED)
resultconverged(result::ContinuousCompositeOptResult) = result.converged
resultconverged(result) = Optim.converged(result)

resultiterations(result::ContinuousNLoptResult) = result.iterations
resultiterations(result::ContinuousCompositeOptResult) = result.iterations
resultiterations(result) = Optim.iterations(result)

resultfcalls(result::ContinuousNLoptResult) = result.iterations
resultfcalls(result::ContinuousCompositeOptResult) = result.f_calls
resultfcalls(result) = Optim.f_calls(result)

function optimizeobjective(
    objective,
    p0::AbstractVector{<:Real};
    method::Symbol,
    max_iterations::Integer,
    rel_tol::Float64,
    lower_bounds::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    init = Vector{Float64}(p0)

    if method === :LN_SBPLX
        opt = NLopt.Opt(:LN_SBPLX, length(init))
        opt.ftol_rel = rel_tol
        opt.xtol_rel = rel_tol
        opt.maxeval = Int(max_iterations)
        lower_bounds !== nothing && (opt.lower_bounds = Vector{Float64}(lower_bounds))
        opt.min_objective = (x, grad) -> begin
            value = objective(x)
            return isfinite(value) ? value : 1e300
        end
        minf, minx, ret = NLopt.optimize(opt, init)
        return ContinuousNLoptResult(minimizer = minx, minimum = minf, ret = ret, iterations = NLopt.numevals(opt))
    end

    if method === :L_BFGS
        xwork = copy(init)
        function grad!(G, x)
            fx = objective(x)
            if !isfinite(fx)
                fill!(G, 0.0)
                return G
            end
            copyto!(xwork, x)
            @inbounds for i in eachindex(x)
                xi = Float64(x[i])
                h = sqrt(eps(Float64)) * max(abs(xi), 1.0)
                xwork[i] = xi + h
                fplus = objective(xwork)
                if isfinite(fplus)
                    G[i] = (fplus - fx) / h
                else
                    xwork[i] = xi - h
                    if lower_bounds !== nothing && xwork[i] <= lower_bounds[i]
                        G[i] = 0.0
                    else
                        fminus = objective(xwork)
                        G[i] = isfinite(fminus) ? (fx - fminus) / h : 0.0
                    end
                end
                xwork[i] = xi
            end
            return G
        end
        od = Optim.OnceDifferentiable(objective, grad!, init)
        try
            return Optim.optimize(
                od,
                init,
                Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                Optim.Options(
                    iterations = Int(max_iterations),
                    f_reltol = rel_tol,
                    g_tol = 1e-4,
                    allow_f_increases = false,
                ),
            )
        catch err
            err isa AssertionError || rethrow()
            return optimizeobjective(
                objective,
                p0;
                method = :LN_SBPLX,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        end
    end

    throw(ArgumentError("Unsupported internal method=$method"))
end

function twostageresult(
    objective,
    p0::AbstractVector{<:Real};
    rough_iterations::Integer,
    polish_iterations::Integer,
    rel_tol::Float64,
    lower_bounds::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    rough = optimizeobjective(
        objective,
        p0;
        method = :LN_SBPLX,
        max_iterations = rough_iterations,
        rel_tol = rel_tol,
        lower_bounds = lower_bounds,
    )
    polish = optimizeobjective(
        objective,
        resultminimizer(rough);
        method = :L_BFGS,
        max_iterations = polish_iterations,
        rel_tol = min(rel_tol, 1e-7),
        lower_bounds = lower_bounds,
    )
    rough_min = Float64(resultminimum(rough))
    polish_min = Float64(resultminimum(polish))
    best = polish_min <= rough_min ? polish : rough
    return ContinuousCompositeOptResult(
        minimizer = Vector{Float64}(resultminimizer(best)),
        minimum = min(rough_min, polish_min),
        converged = resultconverged(rough) || resultconverged(polish),
        iterations = resultiterations(rough) + resultiterations(polish),
        f_calls = resultfcalls(rough) + resultfcalls(polish),
    )
end

function multistartserial(
    objective,
    candidates::Vector{Vector{Float64}};
    max_iterations::Integer,
    rel_tol::Float64,
    lower_bounds::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    isempty(candidates) && throw(ArgumentError("at least one initial candidate is required"))
    rough_iterations = Int(max_iterations)
    polish_iterations = 100

    best = twostageresult(
        objective,
        candidates[1];
        rough_iterations = rough_iterations,
        polish_iterations = polish_iterations,
        rel_tol = rel_tol,
        lower_bounds = lower_bounds,
    )
    for i in 2:length(candidates)
        current = twostageresult(
            objective,
            candidates[i];
            rough_iterations = rough_iterations,
            polish_iterations = polish_iterations,
            rel_tol = rel_tol,
            lower_bounds = lower_bounds,
        )
        current.minimum < best.minimum && (best = current)
    end
    return best
end
