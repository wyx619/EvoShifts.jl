Base.@kwdef struct OUSpec
    model::Symbol
    theta_mode::Symbol = :shared
    alpha_mode::Symbol = :shared
    sigma_mode::Symbol = :shared
    root_mean_mode::Symbol = :theta
    root_cov_mode::Symbol = :nonstationary
end

Base.@kwdef struct OUParameterBundle
    theta::Vector{Float64} = Float64[]
    alpha::Vector{Float64} = Float64[]
    sigma2::Vector{Float64} = Float64[]
    theta0::Union{Nothing, Float64} = nothing
end

struct OUMEdgeSegmentCache
    nregimes::Int
    root_regime::Int
    edge_first_segment::Vector{Int32}
    edge_last_segment::Vector{Int32}
    segment_states::Vector{Int32}
    segment_lengths::Vector{Float64}
end

function ou_spec(model::Symbol)
    if model === :OU1
        return OUSpec(model = :OU1, theta_mode = :shared, alpha_mode = :shared, sigma_mode = :shared, root_mean_mode = :theta, root_cov_mode = :fixed)
    elseif model === :OUM
        return OUSpec(model = :OUM, theta_mode = :by_regime, alpha_mode = :shared, sigma_mode = :shared, root_mean_mode = :root_regime_theta, root_cov_mode = :fixed)
    elseif model === :OUMV
        return OUSpec(model = :OUMV, theta_mode = :by_regime, alpha_mode = :shared, sigma_mode = :by_regime, root_mean_mode = :root_regime_theta, root_cov_mode = :fixed)
    elseif model === :OUMA
        return OUSpec(model = :OUMA, theta_mode = :by_regime, alpha_mode = :by_regime, sigma_mode = :shared, root_mean_mode = :root_regime_theta, root_cov_mode = :fixed)
    elseif model === :OUMVA
        return OUSpec(model = :OUMVA, theta_mode = :by_regime, alpha_mode = :by_regime, sigma_mode = :by_regime, root_mean_mode = :root_regime_theta, root_cov_mode = :fixed)
    end
    throw(ArgumentError("Unsupported OU model $model"))
end

@inline _ou_regime_value(mode::Symbol, values::Vector{Float64}, state::Integer) =
    mode === :shared ? values[1] : values[state]

function _ou_nparams(spec::OUSpec, nregimes::Integer)
    ntheta = spec.theta_mode === :shared ? 1 : nregimes
    nalpha = spec.alpha_mode === :shared ? 1 : nregimes
    nsigma = spec.sigma_mode === :shared ? 1 : nregimes
    nroot = spec.root_mean_mode === :free_theta0 ? 1 : 0
    return ntheta + nalpha + nsigma + nroot
end

function _ou_unpack_params(spec::OUSpec, pars::AbstractVector{<:Real}, nregimes::Integer)
    idx = 1
    nalphas = spec.alpha_mode === :shared ? 1 : nregimes
    nsigmas = spec.sigma_mode === :shared ? 1 : nregimes
    nthetas = spec.theta_mode === :shared ? 1 : nregimes

    length(pars) == nalphas + nsigmas + nthetas + (spec.root_mean_mode === :free_theta0 ? 1 : 0) ||
        throw(ArgumentError("Parameter vector length does not match OUSpec $(spec.model)"))

    alpha = max.(Float64.(pars[idx:(idx + nalphas - 1)]), 1e-8)
    idx += nalphas
    sigma2 = max.(Float64.(pars[idx:(idx + nsigmas - 1)]), 1e-8)
    idx += nsigmas
    theta = Float64.(pars[idx:(idx + nthetas - 1)])
    idx += nthetas
    theta0 = spec.root_mean_mode === :free_theta0 ? Float64(pars[idx]) : nothing
    return OUParameterBundle(theta = theta, alpha = alpha, sigma2 = sigma2, theta0 = theta0)
end

function _ou_initial_params(
    spec::OUSpec,
    nregimes::Integer;
    tree::CompactTree,
    trait::AbstractVector{<:Real},
)
    tree_height = maximum(tree.dist_from_root)
    observed_trait = filter(!isnan, Float64.(trait))
    alpha_scalar = max(log(2.0) / max(tree_height / 4.0, 1e-8), 1e-8)
    sigma_scalar = max(var(observed_trait), 1e-8)

    alpha_init = spec.alpha_mode === :shared ? [alpha_scalar] : fill(alpha_scalar, nregimes)
    sigma_init = spec.sigma_mode === :shared ? [sigma_scalar] : fill(sigma_scalar, nregimes)
    theta_init = spec.theta_mode === :shared ? [mean(observed_trait)] : fill(mean(observed_trait), nregimes)

    parts = Vector{Float64}()
    append!(parts, alpha_init)
    append!(parts, sigma_init)
    append!(parts, theta_init)
    if spec.root_mean_mode === :free_theta0
        push!(parts, mean(observed_trait))
    end
    return parts
end

function _ou_initial_params_from_starts(
    spec::OUSpec,
    nregimes::Integer;
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    start_alpha::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_sigma2::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_theta::Union{Nothing, Real} = nothing,
    start_theta_regimes::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    init = _ou_initial_params(spec, nregimes; tree = tree, trait = trait)
    idx = 1
    nalphas = spec.alpha_mode === :shared ? 1 : nregimes
    nsigmas = spec.sigma_mode === :shared ? 1 : nregimes
    nthetas = spec.theta_mode === :shared ? 1 : nregimes

    if start_alpha !== nothing
        if start_alpha isa AbstractVector
            length(start_alpha) == nalphas || throw(ArgumentError("start_alpha must have $nalphas entries"))
            init[idx:(idx + nalphas - 1)] .= max.(Float64.(start_alpha), 1e-8)
        else
            init[idx:(idx + nalphas - 1)] .= max(Float64(start_alpha), 1e-8)
        end
    end
    idx += nalphas

    if start_sigma2 !== nothing
        if start_sigma2 isa AbstractVector
            length(start_sigma2) == nsigmas || throw(ArgumentError("start_sigma2 must have $nsigmas entries"))
            init[idx:(idx + nsigmas - 1)] .= max.(Float64.(start_sigma2), 1e-8)
        else
            init[idx:(idx + nsigmas - 1)] .= max(Float64(start_sigma2), 1e-8)
        end
    end
    idx += nsigmas

    if spec.theta_mode === :shared
        start_theta !== nothing && (init[idx] = Float64(start_theta))
    elseif start_theta_regimes !== nothing
        length(start_theta_regimes) == nthetas || throw(ArgumentError("start_theta_regimes must have $nthetas entries"))
        init[idx:(idx + nthetas - 1)] .= Float64.(start_theta_regimes)
    end
    return init
end

@inline _ou_result_theta(spec::OUSpec, bundle::OUParameterBundle, root_state::Float64) =
    spec.theta_mode === :shared ? bundle.theta[1] : root_state
@inline _ou_result_alpha(spec::OUSpec, bundle::OUParameterBundle) =
    spec.alpha_mode === :shared ? bundle.alpha[1] : NaN
@inline _ou_result_alpha_regimes(spec::OUSpec, bundle::OUParameterBundle) =
    spec.alpha_mode === :shared ? Float64[] : copy(bundle.alpha)

@inline function _allow_zero_internal_edge_mismatch(tree::CompactTree, edge::Integer, segsum::Float64; atol::Float64 = 1e-8)
    child = Int(tree.child_of_edge[edge])
    tree.is_tip[child] && return false
    tree.edge_length[edge] == 0.0 || return false
    return abs(segsum) <= 10 * atol
end

function _validate_edge_segments(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}}; atol::Float64 = 1e-8)
    length(edge_segments) == tree.nedges || throw(ArgumentError("edge_segments must have $(tree.nedges) entries"))
    max_state = 0
    for edge in 1:tree.nedges
        segs = edge_segments[edge]
        isempty(segs) && throw(ArgumentError("edge_segments[$edge] is empty"))
        segsum = 0.0
        for seg in segs
            isfinite(seg.length) && seg.length >= 0.0 || throw(ArgumentError("edge_segments[$edge] contains invalid segment length"))
            seg.state >= 1 || throw(ArgumentError("edge_segments[$edge] contains invalid regime state"))
            segsum += seg.length
            max_state = max(max_state, Int(seg.state))
        end
        if !(isapprox(segsum, tree.edge_length[edge]; atol = atol) || _allow_zero_internal_edge_mismatch(tree, edge, segsum; atol = atol))
            throw(ArgumentError("edge_segments[$edge] lengths do not sum to branch length"))
        end
    end
    return max_state
end

function _root_regime_from_edge_segments(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}})
    root = Int(tree.root)
    first_edge = Int(tree.first_child_edge[root])
    first_edge > 0 || throw(ArgumentError("Root must have outgoing edges"))
    regime = Int(edge_segments[first_edge][1].state)
    for edge in tree.first_child_edge[root]:tree.last_child_edge[root]
        Int(edge_segments[edge][1].state) == regime || throw(ArgumentError("Root outgoing edges do not share a consistent initial regime"))
    end
    return regime
end

function _prepare_oum_edge_cache(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}}; atol::Float64 = 1e-8)
    nregimes = _validate_edge_segments(tree, edge_segments; atol = atol)
    root_regime = _root_regime_from_edge_segments(tree, edge_segments)
    edge_first_segment = fill(Int32(0), tree.nedges)
    edge_last_segment = fill(Int32(0), tree.nedges)
    segment_states = Int32[]
    segment_lengths = Float64[]

    for edge in 1:tree.nedges
        segs = edge_segments[edge]
        edge_first_segment[edge] = Int32(length(segment_states) + 1)
        for seg in segs
            push!(segment_states, seg.state)
            push!(segment_lengths, seg.length)
        end
        edge_last_segment[edge] = Int32(length(segment_states))
    end

    return OUMEdgeSegmentCache(
        nregimes,
        root_regime,
        edge_first_segment,
        edge_last_segment,
        segment_states,
        segment_lengths,
    )
end

@inline function _ou_edge_affine(cache, edge::Integer, spec::OUSpec, bundle::OUParameterBundle)
    a = 1.0
    b = 0.0
    v = 0.0
    first_seg = Int(cache.edge_first_segment[edge])
    last_seg = Int(cache.edge_last_segment[edge])
    @inbounds for seg_idx in first_seg:last_seg
        state = Int(cache.segment_states[seg_idx])
        seg_length = cache.segment_lengths[seg_idx]
        alpha = _ou_regime_value(spec.alpha_mode, bundle.alpha, state)
        sigma2 = _ou_regime_value(spec.sigma_mode, bundle.sigma2, state)
        theta = _ou_regime_value(spec.theta_mode, bundle.theta, state)
        phi = exp(-alpha * seg_length)
        a = phi * a
        b = phi * b + (1.0 - phi) * theta
        v = phi^2 * v + sigma2 * (1.0 - phi^2) / (2.0 * alpha)
    end
    return (a = a, b = b, v = v)
end

function _build_ou_edges(tree::CompactTree, spec::OUSpec, bundle::OUParameterBundle; cache = nothing)
    edge_a = zeros(Float64, tree.nedges)
    edge_b = zeros(Float64, tree.nedges)
    edge_v = zeros(Float64, tree.nedges)

    if cache === nothing
        spec.model === :OU1 || throw(ArgumentError("cache is required for $(spec.model)"))
        alpha = bundle.alpha[1]
        sigma2 = bundle.sigma2[1]
        theta = bundle.theta[1]
        edge_a .= exp.(-alpha .* tree.edge_length)
        edge_b .= (1.0 .- edge_a) .* theta
        edge_v .= sigma2 .* (1.0 .- edge_a .^ 2) ./ (2.0 * alpha)
    else
        for edge in 1:tree.nedges
            aff = _ou_edge_affine(cache, edge, spec, bundle)
            edge_a[edge] = aff.a
            edge_b[edge] = aff.b
            edge_v[edge] = aff.v
        end
    end

    return (edge_a = edge_a, edge_b = edge_b, edge_v = edge_v)
end

function _ou_root_prior(spec::OUSpec, bundle::OUParameterBundle; cache = nothing)
    mean =
        if spec.root_mean_mode === :theta
            bundle.theta[1]
        elseif spec.root_mean_mode === :root_regime_theta
            cache === nothing && throw(ArgumentError("cache is required for root_regime_theta"))
            bundle.theta[cache.root_regime]
        elseif spec.root_mean_mode === :free_theta0
            bundle.theta0 === nothing && throw(ArgumentError("theta0 is required for free_theta0"))
            bundle.theta0
        else
            throw(ArgumentError("Unsupported OU root_mean_mode=$(spec.root_mean_mode)"))
        end

    if spec.root_cov_mode === :stationary
        root_state = cache === nothing ? 1 : cache.root_regime
        alpha = _ou_regime_value(spec.alpha_mode, bundle.alpha, root_state)
        sigma2 = _ou_regime_value(spec.sigma_mode, bundle.sigma2, root_state)
        return (mean = mean, var = sigma2 / (2.0 * alpha), profile_root = false)
    elseif spec.root_cov_mode === :fixed
        return (mean = mean, var = 0.0, profile_root = false)
    elseif spec.root_cov_mode === :nonstationary
        return (mean = mean, var = Inf, profile_root = true)
    elseif spec.root_cov_mode === :free
        return (mean = mean, var = Inf, profile_root = true)
    end
    throw(ArgumentError("Unsupported OU root_cov_mode=$(spec.root_cov_mode)"))
end

function _ou_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec,
    bundle::OUParameterBundle;
    cache = nothing,
)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    edges = _build_ou_edges(tree, spec, bundle; cache = cache)
    root = _ou_root_prior(spec, bundle; cache = cache)
    prof = _linear_gaussian_loglik(
        tree,
        trait,
        edges.edge_a,
        edges.edge_b,
        edges.edge_v;
        root_prior_mean = root.mean,
        root_prior_var = root.var,
        profile_root = root.profile_root,
    )
    return (
        success = prof.success,
        loglik = prof.loglik,
        root_state = prof.root_state,
        edges = edges,
        root = root,
    )
end

function _ou_optimize_from_initial(
    objective,
    init::AbstractVector{<:Real},
    spec::OUSpec,
    nregimes::Integer,
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    cache;
    max_iterations::Integer,
    rel_tol::Float64,
    lower_bounds::AbstractVector{<:Real},
)
    if spec.model === :OU1
        return optimizeobjective(
            objective,
            init;
            method = :L_BFGS,
            max_iterations = max_iterations,
            rel_tol = rel_tol,
            lower_bounds = lower_bounds,
        )
    end
    candidates = _ou_multistart_candidates(spec, init, nregimes, tree, trait, cache)
    return multistartserial(
        objective,
        candidates;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = lower_bounds,
    )
end

function _ou_fit(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec;
    cache = nothing,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-6,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)

    nregimes = cache === nothing ? 1 : cache.nregimes
    init = _ou_initial_params(spec, nregimes; tree = tree, trait = tr)
    objective = function (par)
        bundle = _ou_unpack_params(spec, par, nregimes)
        prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
        return prof.success ? -prof.loglik : Inf
    end
    lower_bounds = vcat(
        fill(1e-8, spec.alpha_mode === :shared ? 1 : nregimes),
        fill(1e-8, spec.sigma_mode === :shared ? 1 : nregimes),
        fill(-Inf, spec.theta_mode === :shared ? 1 : nregimes),
        spec.root_mean_mode === :free_theta0 ? [-Inf] : Float64[],
    )
    result = _ou_optimize_from_initial(
        objective,
        init,
        spec,
        nregimes,
        tree,
        tr,
        cache;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = lower_bounds,
    )

    minimizer = resultminimizer(result)
    bundle = _ou_unpack_params(spec, minimizer, nregimes)
    prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
    return (bundle = bundle, profile = prof, result = result, nregimes = nregimes)
end

function _ou_fit_with_starts(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec;
    cache = nothing,
    method::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    start_alpha::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_sigma2::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_theta::Union{Nothing, Real} = nothing,
    start_theta_regimes::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)

    nregimes = cache === nothing ? 1 : cache.nregimes
    init = _ou_initial_params_from_starts(
        spec,
        nregimes;
        tree = tree,
        trait = tr,
        start_alpha = start_alpha,
        start_sigma2 = start_sigma2,
        start_theta = start_theta,
        start_theta_regimes = start_theta_regimes,
    )
    objective = function (par)
        bundle = _ou_unpack_params(spec, par, nregimes)
        prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
        return prof.success ? -prof.loglik : Inf
    end
    lower_bounds = vcat(
        fill(1e-8, spec.alpha_mode === :shared ? 1 : nregimes),
        fill(1e-8, spec.sigma_mode === :shared ? 1 : nregimes),
        fill(-Inf, spec.theta_mode === :shared ? 1 : nregimes),
        spec.root_mean_mode === :free_theta0 ? [-Inf] : Float64[],
    )
    result =
        if method in (:SBPLX_L_BFGS, :LN_SBPLX_L_BFGS, :TWO_STAGE)
            _ou_optimize_from_initial(
                objective,
                init,
                spec,
                nregimes,
                tree,
                tr,
                cache;
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        elseif method === :LN_SBPLX
            optimizeobjective(
                objective,
                init;
                method = :LN_SBPLX,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        else
            method === :L_BFGS || throw(ArgumentError("Unsupported internal method=$method"))
            optimizeobjective(
                objective,
                init;
                method = :L_BFGS,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        end

    minimizer = resultminimizer(result)
    bundle = _ou_unpack_params(spec, minimizer, nregimes)
    prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
    return (bundle = bundle, profile = prof, result = result, nregimes = nregimes)
end
