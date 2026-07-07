Base.@kwdef struct OUParameterBundle
    theta::Vector{Float64} = Float64[]
    alpha::Vector{Float64} = Float64[]
    sigma2::Vector{Float64} = Float64[]
    theta0::Union{Nothing, Float64} = nothing
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

    alpha_init =
        if spec.alpha_mode === :shared
            [alpha_scalar]
        else
            fill(alpha_scalar, nregimes)
        end
    sigma_init =
        if spec.sigma_mode === :shared
            [sigma_scalar]
        else
            fill(sigma_scalar, nregimes)
        end

    theta_init =
        if spec.theta_mode === :shared
            [mean(observed_trait)]
        else
            fill(mean(observed_trait), nregimes)
        end

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
