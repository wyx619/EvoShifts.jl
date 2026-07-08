Base.@kwdef struct OUSpec
    model::Symbol
    theta_mode::Symbol = :shared
    alpha_mode::Symbol = :shared
    sigma_mode::Symbol = :shared
    root_mean_mode::Symbol = :theta
    root_cov_mode::Symbol = :nonstationary
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

