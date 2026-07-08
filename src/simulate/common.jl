Base.@kwdef struct SimulatedTree
    parent::Vector{Int32}
    children::Vector{Vector{Int32}}
    node_time::Vector{Float64}
    is_tip::BitVector
    tip_labels::Vector{String}
end

mutable struct _MVGaussianStepWorkspace
    z::Vector{Float64}
    noise::Vector{Float64}
end

function _mvsim_gaussian_workspace(p::Integer)
    return _MVGaussianStepWorkspace(zeros(Float64, p), zeros(Float64, p))
end

@inline function _mvsim_add_chol_noise!(
    dest::AbstractVector{Float64},
    L::AbstractMatrix{Float64},
    scale::Float64,
    rng::AbstractRNG,
    ws::_MVGaussianStepWorkspace,
)
    scale == 0.0 && return dest
    scale >= 0.0 || throw(ArgumentError("Gaussian variance scale must be non-negative"))
    p = length(ws.z)
    if p == 1
        dest[1] += sqrt(scale) * L[1, 1] * randn(rng)
        return dest
    end
    randn!(rng, ws.z)
    mul!(ws.noise, L, ws.z)
    s = sqrt(scale)
    @inbounds for i in 1:p
        dest[i] += s * ws.noise[i]
    end
    return dest
end
