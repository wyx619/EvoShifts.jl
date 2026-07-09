using Test
using LinearAlgebra
using Random
using Statistics
using DataFrames

using EvoShifts

const SUBSET_TESTS = Dict(
    "core" => "core.jl",
    "proposal" => "proposal.jl",
    "refit" => "refit.jl",
    "api" => "api.jl",
    "convergence" => "convergence.jl",
)

function _normalize_subset_arg(arg::AbstractString)
    normalized = replace(arg, '\\' => '/')
    normalized = startswith(normalized, "test/") ? normalized[6:end] : normalized
    normalized = endswith(normalized, ".jl") ? normalized[1:end - 3] : normalized
    return normalized
end

if isempty(ARGS)
    println("Usage: julia --project=. test/run_subset.jl <subset> [<subset> ...]")
    println("Available subsets:")
    for name in sort!(collect(keys(SUBSET_TESTS)))
        println("  ", name)
    end
    exit(1)
end

for raw_arg in ARGS
    subset = _normalize_subset_arg(raw_arg)
    haskey(SUBSET_TESTS, subset) || error("Unknown test subset: $raw_arg")
    @testset "subset: $subset" begin
        include(joinpath(@__DIR__, SUBSET_TESTS[subset]))
    end
end
