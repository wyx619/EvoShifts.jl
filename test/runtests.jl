using Test
using LinearAlgebra
using Random
using Statistics
using DataFrames

include(joinpath(@__DIR__, "..", "src", "EvoShifts.jl"))
using .EvoShifts

include("continuous/shifts/core.jl")
include("continuous/shifts/proposal.jl")
include("continuous/shifts/fit_and_ic.jl")
include("continuous/shifts/missing.jl")
include("continuous/shifts/detection_and_convergence.jl")
