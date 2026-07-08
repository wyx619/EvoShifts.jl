function _normalize_trait_columns(df::AbstractDataFrame, trait_cols)
    if trait_cols === nothing
        return Symbol.(names(df))
    elseif trait_cols isa Symbol || trait_cols isa AbstractString
        return [Symbol(trait_cols)]
    else
        return Symbol.(trait_cols)
    end
end

function _check_dataframe_columns(df::AbstractDataFrame, cols::AbstractVector{Symbol})
    for col in cols
        hasproperty(df, col) || throw(ArgumentError("data does not contain column `$col`"))
    end
    return nothing
end

function _is_missing_string(x::AbstractString)
    s = lowercase(strip(String(x)))
    return isempty(s) || s in ("na", "nan", "missing", "null")
end

function _aligned_trait_value(val, allow_missing::Bool)
    if ismissing(val)
        allow_missing || throw(ArgumentError("single-trait shift detection does not allow missing values"))
        return NaN
    elseif val isa AbstractString
        _is_missing_string(val) && begin
            allow_missing || throw(ArgumentError("single-trait shift detection does not allow missing values"))
            return NaN
        end
        parsed = tryparse(Float64, strip(String(val)))
        parsed === nothing && throw(ArgumentError("trait data contains a non-numeric value: `$val`"))
        return parsed
    else
        return Float64(val)
    end
end

function _aligned_dataframe_rows(tree::CompactTree, df::AbstractDataFrame, taxon_col)
    if taxon_col === nothing
        DataFrames.nrow(df) == tree.ntips ||
            throw(ArgumentError("data must have $(tree.ntips) rows when taxon_col is not supplied; rows are assumed to match tree.tip_labels order"))
        return collect(1:tree.ntips)
    end

    taxon = Symbol(taxon_col)
    _check_dataframe_columns(df, [taxon])
    row_by_tip = Dict{String, Int}()
    for row in 1:DataFrames.nrow(df)
        label = string(df[row, taxon])
        haskey(row_by_tip, label) && throw(ArgumentError("duplicate taxon label `$label` in data"))
        row_by_tip[label] = row
    end

    rows = Vector{Int}(undef, tree.ntips)
    missing_tips = String[]
    @inbounds for (i, label) in enumerate(tree.tip_labels)
        row = get(row_by_tip, label, 0)
        if row == 0
            push!(missing_tips, label)
        else
            rows[i] = row
        end
    end
    if !isempty(missing_tips)
        example = join(first(missing_tips, min(length(missing_tips), 5)), ", ")
        throw(ArgumentError("data is missing $(length(missing_tips)) tree tips, e.g. $example"))
    end
    return rows
end

function align_traits_to_tree(
    tree::CompactTree,
    data::AbstractDataFrame;
    taxon_col::Union{Nothing, Symbol, AbstractString} = nothing,
    trait_cols = nothing,
)
    df = DataFrames.DataFrame(data)
    cols = _normalize_trait_columns(df, trait_cols)
    if taxon_col !== nothing
        cols = [col for col in cols if col != Symbol(taxon_col)]
    end
    isempty(cols) && throw(ArgumentError("trait_cols selects no trait columns"))
    _check_dataframe_columns(df, cols)

    rows = _aligned_dataframe_rows(tree, df, taxon_col)
    values = Matrix(df[rows, cols])
    out = Matrix{Float64}(undef, size(values, 1), size(values, 2))
    allow_missing = size(values, 2) > 1
    @inbounds for j in axes(values, 2), i in axes(values, 1)
        out[i, j] = _aligned_trait_value(values[i, j], allow_missing)
    end
    any(isinf, out) && throw(ArgumentError("trait data contains infinite values"))
    if size(out, 2) == 1
        any(isnan, out) && throw(ArgumentError("single-trait shift detection does not allow missing values"))
    else
        @inbounds for i in axes(out, 1)
            all(isnan, @view(out[i, :])) && throw(ArgumentError("trait row $i contains no observed values"))
        end
        @inbounds for j in axes(out, 2)
            all(isnan, @view(out[:, j])) && throw(ArgumentError("trait column $j contains no observed values"))
        end
    end
    return size(out, 2) == 1 ? vec(out) : out
end

function align_traits_to_tree(tree::CompactTree, data::AbstractVector{<:Real}; kwargs...)
    isempty(kwargs) || throw(ArgumentError("keyword arguments are only supported for DataFrame input"))
    length(data) == tree.ntips || throw(ArgumentError("trait length must match tree.ntips"))
    out = Float64.(data)
    all(isfinite, out) || throw(ArgumentError("trait data contains non-finite values"))
    return out
end

function align_traits_to_tree(tree::CompactTree, data::AbstractMatrix{<:Real}; kwargs...)
    isempty(kwargs) || throw(ArgumentError("keyword arguments are only supported for DataFrame input"))
    size(data, 1) == tree.ntips || throw(ArgumentError("trait matrix row count must match tree.ntips"))
    out = Matrix{Float64}(data)
    any(isinf, out) && throw(ArgumentError("trait data contains infinite values"))
    @inbounds for i in axes(out, 1)
        all(isnan, @view(out[i, :])) && throw(ArgumentError("trait row $i contains no observed values"))
    end
    @inbounds for j in axes(out, 2)
        all(isnan, @view(out[:, j])) && throw(ArgumentError("trait column $j contains no observed values"))
    end
    return out
end
