"""
    pcaStructure{T}

Container for a fitted PCA model, as returned by `pca`
# Fields
- `mean::Vector{T}`: The column means of the training data (length p), removed
  during centering
- `scale::Vector{T}`: The column scales (length p) — the column standard
  deviations when `standardize=true`, otherwise ones
- `loadings::Matrix{T}`: The p×k principal directions (right singular vectors of
  the centered data / leading eigenvectors of the covariance matrix), unit-norm
  columns ordered by decreasing explained variance
- `variances::Vector{T}`: The variance explained by each of the k components
  (the eigenvalues, σᵢ² / (n-1))
- `propOFvar::Vector{T}`: The proportion of total variance explained by each
  component (variances ./ total variance)
"""
struct pcaStructure{T}
    mean::Vector{T}
    scale::Vector{T}      # column std devs used for scaling (ones if standardize=false)
    loadings::Matrix{T}
    variances::Vector{T}
    propOFvar::Vector{T}
end

"""
    pca_transform(m::pcaStructure, X::Matrix{Float64})

Project data onto the principal directions of a fitted PCA model
# Arguments
- `m::pcaStructure`: A fitted PCA model, as returned by `pca`
- `X::Matrix{Float64}`: 2d array of floats; the observations (rows) by features
  (columns) to project, with the same features as the training data
# Value
2d array of floats; the n×k matrix of principal-component scores
"""
function pca_transform(m::pcaStructure, X::Matrix{Float64})
    Xc = (X .- m.mean') ./ m.scale'    # center and scale using the stored stats
    return Xc * m.loadings             # project onto the principal directions
end

"""
    pca_invtransform(m::pcaStructure, scores::Matrix{Float64})

Reconstruct data in the original feature space from principal-component scores
# Arguments
- `m::pcaStructure`: A fitted PCA model, as returned by `pca`
- `scores::Matrix{Float64}`: 2d array of floats; the n×k matrix of
  principal-component scores to invert, with k matching the number of components
  retained in `m`
# Value
2d array of floats; the n×p reconstruction in the original units. Exact only
when all components are retained (k = p); otherwise a low-rank approximation
"""
function pca_invtransform(m::pcaStructure, scores::Matrix{Float64})
    Xc = scores * m.loadings'          # back to full feature width (centered space)
    return Xc .* m.scale' .+ m.mean'   # undo the scaling, then undo the centering
end






"""
    pca(X::Matrix{Float64}; k::Int = minimum(size(X)),
        standardize::Bool = false, method::Symbol = :svd)

Fit a principal component analysis (PCA) model
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; the observations (rows) by features
  (columns) data matrix
- `k::Int`: The number of principal components to retain; must satisfy
  1 ≤ k ≤ min(n, p). Defaults to min(n, p) (all components)
- `standardize::Bool`: Whether to scale each column to unit standard deviation
  in addition to centering. Defaults to false (center only)
- `method::Symbol`: The decomposition to use, either `:svd` (singular value
  decomposition of the centered data) or `:cov` (eigendecomposition of the
  covariance matrix). Defaults to `:svd`
# Value
A `pcaStructure` holding the column means, scales, k loadings (principal
directions), component variances, and proportion of variance explained
"""
function pca(X::Matrix{Float64}; k::Int = minimum(size(X)),
             standardize::Bool = false, method::Symbol = :svd)
    n, p = size(X)                                              # n observations, p features
    1 <= k <= min(n, p) || throw(ArgumentError("$k you chose is not in range. Please reselect $k"))
    means = vec(mean(X, dims = 1))                             # column means (p-vector)
    sigma = standardize ? vec(std(X, dims = 1)) : ones(eltype(means), p)  # column std devs, or ones if not standardizing
    Xc = (X .- means') ./ sigma'                              # center (and optionally scale) in one fused pass
    T  = eltype(Xc)
    total = sum(abs2, Xc) / (n - 1)                           # total variance (trace of the covariance matrix)

    if method === :cov
        C = Symmetric(Xc'Xc)                                  # p×p scatter; the /(n-1) is folded into vars below
        F = eigen(C)                                         # eigen returns eigenvalues in ASCENDING order
        idx = p:-1:(p-k+1)                                    # so the top k are the LAST k, taken in descending order
        vars = @view(F.values[idx]) ./ (n - 1)               # variances explained by the top k components
        V = F.vectors[:, idx]                                # loadings: the corresponding eigenvectors
        vars = collect(vars)
    elseif method === :svd
        F = svd(Xc; full = false)                            # thin SVD: U is n×min(n,p), not n×n
        vars = @view(F.S[1:k]) .^ 2 ./ (n - 1)               # variances = squared singular values / (n-1)
        V = F.V[:, 1:k]                                       # loadings: the top k right singular vectors
        vars = collect(vars)
    else
        error("unknown method :$method")
    end

    SignConsistency_opt!(V)                                   # fix arbitrary PC signs for reproducibility
    return pcaStructure{T}(T.(means), T.(sigma), Matrix(V), vars, vars ./ total)
end