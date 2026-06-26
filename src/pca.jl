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
    scale::Vector{T}
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
        standardize::Bool = false, method::Symbol = :auto)

Fit a principal component analysis (PCA) model
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; observations (rows) by features (columns)
- `k::Int`: The number of components to retain; 1 ≤ k ≤ min(n,p). Defaults to min(n,p)
- `standardize::Bool`: Whether to scale columns to unit variance in addition to
  centering. Defaults to false (center only)
- `method::Symbol`: `:auto` (pick by shape — `:cov` when n ≥ p, `:svd` when p > n),
  `:cov` (eigendecomposition of the p×p covariance), or `:svd` (SVD of the centered
  data). Defaults to `:auto`
# Value
A `pcaStructure` holding the column means, scales, k loadings, component variances,
and proportion of variance explained
"""
function pca(X::Matrix{Float64}; k::Int = minimum(size(X)),
             standardize::Bool = false, method::Symbol = :auto)
    n, p = size(X)
    1 <= k <= min(n, p) || throw(ArgumentError("k=$k must be in 1:min(n,p)=$(min(n,p))"))

    # choose the cheaper decomposition by shape: tall ⇒ small p×p covariance,
    # wide ⇒ SVD of the data matrix
    if method === :auto
        method = n >= p ? :cov : :svd
    end
    method in (:cov, :svd) || throw(ArgumentError("method must be :auto, :cov, or :svd, got :$method"))

    means = vec(mean(X, dims = 1))                          # column means (length p)
    sigma = standardize ? vec(std(X, dims = 1)) : Float64[] # empty sentinel = "no scaling"

    if method === :cov
        # Build the p×p scatter S = Xcᵀ Xc WITHOUT materializing a centered copy.
        # Xcᵀ Xc = Xᵀ X − n·(means meansᵀ), a rank-1 correction to the raw Gram.
        Smat = mul!(Matrix{Float64}(undef, p, p), transpose(X), X)   # Xᵀ X
        @inbounds for j in 1:p, i in 1:p
            Smat[i, j] -= n * means[i] * means[j]          # subtract n·μμᵀ ⇒ centered scatter
        end
        if standardize                                     # fold in column scaling: S ← D⁻¹ S D⁻¹
            @inbounds for j in 1:p, i in 1:p
                Smat[i, j] /= (sigma[i] * sigma[j])
            end
        end
        Ssym = Symmetric(Smat)

        total = tr(Ssym) / (n - 1)                         # total variance = trace, no extra data pass
        E = eigen(Ssym, p - k + 1 : p)                     # only the top k eigenpairs
        vars = reverse(E.values) ./ (n - 1)                # reorder to descending
        V = E.vectors[:, k:-1:1]                           # loadings, descending
    else  # :svd  — center, then SVD in the orientation LAPACK prefers (tall ≥ wide)
        Xc = standardize ? (X .- means') ./ vec(std(X, dims = 1))' : X .- means'
        if p > n
            # wide data: SVD the TALL transpose Xcᵀ (p×n). LAPACK's SVD is faster on
            # tall matrices; the loadings we want are then the LEFT singular vectors
            # of Xcᵀ (= right singular vectors of Xc).
            F = svd!(permutedims(Xc))                      # Xcᵀ is p×n (tall)
            s = @view F.S[1:k]
            vars = collect(s .^ 2 ./ (n - 1))
            V = Matrix(@view F.U[:, 1:k])                  # ← U (transpose ⇒ left vecs are the loadings)
            total = sum(abs2, F.S) / (n - 1)
        else
            # tall (or square) data: SVD Xc directly; loadings are the right singular vectors
            F = svd!(Xc)
            s = @view F.S[1:k]
            vars = collect(s .^ 2 ./ (n - 1))
            V = Matrix(@view F.V[:, 1:k])
            total = sum(abs2, F.S) / (n - 1)
        end
    end

    SignConsistency_opt!(V)                                # fix arbitrary PC signs for reproducibility
    scale_out = standardize ? sigma : ones(Float64, p)     # store ones when not standardizing
    return pcaStructure{Float64}(means, scale_out, V, vars, vars ./ total)
end