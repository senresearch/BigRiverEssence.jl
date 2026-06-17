# Abhisek Banerjee
# splsda —  mixOmics' sPLS-DA (single-block path).
# Reproduces splsda(X, Y, ncomp, keepX): PLS NIPALS with L1 variable selection
# on X-loadings, regression deflation on X. as in the paper SPLSDA (MixOmics v6.20.0, 2011-2024) by Le Cao et al., J. Chemometrics 23, 149-155.



struct SplsdaResult{T}
    variates_X::Matrix{T}     # X scores (n × ncomp)
    variates_Y::Matrix{T}     # Y scores (n × ncomp)
    loadings_X::Matrix{T}     # X loadings (p × ncomp), sparse
    loadings_Y::Matrix{T}     # Y loadings (k × ncomp)
    ncomp::Int
    keepX::Vector{Int}
    Y_dummy::Matrix{T}        # the one-hot indicator
    classes::Vector           # class levels (in dummy-column order)
end

# one-hot encode a class vector (mixOmics' unmap). levels can be provided to specify the class order (and thus the dummy column order); if not provided, classes are sorted alphabetically.
function _unmap(y::AbstractVector; levels=nothing)
    classes = levels === nothing ? sort(unique(y)) : collect(levels)  # preserve order if levels provided, else sort
    length(classes) == length(unique(y)) ||
        throw(ArgumentError("`levels` must list each class exactly once (got $(length(classes)) levels for $(length(unique(y))) classes)"))
    k = length(classes)
    Yd = zeros(Float64, length(y), k)
    for (i, yi) in enumerate(y)
        idx = findfirst(==(yi), classes)  # it checks yi == classes[j] for j in 1:k, returns the first j where it's true (the dummy column index for class yi)
        idx === nothing && throw(ArgumentError("class $(yi) not found in supplied `levels`"))
        Yd[i, idx] = 1.0
    end
    return Yd, classes
end

# center + scale columns (scale = unbiased SD, n-1, matching the paper and mixOmics' default). Returns the centered+scaled matrix; the caller can keep the means and sds if needed for later.
function _center_scale(M::AbstractMatrix; scale=true)
    Mc = M .- mean(M, dims=1)
    if scale
        s = std(M, dims=1; corrected=true)        # n-1 SD, like colSds
        s[s .== 0] .= 1.0                           # avoid div by zero (mixOmics zeros those cols)
        Mc = Mc ./ s
        zerocols = vec(std(M, dims=1; corrected=true) .== 0)
        Mc[:, zerocols] .= 0.0
    end
    return Mc
end

# soft-thresholding scalar  S(a,Δ) = sign(a)·(|a|-Δ)₊   (unchanged, used elsewhere). Got this function from mixOmics' soft_thresholding_L1, which is in the code file internal_mint.block_helpers.R.
# it implements mixOmics' soft_thresholding_L1. Its job: given a loading vector x, keep only the keepX largest-magnitude entries (zeroing the rest) and shrink the kept ones. 
# The parameter nx is how many to drop (nx = p − keepX).
function _soft_threshold_L1(x::AbstractVector, nx::Int)
    nx <= 0 && return copy(x)  # if nx is zero or negative, we keep all entries, so just return a copy of x; this is a shortcut to avoid unnecessary computation when no sparsity is needed.
    absx = abs.(x)             # absolute values of the entries of x, which we need to determine which ones to keep based on their magnitude; this creates a temporary array, but it's necessary for the ranking step that follows; we will use absx to find the threshold for soft-thresholding.
    # rank with ties="max": an entry's rank = number of entries ≤ it.
    # keep entries whose rank > nx (the keepX largest).
    p = length(x)
    ord = sortperm(absx)  # indices that would sort absx in ascending order; the largest-magnitude entries will be at the end of this order, and we will use it to determine which entries to keep based on their rank.
    ranks = zeros(Int, p)
    # assign ranks with ties: we loop through the sorted indices, and for each group of entries with the same absolute value, we assign them all the same rank equal to the maximum rank in that group; this way, if there are ties in magnitude, they will either all be kept or all be dropped together.
    i = 1
    while i <= p
        j = i
        while j < p && absx[ord[j+1]] == absx[ord[i]]
            j += 1
        end
        for m in i:j; ranks[ord[m]] = j; end        # ties get the max rank
        i = j + 1
    end
    # keep entries whose rank > nx (the keepX largest); if all ranks are > nx, we can just return x; otherwise, we need to find the threshold for soft-thresholding,
    # which is the largest absolute value among the entries that are dropped (those with rank ≤ nx), and then apply the soft-thresholding formula to shrink the kept entries and zero out the rest.
    keep = ranks .> nx                               # TRUE = keep
    all(keep) && return copy(x)
    lambda = maximum(absx[.!keep])                   # largest dropped magnitude
    out = similar(x)
    # apply soft-thresholding: for kept entries, shrink by lambda; for dropped entries, set to zero; this implements the formula S(a, Δ) = sign(a) * max(abs(a) - Δ, 0), where Δ is the threshold lambda we just computed.
    for i in 1:p
        out[i] = keep[i] ? sign(x[i]) * (absx[i] - lambda) : 0.0  # if keep[i] is true, we keep and shrink the entry; if false, we set it to zero; this loop applies the soft-thresholding to each entry of x based on whether it is among the keepX largest in magnitude or not.
    end
    return out
end

l2norm(x) = x ./ sqrt(sum(abs2, x))

function splsda(X::AbstractMatrix, y::AbstractVector, ncomp::Int, keepX::Vector{Int}; # ncomp = number of components, keepX is how many variables (genes) each component is allowed to use — the sparsity level. It's a vector, one entry per component, because each component can have a different sparsity.
                scale=true, tol=1e-6, max_iter=100, levels=nothing) 
    n, p = size(X)
    Yd, classes = _unmap(y; levels=levels)
    k = size(Yd, 2)
    length(keepX) == ncomp || throw(ArgumentError("keepX must have length ncomp"))

    Xc = _center_scale(Matrix{Float64}(X); scale=scale)
    Yc = _center_scale(Yd; scale=scale)

    TX = zeros(n, ncomp); TY = zeros(n, ncomp)
    PX = zeros(p, ncomp); PY = zeros(k, ncomp)

    R = copy(Xc)                                     # X residual (deflated each comp)
    Ry = copy(Yc)                                    # Y residual (not deflated for DA)

    for comp in 1:ncomp
        #  init via SVD of XᵀY 
        M = R' * Ry
        F = svd(M)
        uh = F.U[:, 1]
        vh = F.V[:, 1]

        uh_old = copy(uh); vh_old = copy(vh)
        iter = 1
        while true
            tY = Ry * vh
            # block X: outer weight, sparsity, normalize
            uh = R' * tY
            uh = _soft_threshold_L1(uh, p - keepX[comp])
            uh = l2norm(uh)
            tX = R * uh
            # block Y: outer weight, normalize (no sparsity)
            vh = Ry' * tX
            vh = l2norm(vh)

            dX = sum(abs2, uh .- uh_old)
            dY = sum(abs2, vh .- vh_old)
            (max(dX, dY) < tol || iter > max_iter) && break
            uh_old = copy(uh); vh_old = copy(vh)
            iter += 1
        end

        tX = R * uh; tY = Ry * vh
        TX[:, comp] = tX; TY[:, comp] = tY
        PX[:, comp] = uh; PY[:, comp] = vh

        #  regression deflation of X by its own variate tX 
        pX = (R' * tX) / (tX' * tX)
        R = R .- tX * pX'
        # Y not deflated for DA (mode="regression")
    end

    return SplsdaResult(TX, TY, PX, PY, ncomp, keepX, Yd, classes)
end