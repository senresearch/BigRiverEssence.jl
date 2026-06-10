# Abhisek Banerjee
# PLS regression — Dayal & MacGregor (1997) "Improved PLS algorithms",
# J. Chemometrics 11, 73-85.  Improved kernel algorithms #1 (default) and #2.
# Algorithm is in page 77 and 76 of the paper (eqs. 28-38).
#   :alg1 — never forms XᵀX; uses X directly for the loadings (eqs. 31-33).
#   :alg2 — forms XᵀX once, works through it; no t vector formed (eqs. 34-35).

# result struct: weights W, X-loadings P, Y-loadings Q, score-weights R, scores T,
# plus the centering/scaling info needed to predict on new data.

struct Plsr{T}
    W::Matrix{T}      # X-weights              (p × nlv)  nlv: number of latent variables or how many components to include
    P::Matrix{T}      # X-loadings             (p × nlv)
    Q::Matrix{T}      # Y-loadings             (q × nlv)
    R::Matrix{T}      # weights to get scores straight from X  (p × nlv)
    T::Matrix{T}      # X-scores               (n × nlv)
    xmeans::Vector{T}
    xscales::Vector{T}
    ymeans::Vector{T}
    yscales::Vector{T}
end

# normalize a vector to unit length
_normalize(v) = v ./ norm(v)

"""
    pls(X, Y; nlv, standardize = false, method = :algo1)

PLS regression with the Dayal & MacGregor (1997) improved kernel algorithm.
- `X` (n×p), `Y` (n×q): predictors and responses (Y may be a vector or matrix).
- `nlv`: number of latent variables (components).
- `standardize`: scale each column of X and Y to unit std if true.
- `method`: `:algo1` (default, no XᵀX) or `:algo2` (forms XᵀX once).
"""
function plskern(X, Y; nlv = 2, standardize = false, method = :algo1)
    X = Matrix{Float64}(X)
    Y = Y isa AbstractVector ? reshape(Float64.(Y), :, 1) : Matrix{Float64}(Y)  # ensure Y is a matrix
    n, p = size(X)
    q    = size(Y, 2)
    nlv  = min(nlv, n, p)  # limit nlv to the minimum of n, p, and the provided nlv
    method in (:algo1, :algo2) || throw(ArgumentError("method must be :algo1 or :algo2, you entered :$method"))
    T_ = Float64

    # center (and optionally scale) X and Y, like our pca 
    xmeans = vec(mean(X, dims = 1))
    ymeans = vec(mean(Y, dims = 1))
    xscales = standardize ? vec(std(X, dims = 1)) : ones(T_, p)
    yscales = standardize ? vec(std(Y, dims = 1)) : ones(T_, q)
    Xc = (X .- xmeans') ./ xscales'
    Yc = (Y .- ymeans') ./ yscales'

    # the cross-product that drives everything; computed once (step 1) 
    XtY = Xc' * Yc                       # p × q
    XtX = method === :algo2 ? Xc' * Xc : nothing   # p × p, only for algo2 (step 1, #2)

    # pre-allocate result matrices 
    W = zeros(T_, p, nlv)
    P = zeros(T_, p, nlv)
    Q = zeros(T_, q, nlv)
    R = zeros(T_, p, nlv)
    Tt = zeros(T_, n, nlv)

    for a in 1:nlv                       # loop over the number of latent variables
        #  step 2: X-weight w 
        if q == 1
            w = XtY[:, 1]                # single Y: w ∝ XᵀY directly (appendix, M==1) pg 83, 84
        else
            # multiple Y: w from top eigenvector of XᵀY YᵀX  (eq. 11), via SVD
            w = svd(XtY).U[:, 1] # we chose XtY instead of the full XᵀY YᵀX for as they have same eigenvalues
        end
        w = _normalize(w)               # eq. 29

        #  step 3: r = w orthogonalized against previous components (eq. 30)
        r = copy(w)
        for j in 1:(a - 1)
            r .-= dot(P[:, j], w) .* R[:, j]
        end

        #  step 4 + tt. Computed for the two algorithms here 
        if method === :algo1
            # 4(a): use X directly (eqs. 31-33)
            t  = Xc * r                  # eq. 31:  t = X r
            tt = dot(t, t)               #          tt = tᵀt
            p_ = (Xc' * t) ./ tt         # eq. 32:  p = Xᵀt / tt
            Tt[:, a] = t                 # store the score (alg1 has t)
        else
            # 4(b): use XᵀX computed before (eqs. 34-35)
            XtXr = XtX * r               # compute XᵀX * r
            tt   = dot(r, XtXr)          # tᵀt = rᵀ(XᵀX)r   
            p_   = XtXr ./ tt            # eq. 34:  p = rᵀ(XᵀX) / tt
            Tt[:, a] = Xc * r            # recover the score for output (cheap, once)
        end
        q_ = (XtY' * r) ./ tt            # eq. 33 / 35:  q = (rᵀ XᵀY)ᵀ / tt  (same in both)

        # step 5: Update the covariance matrix XtY by deflating ONLY XᵀY (eq. 36) — the paper's key result 
        XtY .-= (p_ * q_') .* tt

        #  step 6: store the results
        W[:, a] = w
        P[:, a] = p_
        Q[:, a] = q_
        R[:, a] = r
    end

    return Plsr{T_}(W, P, Q, R, Tt, xmeans, xscales, ymeans, yscales)
end

# regression coefficients B (p × q) and intercept, in original units (eq. 38)
function plskerncoef(m::Plsr; nlv = size(m.R, 2))
    nlv = min(nlv, size(m.R, 2))
    B = (m.R[:, 1:nlv] ./ m.xscales) * (m.Q[:, 1:nlv]') .* m.yscales'   # R Qᵀ, rescaled
    intercept = m.ymeans' .- m.xmeans' * B  # Y = Ymeans + (X - Xmeans) * B = (Ymeans - Xmeans * B) + Xmeans * B
    return B, intercept
end

# predict Y for new X
function plskernpredict(m::Plsr, Xnew; nlv = size(m.R, 2))
    Xnew = Matrix{Float64}(Xnew)
    B, intercept = plskerncoef(m; nlv = nlv)
    return intercept .+ Xnew * B    # Ŷ = intercept .+ Xnew * B
end

# project new X onto the latent variables (scores): T = Xc * R  (eq. 20)
function plskerntransform(m::Plsr, Xnew; nlv = size(m.R, 2))
    Xnew = Matrix{Float64}(Xnew)              # ensure concrete Float64 matrix
    nlv  = min(nlv, size(m.R, 2))             # don't ask for more components than exist
    Xc   = (Xnew .- m.xmeans') ./ m.xscales'  # center and scale using the STORED stats
    return Xc * m.R[:, 1:nlv]                 # T = Xc * R  (eq. 20)
end