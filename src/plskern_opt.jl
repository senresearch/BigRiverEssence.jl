# Abhisek Banerjee
# PLS regression — Dayal & MacGregor (1997), OPTIMIZED version.
# Optimizations: preallocated buffers + mul! for the per-component products,
# @view for column slicing, BLAS.ger! for the rank-1 XtY deflation.
# Note: PLS is largely compute-bound (the matrix products dominate), so the
# main gains here are in allocations/memory; time is near the algorithmic floor.

using LinearAlgebra, Statistics

# (Plsr struct and _normalize are defined in pls.jl — not redefined here.)

"""
    plskern_opt(X, Y; nlv, standardize = false, method = :algo1)

Optimized PLS regression (Dayal & MacGregor 1997 improved kernel algorithm).
Identical results to `plskern`; preallocated buffers and in-place BLAS.
"""
function plskern_opt(X, Y; nlv = 2, standardize = false, method = :algo1)
    X = Matrix{Float64}(X)
    Y = Y isa AbstractVector ? reshape(Float64.(Y), :, 1) : Matrix{Float64}(Y)
    n, p = size(X)
    q    = size(Y, 2)
    nlv  = min(nlv, n, p)
    method in (:algo1, :algo2) || throw(ArgumentError("method must be :algo1 or :algo2, you entered :$method"))
    T_ = Float64

    # center (and optionally scale) X and Y
    xmeans  = vec(mean(X, dims = 1))
    ymeans  = vec(mean(Y, dims = 1))
    xscales = standardize ? vec(std(X, dims = 1)) : ones(T_, p)
    yscales = standardize ? vec(std(Y, dims = 1)) : ones(T_, q)
    Xc = (X .- xmeans') ./ xscales'
    Yc = (Y .- ymeans') ./ yscales'

    # cross-products computed once
    XtY = Xc' * Yc                                   # p × q
    XtX = method === :algo2 ? Xc' * Xc : nothing     # p × p, only for algo2

    # result matrices
    W  = zeros(T_, p, nlv)
    P  = zeros(T_, p, nlv)
    Q  = zeros(T_, q, nlv)
    R  = zeros(T_, p, nlv)
    Tt = zeros(T_, n, nlv)

    # preallocated working buffers (reused every component)
    w    = Vector{T_}(undef, p)
    r    = Vector{T_}(undef, p)
    t    = Vector{T_}(undef, n)
    pbuf = Vector{T_}(undef, p)
    qbuf = Vector{T_}(undef, q)

    Xct = transpose(Xc)                               # lazy transpose, no copy

    for a in 1:nlv
        # --- step 2: X-weight w ---
        if q == 1
            @views w .= XtY[:, 1]                      # single Y: w ∝ XᵀY
        else
            w .= svd(XtY).U[:, 1]                      # multi-Y: leading left singular vector
        end
        w ./= norm(w)                                  # eq. 29

        # --- step 3: r = w orthogonalized against previous components (eq. 30) ---
        copyto!(r, w)
        for j in 1:(a - 1)
            r .-= dot(@view(P[:, j]), w) .* @view(R[:, j])
        end

        # --- step 4 + score ---
        if method === :algo1
            mul!(t, Xc, r)                             # eq. 31: t = Xc*r
            tt = dot(t, t)
            mul!(pbuf, Xct, t); pbuf ./= tt            # eq. 32: p = Xc'*t / tt
            @views Tt[:, a] .= t
        else
            mul!(pbuf, XtX, r)                         # pbuf = XtX*r
            tt = dot(r, pbuf)                          # tt = rᵀ(XtX)r
            pbuf ./= tt                                # eq. 34: p = rᵀ(XᵀX) / tt
            mul!(t, Xc, r); @views Tt[:, a] .= t       # recover score for output
        end
        mul!(qbuf, transpose(XtY), r); qbuf ./= tt     # eq. 33/35: q = (rᵀ XᵀY)ᵀ / tt

        # --- step 5: deflate XtY by rank-1 update (avoid forming p_*q_') ---
        BLAS.ger!(-tt, pbuf, qbuf, XtY)                # XtY += (-tt)·pbuf·qbuf'

        # --- step 6: store ---
        @views W[:, a] .= w
        @views P[:, a] .= pbuf
        @views Q[:, a] .= qbuf
        @views R[:, a] .= r
    end

    return Plsr{T_}(W, P, Q, R, Tt, xmeans, xscales, ymeans, yscales)
end
