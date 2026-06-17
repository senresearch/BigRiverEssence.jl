# Abhisek Banerjee
# splsda — OPTIMIZED version of mixOmics' sPLS-DA (single-block path).
# Same algorithm and results as splsda; preallocated buffers, mul! for products,
# in-place soft-threshold, BLAS.ger! rank-1 deflation.
# Uses (already defined in splsda.jl, NOT redefined here): SplsdaResult, _unmap,
# _center_scale.



# in-place soft-threshold: writes result into preallocated `out`, using
# preallocated `absx`, `ord`, `ranks` work buffers. No per-call allocation.
function _soft_threshold_L1!(out, x, nx::Int, absx, ord, ranks)
    p = length(x)
    if nx <= 0
        copyto!(out, x)
        return out
    end
    @. absx = abs(x)
    sortperm!(ord, absx)                       # in-place: ord = ascending order of absx
    fill!(ranks, 0)
    i = 1
    while i <= p
        j = i
        while j < p && absx[ord[j+1]] == absx[ord[i]]
            j += 1
        end
        for m in i:j; ranks[ord[m]] = j; end   # ties get the max rank
        i = j + 1
    end
    # find lambda = largest dropped magnitude (entries with rank <= nx)
    lambda = 0.0; anydrop = false
    @inbounds for t in 1:p
        if ranks[t] <= nx
            anydrop = true
            absx[t] > lambda && (lambda = absx[t])
        end
    end
    if !anydrop                                # nothing dropped → keep all
        copyto!(out, x)
        return out
    end
    @inbounds for t in 1:p
        out[t] = ranks[t] > nx ? sign(x[t]) * (absx[t] - lambda) : 0.0
    end
    return out
end

# allocation-free ‖a - b‖²
function _sqdiff(a, b)
    s = zero(eltype(a))
    @inbounds @simd for i in eachindex(a)
        d = a[i] - b[i]; s += d * d
    end
    return s
end

function splsda_opt(X::Matrix{Float64}, y::Vector, ncomp::Int, keepX::Vector{Int};
                    scale=true, tol=1e-6, max_iter=100, levels=nothing)
    n, p = size(X)
    Yd, classes = _unmap(y; levels=levels)
    k = size(Yd, 2)
    length(keepX) == ncomp || throw(ArgumentError("keepX must have length ncomp"))

    Xc = _center_scale(Matrix{Float64}(X); scale=scale)
    Yc = _center_scale(Yd; scale=scale)

    TX = zeros(n, ncomp); TY = zeros(n, ncomp)
    PX = zeros(p, ncomp); PY = zeros(k, ncomp)

    R  = copy(Xc)                              # X residual (deflated each comp)
    Ry = copy(Yc)                              # Y residual (not deflated for DA)

    #  preallocated buffers reused across components and inner iterations 
    M     = Matrix{Float64}(undef, p, k)
    uh    = Vector{Float64}(undef, p); uh_old = Vector{Float64}(undef, p)
    vh    = Vector{Float64}(undef, k); vh_old = Vector{Float64}(undef, k)
    tX    = Vector{Float64}(undef, n)
    tY    = Vector{Float64}(undef, n)
    uraw  = Vector{Float64}(undef, p)          # R'*tY before thresholding
    pX    = Vector{Float64}(undef, p)
    absx  = Vector{Float64}(undef, p)
    ord   = Vector{Int}(undef, p)
    ranks = Vector{Int}(undef, p)

    for comp in 1:ncomp
        #  init via SVD of M = R'*Ry 
        mul!(M, transpose(R), Ry)
        F = svd(M)
        copyto!(uh, @view F.U[:, 1])
        copyto!(vh, @view F.V[:, 1])
        copyto!(uh_old, uh); copyto!(vh_old, vh)

        iter = 1
        while true
            mul!(tY, Ry, vh)                          # tY = Ry*vh
            mul!(uraw, transpose(R), tY)              # uraw = R'*tY  (= M*vh)
            _soft_threshold_L1!(uh, uraw, p - keepX[comp], absx, ord, ranks)
            uh ./= sqrt(sum(abs2, uh))                # normalize in place
            mul!(tX, R, uh)                           # tX = R*uh
            mul!(vh, transpose(Ry), tX)               # vh = Ry'*tX
            vh ./= sqrt(sum(abs2, vh))                # normalize in place

            dX = _sqdiff(uh, uh_old)
            dY = _sqdiff(vh, vh_old)
            (max(dX, dY) < tol || iter > max_iter) && break
            copyto!(uh_old, uh); copyto!(vh_old, vh)
            iter += 1
        end

        mul!(tX, R, uh); mul!(tY, Ry, vh)
        @views TX[:, comp] .= tX; @views TY[:, comp] .= tY
        @views PX[:, comp] .= uh; @views PY[:, comp] .= vh

        #  regression deflation of X by tX:  pX = R'tX/(tX'tX);  R -= tX*pX' 
        mul!(pX, transpose(R), tX)
        pX ./= dot(tX, tX)
        BLAS.ger!(-1.0, tX, pX, R)                    # R -= tX*pX'  (rank-1, in place)
        # Ry not deflated (DA)
    end

    return SplsdaResult(TX, TY, PX, PY, ncomp, keepX, Yd, classes)
end