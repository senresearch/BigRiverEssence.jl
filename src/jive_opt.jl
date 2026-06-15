function _jive_rjive_core_opt(Xc::Vector{Matrix{Float64}}, n::Int, r::Int, ri::Vector{Int};
                              conv::Float64, maxiter::Int)
    T_ = Float64
    k = length(Xc)

    # SVD-reduction (unchanged — runs once, not in the loop)
    Ubig = Vector{Matrix{T_}}(undef, k)
    Xr   = Vector{Matrix{T_}}(undef, k)
    for i in 1:k
        if size(Xc[i],1) > size(Xc[i],2)
            F = safe_svd(Xc[i]); nc = size(Xc[i], 2)
            Xr[i] = Diagonal(F.S[1:nc]) * F.Vt[1:nc, :]
            Ubig[i] = F.U[:, 1:nc]
        else
            Xr[i] = Xc[i]
            Ubig[i] = Matrix{T_}(I, size(Xc[i],1), size(Xc[i],1))
        end
    end

    pis = [size(X,1) for X in Xr]
    rowranges = (let rr = Vector{UnitRange{Int}}(undef, k); idx=1
                     for i in 1:k; rr[i]=idx:idx+pis[i]-1; idx+=pis[i]; end; rr end)

    A = [zeros(T_, pis[i], n) for i in 1:k]
    J = [zeros(T_, pis[i], n) for i in 1:k]
    Vind = [zeros(T_, n, ri[i]) for i in 1:k]
    Xtot = reduce(vcat, Xr)
    ptot = size(Xtot, 1)

    # preallocated buffers
    Jtot  = fill(-1.0, ptot, n)
    Atot  = fill(-1.0, ptot, n)
    Jlast = similar(Jtot)
    Alast = similar(Atot)
    tmpJ  = Matrix{T_}(undef, ptot, n)        # for Xtot - Atot

    nrun = 0; converged = false
    while nrun < maxiter && !converged
        copyto!(Jlast, Jtot); copyto!(Alast, Atot)

        # --- joint: rank-r SVD of (Xtot - Atot) ---
        if r > 0
            @. tmpJ = Xtot - Atot
            s = safe_svd(tmpJ)
            US = s.U[:,1:r] * Diagonal(s.S[1:r])     # ptot × r (small, r tiny)
            mul!(Jtot, US, @view s.Vt[1:r,:])         # Jtot = US * Vt[1:r,:]
            V = permutedims(@view s.Vt[1:r,:])        # n × r  (joint loadings)
        else
            fill!(Jtot, 0); V = zeros(T_, n, 0)
        end
        for i in 1:k
            @views J[i] .= Jtot[rowranges[i], :]
        end

        # --- individual: project away from joint AND other individuals ---
        for i in 1:k
            if ri[i] > 0
                # tmp = (Xr[i] - J[i]) projected ⟂ V  via  tmp - (tmp*V)*V'
                tmp = Xr[i] .- J[i]                    # pis[i] × n
                if r > 0
                    tmp .-= (tmp * V) * V'             # (tmp*V): pis×r,  *V': pis×n  — NO n×n matrix
                end
                if nrun > 0
                    for j in 1:k
                        j == i && continue
                        Vj = Vind[j]
                        tmp .-= (tmp * Vj) * Vj'       # same trick for each other individual
                    end
                end
                s = safe_svd(tmp)
                Vind[i] = permutedims(@view s.Vt[1:ri[i], :])
                A[i] = s.U[:,1:ri[i]] * Diagonal(s.S[1:ri[i]]) * @view(s.Vt[1:ri[i],:])
            else
                fill!(A[i], 0)
            end
        end

        # first-iteration re-orthogonalization
        if nrun == 0
            for i in 1:k, j in 1:k
                j == i && continue
                Vj = Vind[j]
                A[i] .-= (A[i] * Vj) * Vj'             # avoid n×n here too
            end
            for i in 1:k
                if ri[i] > 0
                    s = safe_svd(A[i]); Vind[i] = permutedims(@view s.Vt[1:ri[i], :])
                end
            end
        end

        # rebuild Atot
        for i in 1:k
            @views Atot[rowranges[i], :] .= A[i]
        end

        if norm(Jtot .- Jlast) <= conv && norm(Atot .- Alast) <= conv
            converged = true
        end
        nrun += 1
    end

    # map back & factorize (unchanged — runs once)
    Jfull = [Ubig[i] * J[i] for i in 1:k]
    Afull = [Ubig[i] * A[i] for i in 1:k]
    Fj = safe_svd(reduce(vcat, Jfull))
    S = Matrix(@view Fj.Vt[1:r, :])
    pis_full = [size(Ji,1) for Ji in Jfull]
    Ufull = Fj.U[:,1:r] * Diagonal(Fj.S[1:r])
    U = Matrix{T_}[]; idx=1
    for p in pis_full; push!(U, Ufull[idx:idx+p-1,:]); idx+=p; end
    Si = Matrix{T_}[]; Wi = Matrix{T_}[]
    for i in 1:k
        Fi = safe_svd(Afull[i])
        push!(Si, Matrix(@view Fi.Vt[1:ri[i], :]))
        push!(Wi, Fi.U[:,1:ri[i]] * Diagonal(Fi.S[1:ri[i]]))
    end
    return JiveResult{T_}(Jfull, Afull, S, U, Si, Wi, r, ri)
end



function _jive_rjive_core_opt2(Xc::Vector{Matrix{Float64}}, n::Int, r::Int, ri::Vector{Int};
                               conv::Float64, maxiter::Int)
    T_ = Float64
    k = length(Xc)

    # SVD-reduction (once, not in loop)
    Ubig = Vector{Matrix{T_}}(undef, k)
    Xr   = Vector{Matrix{T_}}(undef, k)
    for i in 1:k
        if size(Xc[i],1) > size(Xc[i],2)
            F = safe_svd(Xc[i]); nc = size(Xc[i], 2)
            Xr[i] = Diagonal(F.S[1:nc]) * F.Vt[1:nc, :]
            Ubig[i] = F.U[:, 1:nc]
        else
            Xr[i] = Xc[i]
            Ubig[i] = Matrix{T_}(I, size(Xc[i],1), size(Xc[i],1))
        end
    end

    pis = [size(X,1) for X in Xr]
    rowranges = (let rr = Vector{UnitRange{Int}}(undef, k); idx=1
                     for i in 1:k; rr[i]=idx:idx+pis[i]-1; idx+=pis[i]; end; rr end)
    ptot = sum(pis)

    A    = [zeros(T_, pis[i], n) for i in 1:k]
    J    = [zeros(T_, pis[i], n) for i in 1:k]
    Vind = [zeros(T_, n, ri[i]) for i in 1:k]
    Xtot = reduce(vcat, Xr)

    # ---- preallocated buffers reused every iteration ----
    Jtot  = fill(-1.0, ptot, n)
    Atot  = fill(-1.0, ptot, n)
    Jlast = similar(Jtot)
    Alast = similar(Atot)
    tmpJ  = Matrix{T_}(undef, ptot, n)            # Xtot - Atot (SVD input)
    V     = Matrix{T_}(undef, n, r)               # joint loadings (n × r)
    USj   = Matrix{T_}(undef, ptot, r)            # U[:,1:r]*Σ for joint
    tmpi  = [Matrix{T_}(undef, pis[i], n) for i in 1:k]   # per-dataset residual/proj
    projr = [Matrix{T_}(undef, pis[i], r) for i in 1:k]   # tmp*V  (pis × r)

    nrun = 0; converged = false
    while nrun < maxiter && !converged
        copyto!(Jlast, Jtot); copyto!(Alast, Atot)

        # --- joint: rank-r SVD of (Xtot - Atot) ---
        if r > 0
            @. tmpJ = Xtot - Atot
            s = svd!(copy(tmpJ))                        # svd! overwrites its input
            @views mul!(USj, s.U[:,1:r], Diagonal(s.S[1:r]))   # USj = U[:,1:r]*Σ
            @views mul!(Jtot, USj, s.Vt[1:r,:])         # Jtot = USj * Vt[1:r,:]
            @views copyto!(V, transpose(s.Vt[1:r,:]))   # V = (Vt[1:r,:])'  into preallocated V
        else
            fill!(Jtot, 0)
        end
        for i in 1:k
            @views J[i] .= Jtot[rowranges[i], :]
        end

        # --- individual: project ⟂ joint and ⟂ other individuals, then SVD ---
        for i in 1:k
            if ri[i] > 0
                tmp = tmpi[i]
                @. tmp = Xr[i] - J[i]                          # in place
                if r > 0
                    mul!(projr[i], tmp, V)                      # projr = tmp*V   (pis×r)
                    mul!(tmp, projr[i], transpose(V), -1.0, 1.0)# tmp -= projr*V'  (5-arg, in place)
                end
                if nrun > 0
                    for j in 1:k
                        j == i && continue
                        Vj = Vind[j]
                        pj = tmp * Vj                           # pis × ri[j]  (small alloc)
                        mul!(tmp, pj, transpose(Vj), -1.0, 1.0) # tmp -= pj*Vj'
                    end
                end
                s = svd!(copy(tmp))
                @views copyto!(Vind[i], transpose(s.Vt[1:ri[i], :]))
                @views mul!(A[i], s.U[:,1:ri[i]] * Diagonal(s.S[1:ri[i]]), s.Vt[1:ri[i],:])
            else
                fill!(A[i], 0)
            end
        end

        # first-iteration re-orthogonalization
        if nrun == 0
            for i in 1:k, j in 1:k
                j == i && continue
                Vj = Vind[j]
                pj = A[i] * Vj
                mul!(A[i], pj, transpose(Vj), -1.0, 1.0)        # A[i] -= pj*Vj'
            end
            for i in 1:k
                if ri[i] > 0
                    s = svd!(copy(A[i]))
                    @views copyto!(Vind[i], transpose(s.Vt[1:ri[i], :]))
                end
            end
        end

        for i in 1:k
            @views Atot[rowranges[i], :] .= A[i]
        end

        if norm(Jtot .- Jlast) <= conv && norm(Atot .- Alast) <= conv
            converged = true
        end
        nrun += 1
    end

    # map back & factorize (once)
    Jfull = [Ubig[i] * J[i] for i in 1:k]
    Afull = [Ubig[i] * A[i] for i in 1:k]
    Fj = safe_svd(reduce(vcat, Jfull))
    S = Matrix(@view Fj.Vt[1:r, :])
    pis_full = [size(Ji,1) for Ji in Jfull]
    Ufull = Fj.U[:,1:r] * Diagonal(Fj.S[1:r])
    U = Matrix{T_}[]; idx=1
    for p in pis_full; push!(U, Ufull[idx:idx+p-1,:]); idx+=p; end
    Si = Matrix{T_}[]; Wi = Matrix{T_}[]
    for i in 1:k
        Fi = safe_svd(Afull[i])
        push!(Si, Matrix(@view Fi.Vt[1:ri[i], :]))
        push!(Wi, Fi.U[:,1:ri[i]] * Diagonal(Fi.S[1:ri[i]]))
    end
    return JiveResult{T_}(Jfull, Afull, S, U, Si, Wi, r, ri)
end