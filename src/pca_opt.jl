# ---- optimized helpers ----

# sign consistency without allocating abs.(c) each column
function SignConsistency_opt!(V)
    @inbounds for c in eachcol(V)
        # find argmax of |c| without materializing abs.(c)
        mi = 1; mv = abs(c[1])
        for i in 2:length(c)    # loop to find the index of the largest absolute value in the column vector c; this is done without creating a temporary array for abs.(c), which saves memory and can be faster; we keep track of the maximum absolute value (mv) and its index (mi) as we iterate through the entries of c, updating them whenever we find a larger absolute value.
            a = abs(c[i])
            if a > mv; mv = a; mi = i; end
        end
        s = sign(c[mi])
        s != 0 && (c .*= s)  # if the largest absolute value is not zero, we multiply the entire column by its sign to make it positive; this ensures that the sign of the principal component is consistent across different runs and methods, which is important for interpretability and comparison of results.
    end
    return V
end

function pca_opt(X; k = minimum(size(X)), standardize = false, method = :svd)
    n, p = size(X)
    1 <= k <= min(n, p) || throw(ArgumentError("$k you chose is not in range. Please reselect $k"))
    means = vec(mean(X, dims = 1))
    sigma = standardize ? vec(std(X, dims = 1)) : ones(eltype(means), p)
    Xc = (X .- means') ./ sigma'                 # one fused centered copy (unavoidable)
    T  = eltype(Xc)
    total = sum(abs2, Xc) / (n - 1)

    if method === :cov
        C = Symmetric(Xc'Xc)                      # p×p; ./ (n-1) folded into vars below
        F = eigen(C)                              # ascending eigenvalues
        # take top k without sortperm-of-all + copy: eigen returns ascending, so top k are the LAST k
        idx = p:-1:(p-k+1)                         # descending order of the top k
        vars = @view(F.values[idx]) ./ (n - 1)     # variances explained by the top k components (the eigenvalues divided by n-1); we use @view to avoid copying the slice of eigenvalues, which can save memory and improve performance; this is important because the eigenvalues array can be large, and we only need the top k values for our PCA results.
        V = F.vectors[:, idx]                      # p×k (copy needed for the struct)
        vars = collect(vars)
    elseif method === :svd
        F = svd(Xc; full = false)                  # thin SVD: U is n×min(n,p), not n×n
        vars = @view(F.S[1:k]) .^ 2 ./ (n - 1)
        V = F.V[:, 1:k]
        vars = collect(vars)
    else
        error("unknown method :$method")
    end

    SignConsistency_opt!(V)
    return pcaStructure{T}(T.(means), T.(sigma), Matrix(V), vars, vars ./ total)
end