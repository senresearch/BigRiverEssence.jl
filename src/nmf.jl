"""
	Nmf{T}

Container for a fitted nonnegative matrix factorization, as returned by `nmf`
# Fields
- `w::Matrix{T}`: The n×k nonnegative sample scores, one row per observation
- `h::Matrix{T}`: The k×p nonnegative components, one column per input feature
- `reconstruction_error::T`: The Frobenius norm ‖X − w*h‖_F of the final
  reconstruction residual
- `niter::Int`: The number of alternating coordinate-descent iterations performed
- `converged::Bool`: Whether the normalized projected-gradient violation reached
  the requested tolerance before `maxiter`
- `alpha_w::T`: The elastic-net regularization strength used for `w`
- `alpha_h::T`: The elastic-net regularization strength used for `h`
- `l1_ratio::T`: The fraction of regularization assigned to the L1 penalty, in
  [0,1]; the remainder is assigned to the squared Frobenius penalty
"""
struct Nmf{T}
	w::Matrix{T}
	h::Matrix{T}
	reconstruction_error::T
	niter::Int
	converged::Bool
	alpha_w::T
	alpha_h::T
	l1_ratio::T
end

"""
	nmf_transform(M::Nmf, X::Matrix{Float64}; tol::Real = 1e-4,
		maxiter::Int = 200, shuffle::Bool = false,
		rng::AbstractRNG = Random.GLOBAL_RNG)

Represent new nonnegative observations using the fixed components of a fitted NMF
# Arguments
- `M::Nmf`: A fitted NMF model, as returned by `nmf`
- `X::Matrix{Float64}`: 2d array of nonnegative finite floats; new observations
  in rows and the same p features used to fit `M` in columns
- `tol::Real`: The stopping tolerance for the normalized projected-gradient
  violation. Defaults to 1e-4
- `maxiter::Int`: The maximum number of coordinate-descent sweeps. Defaults to 200
- `shuffle::Bool`: Whether to randomize the component-update order in each sweep.
  Defaults to false
- `rng::AbstractRNG`: Random-number generator used only when `shuffle=true`.
  Defaults to Julia's global RNG
# Value
2d array of floats; the n×k nonnegative score matrix W minimizing ‖X − W*M.h‖²
with the fitted components held fixed and the model's W regularization applied
"""
function nmf_transform(M::Nmf, X::Matrix{Float64}; tol::Real = 1e-4,
	maxiter::Int = 200, shuffle::Bool = false,
	rng::AbstractRNG = Random.GLOBAL_RNG)
	_validate_nmf_data(X)
	size(X, 2) == size(M.h, 2) || throw(DimensionMismatch(
		"X has $(size(X, 2)) features, but the fitted NMF expects $(size(M.h, 2))",
	))
	isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
	maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1"))

	n, p = size(X)
	k = size(M.h, 1)
	w = zeros(Float64, n, k)
	ht = transpose(M.h)
	l1_w = p * M.alpha_w * M.l1_ratio
	l2_w = p * M.alpha_w * (1 - M.l1_ratio)
	_, converged = _nmf_fit_cd!(X, w, ht, l1_w, 0.0, l2_w, 0.0;
		tol = Float64(tol), maxiter = maxiter, shuffle = shuffle,
		rng = rng, update_h = false)
	converged || @warn "NMF transform reached maxiter=$maxiter before convergence; increase maxiter to improve the scores."
	return w
end

"""
	nmf_invtransform(M::Nmf, W::Matrix{Float64})

Reconstruct observations in the original feature space from nonnegative NMF scores
# Arguments
- `M::Nmf`: A fitted NMF model, as returned by `nmf`
- `W::Matrix{Float64}`: 2d array of floats; the n×k score matrix to reconstruct,
  with k matching the number of fitted components
# Value
2d array of floats; the n×p reconstruction W*M.h
"""
function nmf_invtransform(M::Nmf, W::Matrix{Float64})
	size(W, 2) == size(M.h, 1) || throw(DimensionMismatch(
		"W has $(size(W, 2)) components, but the fitted NMF has $(size(M.h, 1))",
	))
	return W * M.h
end

"""
	_validate_nmf_data(X::Matrix{Float64})

Validate a data matrix for nonnegative matrix factorization
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; observations in rows and features in
  columns
# Value
The input matrix `X`. Throws an `ArgumentError` if it is empty, contains a
non-finite value, or contains a negative value
"""
function _validate_nmf_data(X::Matrix{Float64})
	size(X, 1) >= 1 || throw(ArgumentError("X must contain at least one observation"))
	size(X, 2) >= 1 || throw(ArgumentError("X must contain at least one feature"))
	all(isfinite, X) || throw(ArgumentError("X must contain only finite values"))
	all(x -> x >= 0, X) || throw(ArgumentError("NMF requires a nonnegative matrix X"))
	return X
end

"""
	_nmf_random_init(X::Matrix{Float64}, k::Int, rng::AbstractRNG)

Initialize both NMF factors with scaled absolute Gaussian values
# Arguments
- `X::Matrix{Float64}`: 2d array of nonnegative floats; the n×p matrix to factorize
- `k::Int`: The requested factorization rank
- `rng::AbstractRNG`: Random-number generator used for every draw
# Value
A tuple `(w, ht)` containing an n×k score matrix and a p×k transposed-component
matrix. Entries are `abs(randn()) * sqrt(mean(X)/k)`, matching scikit-learn's
random NMF initialization while storing Hᵀ for column-major coordinate updates
"""
function _nmf_random_init(X::Matrix{Float64}, k::Int, rng::AbstractRNG)
	n, p = size(X)
	initial_scale = sqrt(mean(X) / k)
	w = Matrix{Float64}(undef, n, k)
	ht = Matrix{Float64}(undef, p, k)
	@inbounds for i in eachindex(w)
		w[i] = initial_scale * abs(randn(rng))
	end
	@inbounds for i in eachindex(ht)
		ht[i] = initial_scale * abs(randn(rng))
	end
	return w, ht
end

"""
	_nmf_nndsvd_init(X::Matrix{Float64}, k::Int, variant::Symbol,
		rng::AbstractRNG; eps::Float64 = 1e-6)

Initialize NMF factors with Nonnegative Double Singular Value Decomposition
# Arguments
- `X::Matrix{Float64}`: 2d array of nonnegative floats; the n×p matrix to factorize
- `k::Int`: The requested rank, no larger than min(n,p)
- `variant::Symbol`: `:nndsvd` to retain zeros, `:nndsvda` to replace zeros by
  `mean(X)`, or `:nndsvdar` to replace zeros by small random values
- `rng::AbstractRNG`: Random-number generator used only by `:nndsvdar`
- `eps::Float64`: Values below this threshold are set to zero before applying the
  selected zero-filling rule. Defaults to 1e-6
# Value
A tuple `(w, ht)` containing the n×k scores and p×k transposed components.
Positive and negative singular-vector parts are compared without materializing
four temporary vectors, and only the selected pair is written into the factors
"""
function _nmf_nndsvd_init(X::Matrix{Float64}, k::Int, variant::Symbol,
	rng::AbstractRNG; eps::Float64 = 1e-6)
	n, p = size(X)
	k <= min(n, p) || throw(ArgumentError(
		"$variant initialization requires k <= min(n,p)",
	))
	F = svd!(copy(X); full = false)
	w = zeros(Float64, n, k)
	ht = zeros(Float64, p, k)

	first_scale = sqrt(F.S[1])
	@inbounds @simd for i in 1:n
		w[i, 1] = first_scale * abs(F.U[i, 1])
	end # COV_EXCL_LINE
	@inbounds @simd for j in 1:p
		ht[j, 1] = first_scale * abs(F.V[j, 1])
	end # COV_EXCL_LINE

	for component in 2:k
		upos2 = 0.0;
		uneg2 = 0.0
		vpos2 = 0.0;
		vneg2 = 0.0
		@inbounds @simd for i in 1:n
			value = F.U[i, component]
			value >= 0 ? (upos2 += value * value) : (uneg2 += value * value)
		end # COV_EXCL_LINE
		@inbounds @simd for j in 1:p
			value = F.V[j, component]
			value >= 0 ? (vpos2 += value * value) : (vneg2 += value * value)
		end # COV_EXCL_LINE

		upos = sqrt(upos2);
		uneg = sqrt(uneg2)
		vpos = sqrt(vpos2);
		vneg = sqrt(vneg2)
		positive_product = upos * vpos
		negative_product = uneg * vneg
		use_positive = positive_product > negative_product
		sigma = use_positive ? positive_product : negative_product
		iszero(sigma) && continue

		factor_scale = sqrt(F.S[component] * sigma)
		unorm = use_positive ? upos : uneg
		vnorm = use_positive ? vpos : vneg
		@inbounds @simd for i in 1:n
			value = use_positive ? max(F.U[i, component], 0.0) : max(-F.U[i, component], 0.0)
			w[i, component] = factor_scale * value / unorm
		end # COV_EXCL_LINE
		@inbounds @simd for j in 1:p
			value = use_positive ? max(F.V[j, component], 0.0) : max(-F.V[j, component], 0.0)
			ht[j, component] = factor_scale * value / vnorm
		end # COV_EXCL_LINE
	end

	@inbounds @simd for i in eachindex(w)
		w[i] < eps && (w[i] = 0.0)
	end # COV_EXCL_LINE
	@inbounds @simd for i in eachindex(ht)
		ht[i] < eps && (ht[i] = 0.0)
	end # COV_EXCL_LINE

	if variant === :nndsvda
		average = mean(X)
		@inbounds @simd for i in eachindex(w)
			iszero(w[i]) && (w[i] = average)
		end # COV_EXCL_LINE
		@inbounds @simd for i in eachindex(ht)
			iszero(ht[i]) && (ht[i] = average)
		end # COV_EXCL_LINE
	elseif variant === :nndsvdar
		average = mean(X) / 100
		@inbounds for i in eachindex(w)
			iszero(w[i]) && (w[i] = average * abs(randn(rng)))
		end
		@inbounds for i in eachindex(ht)
			iszero(ht[i]) && (ht[i] = average * abs(randn(rng)))
		end
	end
	return w, ht
end

"""
	_nmf_initialize(X::Matrix{Float64}, k::Int, init::Symbol,
		rng::AbstractRNG, w_init, h_init)

Resolve and construct the requested NMF initialization
# Arguments
- `X::Matrix{Float64}`: 2d array of nonnegative floats; the n×p matrix to factorize
- `k::Int`: The requested factorization rank
- `init::Symbol`: `:auto`, `:random`, `:nndsvd`, `:nndsvda`, `:nndsvdar`, or
  `:custom`
- `rng::AbstractRNG`: Random-number generator for randomized initializations
- `w_init`: Optional n×k custom initial score matrix
- `h_init`: Optional k×p custom initial component matrix
# Value
A tuple `(w, ht)` containing mutable Float64 factors. `:auto` selects `:nndsvda`
when k <= min(n,p), otherwise `:random`. Throws an `ArgumentError` or
`DimensionMismatch` for an invalid initialization request
"""
function _nmf_initialize(X::Matrix{Float64}, k::Int, init::Symbol,
	rng::AbstractRNG, w_init, h_init)
	n, p = size(X)
	valid = (:auto, :random, :nndsvd, :nndsvda, :nndsvdar, :custom)
	init in valid || throw(ArgumentError("init must be one of $valid, got :$init"))
	resolved = init === :auto ? (k <= min(n, p) ? :nndsvda : :random) : init

	if resolved === :custom
		w_init === nothing && throw(ArgumentError("w_init is required when init=:custom"))
		h_init === nothing && throw(ArgumentError("h_init is required when init=:custom"))
		size(w_init) == (n, k) || throw(DimensionMismatch(
			"w_init must have size ($n,$k), got $(size(w_init))",
		))
		size(h_init) == (k, p) || throw(DimensionMismatch(
			"h_init must have size ($k,$p), got $(size(h_init))",
		))
		all(isfinite, w_init) && all(isfinite, h_init) || throw(ArgumentError(
			"custom NMF factors must contain only finite values",
		))
		all(x -> x >= 0, w_init) && all(x -> x >= 0, h_init) || throw(ArgumentError(
			"custom NMF factors must be nonnegative",
		))
		return Matrix{Float64}(w_init), permutedims(Matrix{Float64}(h_init))
	end

	(w_init === nothing && h_init === nothing) || throw(ArgumentError(
		"w_init and h_init are accepted only when init=:custom",
	))
	resolved === :random && return _nmf_random_init(X, k, rng)
	return _nmf_nndsvd_init(X, k, resolved, rng)
end

"""
	_nmf_cd_sweep!(A, F, G, gram, cross, gradient, order,
		l1_regularization::Float64, l2_regularization::Float64)

Perform one Fast HALS coordinate-descent sweep over a nonnegative factor
# Arguments
- `A`: 2d array of floats; the m×q target matrix
- `F`: 2d array of floats; the mutable m×k factor updated in place
- `G`: 2d array of floats; the fixed q×k factor for this half-step
- `gram`: Preallocated k×k buffer for GᵀG
- `cross`: Preallocated m×k buffer for A*G
- `gradient`: Preallocated length-m gradient buffer
- `order`: Length-k integer vector giving the coordinate-update order
- `l1_regularization::Float64`: L1 term added to every coordinate gradient
- `l2_regularization::Float64`: L2 term added to the Gram-matrix diagonal
# Value
Float; the sum of absolute projected-gradient violations before each coordinate
update. The sweep updates every column of `F` in place using
`max(F[:,t] - gradient/GᵀG[t,t], 0)`. All matrix products and work vectors are
reused, so the iterative hot path does not allocate an m×q reconstruction
"""
function _nmf_cd_sweep!(A, F, G, gram, cross, gradient, order,
	l1_regularization::Float64, l2_regularization::Float64)
	mul!(gram, transpose(G), G)
	@inbounds for component in axes(gram, 1)
		gram[component, component] += l2_regularization
	end
	mul!(cross, A, G)

	violation = 0.0
	@inbounds for position in eachindex(order)
		component = order[position]
		@views mul!(gradient, F, gram[:, component])
		hessian = gram[component, component]
		@simd for observation in axes(F, 1)
			current = F[observation, component]
			grad = gradient[observation] - cross[observation, component] + l1_regularization
			projected_grad = iszero(current) ? min(0.0, grad) : grad
			violation += abs(projected_grad)
			iszero(hessian) || (F[observation, component] = max(current - grad / hessian, 0.0))
		end # COV_EXCL_LINE
	end
	return violation
end

"""
	_nmf_fit_cd!(X, w, ht, l1_w::Float64, l1_h::Float64,
		l2_w::Float64, l2_h::Float64; tol::Float64, maxiter::Int,
		shuffle::Bool, rng::AbstractRNG, update_h::Bool = true)

Fit mutable NMF factors with alternating Fast HALS coordinate descent
# Arguments
- `X`: 2d array of floats; the n×p nonnegative target matrix
- `w`: Mutable n×k score matrix
- `ht`: Mutable p×k transposed-component matrix; held fixed when `update_h=false`
- `l1_w::Float64`: Scaled L1 regularization applied to W
- `l1_h::Float64`: Scaled L1 regularization applied to H
- `l2_w::Float64`: Scaled L2 regularization applied to W
- `l2_h::Float64`: Scaled L2 regularization applied to H
- `tol::Float64`: Stopping tolerance for violation/initial_violation
- `maxiter::Int`: Maximum number of alternating sweeps
- `shuffle::Bool`: Whether to shuffle component coordinates before each half-step
- `rng::AbstractRNG`: Random-number generator used when shuffling
- `update_h::Bool`: Whether to update H as well as W. Defaults to true
# Value
A tuple `(niter, converged)`. The convergence rule matches scikit-learn's
coordinate-descent NMF: stop when the projected-gradient violation relative to
the first iteration is no greater than `tol`
"""
function _nmf_fit_cd!(X, w, ht, l1_w::Float64, l1_h::Float64,
	l2_w::Float64, l2_h::Float64; tol::Float64, maxiter::Int,
	shuffle::Bool, rng::AbstractRNG, update_h::Bool = true)
	n, p = size(X)
	k = size(w, 2)
	gram = Matrix{Float64}(undef, k, k)
	cross_w = Matrix{Float64}(undef, n, k)
	gradient_w = Vector{Float64}(undef, n)
	cross_h = update_h ? Matrix{Float64}(undef, p, k) : Matrix{Float64}(undef, 0, 0)
	gradient_h = update_h ? Vector{Float64}(undef, p) : Float64[]
	order = collect(1:k)
	initial_violation = 0.0

	for iteration in 1:maxiter
		shuffle && shuffle!(rng, order)
		violation = _nmf_cd_sweep!(X, w, ht, gram, cross_w, gradient_w,
			order, l1_w, l2_w)
		if update_h
			shuffle && shuffle!(rng, order)
			violation += _nmf_cd_sweep!(transpose(X), ht, w, gram, cross_h,
				gradient_h, order, l1_h, l2_h)
		end

		if iteration == 1
			initial_violation = violation
			iszero(initial_violation) && return iteration, true
		end
		violation / initial_violation <= tol && return iteration, true
	end
	return maxiter, false
end

"""
	_nmf_reconstruction_error(X, w, h)

Compute the NMF Frobenius reconstruction error without forming W*H
# Arguments
- `X`: 2d array of floats; the n×p target matrix
- `w`: 2d array of floats; the n×k score matrix
- `h`: 2d array of floats; the k×p component matrix
# Value
Float; ‖X − W*H‖_F. Accumulates each reconstructed entry directly so it avoids
an n×p residual allocation without suffering the cancellation error of the
expanded Gram-matrix identity when the reconstruction is nearly exact
"""
function _nmf_reconstruction_error(X, w, h)
	n, p = size(X)
	k = size(w, 2)
	error_squared = 0.0
	@inbounds for feature in 1:p, observation in 1:n
		reconstructed = 0.0
		@simd for component in 1:k
			reconstructed += w[observation, component] * h[component, feature]
		end # COV_EXCL_LINE
		residual = X[observation, feature] - reconstructed
		error_squared += residual * residual
	end
	return sqrt(error_squared)
end

"""
	nmf(X::Matrix{Float64}; k::Int = minimum(size(X)), init::Symbol = :auto,
		tol::Real = 1e-4, maxiter::Int = 200, alpha_w::Real = 0.0,
		alpha_h::Union{Nothing,Real} = nothing, l1_ratio::Real = 0.0,
		shuffle::Bool = false, rng::AbstractRNG = Random.GLOBAL_RNG,
		w_init = nothing, h_init = nothing)

Fit a nonnegative matrix factorization using Fast HALS coordinate descent
# Arguments
- `X::Matrix{Float64}`: 2d array of nonnegative finite floats; observations in
  rows and features in columns
- `k::Int`: The factorization rank. Defaults to min(n,p)
- `init::Symbol`: Initialization method: `:auto`, `:random`, `:nndsvd`,
  `:nndsvda`, `:nndsvdar`, or `:custom`. `:auto` uses `:nndsvda` when possible
  and `:random` otherwise. Defaults to `:auto`
- `tol::Real`: Stopping tolerance for the normalized projected-gradient
  violation. Defaults to 1e-4
- `maxiter::Int`: Maximum number of alternating coordinate-descent iterations.
  Defaults to 200
- `alpha_w::Real`: Elastic-net strength for W, scaled internally by the number
  of features. Defaults to 0
- `alpha_h::Union{Nothing,Real}`: Elastic-net strength for H, scaled internally
  by the number of observations. `nothing` uses `alpha_w`. Defaults to nothing
- `l1_ratio::Real`: Fraction of regularization assigned to L1, in [0,1]; the
  remainder is squared Frobenius regularization. Defaults to 0
- `shuffle::Bool`: Whether to randomize the component-coordinate order at every
  W and H update. Defaults to false
- `rng::AbstractRNG`: Random-number generator used by randomized initialization
  and coordinate shuffling. Defaults to Julia's global RNG
- `w_init`: Custom n×k nonnegative initial W, required only with `init=:custom`
- `h_init`: Custom k×p nonnegative initial H, required only with `init=:custom`
# Value
An `Nmf` holding nonnegative factors W and H, their Frobenius reconstruction
error, convergence information, and regularization settings. It minimizes the
scikit-learn NMF objective: one-half squared reconstruction error plus
size-scaled elastic-net penalties on W and H, using alternating Fast HALS
coordinate descent. Warns if `maxiter` is reached before convergence
"""
function nmf(X::Matrix{Float64}; k::Int = minimum(size(X)), init::Symbol = :auto,
	tol::Real = 1e-4, maxiter::Int = 200, alpha_w::Real = 0.0,
	alpha_h::Union{Nothing, Real} = nothing, l1_ratio::Real = 0.0,
	shuffle::Bool = false, rng::AbstractRNG = Random.GLOBAL_RNG,
	w_init = nothing, h_init = nothing)
	_validate_nmf_data(X)
	k >= 1 || throw(ArgumentError("k must be at least 1"))
	isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
	maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1"))
	isfinite(alpha_w) && alpha_w >= 0 || throw(ArgumentError("alpha_w must be finite and nonnegative"))
	resolved_alpha_h = alpha_h === nothing ? Float64(alpha_w) : Float64(alpha_h)
	isfinite(resolved_alpha_h) && resolved_alpha_h >= 0 || throw(ArgumentError("alpha_h must be finite and nonnegative"))
	0 <= l1_ratio <= 1 || throw(ArgumentError("l1_ratio must be in [0,1]"))

	n, p = size(X)
	w, ht = _nmf_initialize(X, k, init, rng, w_init, h_init)
	aw = Float64(alpha_w)
	l1 = Float64(l1_ratio)
	l1_w = p * aw * l1
	l1_h = n * resolved_alpha_h * l1
	l2_w = p * aw * (1 - l1)
	l2_h = n * resolved_alpha_h * (1 - l1)
	niter, converged = _nmf_fit_cd!(X, w, ht, l1_w, l1_h, l2_w, l2_h;
		tol = Float64(tol), maxiter = maxiter, shuffle = shuffle, rng = rng)
	h = permutedims(ht)
	reconstruction_error = _nmf_reconstruction_error(X, w, h)
	converged || @warn "NMF reached maxiter=$maxiter before convergence; increase maxiter to improve the fit."
	return Nmf(w, h, reconstruction_error, niter, converged, aw,
		resolved_alpha_h, l1)
end
