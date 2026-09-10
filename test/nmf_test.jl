# Tests for Nmf, nmf, nmf_transform, nmf_invtransform, and every NMF-specific
# internal function in src/nmf.jl.

@testset "Nmf container" begin
	w = [1.0 2.0; 3.0 4.0]
	h = [0.5 1.5 2.5; 3.5 4.5 5.5]
	m = Nmf(w, h, 0.25, 7, true, 0.1, 0.2, 0.3)
	@test m isa Nmf{Float64}
	@test m.w === w
	@test m.h === h
	@test m.reconstruction_error == 0.25
	@test m.niter == 7
	@test m.converged
	@test m.alpha_w == 0.1
	@test m.alpha_h == 0.2
	@test m.l1_ratio == 0.3
end

@testset "_validate_nmf_data" begin
	X = [0.0 1.0; 2.0 3.0]
	@test BigRiverEssence._validate_nmf_data(X) === X
	@test BigRiverEssence._validate_nmf_data(zeros(2, 3)) == zeros(2, 3)
	@test_throws ArgumentError BigRiverEssence._validate_nmf_data(zeros(0, 3))
	@test_throws ArgumentError BigRiverEssence._validate_nmf_data(zeros(3, 0))
	@test_throws ArgumentError BigRiverEssence._validate_nmf_data([-eps() 1.0])
	@test_throws ArgumentError BigRiverEssence._validate_nmf_data([NaN 1.0])
	@test_throws ArgumentError BigRiverEssence._validate_nmf_data([Inf 1.0])
	@test_throws ArgumentError BigRiverEssence._validate_nmf_data([-Inf 1.0])
end

@testset "_nmf_random_init" begin
	X = fill(4.0, 3, 2)
	k = 2
	seed = 110
	w, ht = BigRiverEssence._nmf_random_init(X, k, MersenneTwister(seed))

	scale = sqrt(mean(X) / k)
	reference_rng = MersenneTwister(seed)
	expected_w = Matrix{Float64}(undef, 3, 2)
	expected_ht = Matrix{Float64}(undef, 2, 2)
	for i in eachindex(expected_w)
		expected_w[i] = scale * abs(randn(reference_rng))
	end
	for i in eachindex(expected_ht)
		expected_ht[i] = scale * abs(randn(reference_rng))
	end

	@test w == expected_w
	@test ht == expected_ht
	@test size(w) == (3, 2)
	@test size(ht) == (2, 2)
	@test all(isfinite, w) && all(isfinite, ht)
	@test all(w .>= 0) && all(ht .>= 0)
	@test BigRiverEssence._nmf_random_init(
		X, k, MersenneTwister(seed)) == (w, ht)
	zw, zht = BigRiverEssence._nmf_random_init(
		zeros(4, 5), 3, MersenneTwister(seed))
	@test all(iszero, zw)
	@test all(iszero, zht)
end

@testset "_nmf_nndsvd_init" begin
	X = [4.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.25; 0.0 0.0 0.0]
	X_original = copy(X)
	w, ht = BigRiverEssence._nmf_nndsvd_init(
		X, 3, :nndsvd, MersenneTwister(120),
	)
	@test X == X_original
	@test size(w) == (4, 3)
	@test size(ht) == (3, 3)
	@test all(w .>= 0) && all(ht .>= 0)
	@test w * transpose(ht) ≈ X atol = tol_ord
	@test count(iszero, w) > 0
	@test count(iszero, ht) > 0

	wa, hta = BigRiverEssence._nmf_nndsvd_init(
		X, 3, :nndsvda, MersenneTwister(121),
	)
	@test all(wa .> 0)
	@test all(hta .> 0)
	@test all(wa[iszero.(w)] .== mean(X))
	@test all(hta[iszero.(ht)] .== mean(X))

	war1, htar1 = BigRiverEssence._nmf_nndsvd_init(
		X, 3, :nndsvdar, MersenneTwister(122),
	)
	war2, htar2 = BigRiverEssence._nmf_nndsvd_init(
		X, 3, :nndsvdar, MersenneTwister(122),
	)
	@test war1 == war2
	@test htar1 == htar2
	@test all(war1 .>= 0) && all(htar1 .>= 0)
	@test all(war1[iszero.(w)] .> 0)
	@test all(htar1[iszero.(ht)] .> 0)

	for variant in (:nndsvd, :nndsvda, :nndsvdar)
		vw, vht = BigRiverEssence._nmf_nndsvd_init(
			zeros(4, 3), 2, variant, MersenneTwister(123),
		)
		@test all(iszero, vw)
		@test all(iszero, vht)
	end
	@test_throws ArgumentError BigRiverEssence._nmf_nndsvd_init(
		X, 4, :nndsvd, MersenneTwister(124),
	)
end

@testset "_nmf_initialize dispatcher" begin
	X = reshape(collect(1.0:60.0), 12, 5)

	for init in (:nndsvd, :nndsvda, :nndsvdar, :random, :auto)
		w, ht = BigRiverEssence._nmf_initialize(
			X, 3, init, MersenneTwister(130), nothing, nothing,
		)
		@test size(w) == (12, 3)
		@test size(ht) == (5, 3)
		@test all(isfinite, w) && all(isfinite, ht)
		@test all(w .>= 0) && all(ht .>= 0)
	end

	random_dispatch = BigRiverEssence._nmf_initialize(
		X, 3, :random, MersenneTwister(131), nothing, nothing,
	)
	random_direct = BigRiverEssence._nmf_random_init(
		X, 3, MersenneTwister(131),
	)
	@test random_dispatch == random_direct

	auto_nndsvd = BigRiverEssence._nmf_initialize(
		X, 3, :auto, MersenneTwister(132), nothing, nothing,
	)
	direct_nndsvd = BigRiverEssence._nmf_nndsvd_init(
		X, 3, :nndsvda, MersenneTwister(132),
	)
	@test auto_nndsvd == direct_nndsvd

	short_X = reshape(collect(1.0:6.0), 2, 3)
	auto_random = BigRiverEssence._nmf_initialize(
		short_X, 4, :auto, MersenneTwister(133), nothing, nothing,
	)
	direct_random = BigRiverEssence._nmf_random_init(
		short_X, 4, MersenneTwister(133),
	)
	@test auto_random == direct_random

	w0 = reshape(collect(1.0:36.0), 12, 3)
	h0 = reshape(collect(1.0:15.0), 3, 5)
	w, ht = BigRiverEssence._nmf_initialize(
		X, 3, :custom, MersenneTwister(134), w0, h0,
	)
	@test w == w0
	@test ht == transpose(h0)
	@test w !== w0
	@test !Base.mightalias(ht, h0)
	w[1, 1] = -1.0
	ht[1, 1] = -1.0
	@test w0[1, 1] > 0
	@test h0[1, 1] > 0

	@test_throws ArgumentError BigRiverEssence._nmf_initialize(
		X, 3, :unknown, MersenneTwister(135), nothing, nothing)
	@test_throws ArgumentError BigRiverEssence._nmf_initialize(
		X, 3, :custom, MersenneTwister(135), nothing, h0)
	@test_throws ArgumentError BigRiverEssence._nmf_initialize(
		X, 3, :custom, MersenneTwister(135), w0, nothing)
	@test_throws DimensionMismatch BigRiverEssence._nmf_initialize(
		X, 3, :custom, MersenneTwister(135), ones(11, 3), h0)
	@test_throws DimensionMismatch BigRiverEssence._nmf_initialize(
		X, 3, :custom, MersenneTwister(135), w0, ones(3, 4))
	@test_throws ArgumentError BigRiverEssence._nmf_initialize(
		X, 3, :custom, MersenneTwister(135), fill(NaN, 12, 3), h0)
	@test_throws ArgumentError BigRiverEssence._nmf_initialize(
		X, 3, :custom, MersenneTwister(135), w0, fill(Inf, 3, 5))
	@test_throws ArgumentError BigRiverEssence._nmf_initialize(
		X, 3, :custom, MersenneTwister(135), -w0, h0)
	@test_throws ArgumentError BigRiverEssence._nmf_initialize(
		X, 3, :custom, MersenneTwister(135), w0, -h0)
	@test_throws ArgumentError BigRiverEssence._nmf_initialize(
		X, 3, :random, MersenneTwister(135), w0, h0)
end

@testset "_nmf_cd_sweep!" begin
	A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
	F0 = [0.5 1.0; 1.5 0.25; 0.75 1.25]
	G = [1.0 0.5; 0.25 2.0]
	expected_cross = [1.5 4.5; 4.0 9.5; 6.5 14.5]
	cases = (
		(
			order = [1, 2],
			l1 = 0.0,
			l2 = 0.0,
			F = [
				0.47058823529411764 0.9480968858131488
				3.5294117647058822  1.4048442906574394
				4.9411764705882355  2.2491349480968856
			],
			gram = [1.0625 1.0; 1.0 4.25],
			violation = 16.015625,
		),
		(
			order = [2, 1],
			l1 = 0.0,
			l2 = 0.0,
			F = [
				0.5259515570934257 0.9411764705882353
				1.9930795847750864 1.8823529411764706
				3.0726643598615917 3.235294117647059
			],
			gram = [1.0625 1.0; 1.0 4.25],
			violation = 18.644301470588236,
		),
		(
			order = [1, 2],
			l1 = 0.3,
			l2 = 0.7,
			F = [
				0.11347517730496454 0.8255605702414213
				1.9574468085106385  1.4631420588867394
				2.808510638297873   2.301310982162046
			],
			gram = [1.7625 1.0; 1.0 4.95],
			violation = 17.188142730496452,
		),
	)

	for case in cases
		F = copy(F0)
		gram = Matrix{Float64}(undef, 2, 2)
		cross = Matrix{Float64}(undef, 3, 2)
		gradient = Vector{Float64}(undef, 3)
		A_original = copy(A)
		G_original = copy(G)
		violation = BigRiverEssence._nmf_cd_sweep!(
			A, F, G, gram, cross, gradient, case.order, case.l1, case.l2,
		)
		@test F ≈ case.F atol = tol_ord
		@test gram ≈ case.gram atol = tol_ord
		@test cross ≈ expected_cross atol = tol_ord
		@test violation ≈ case.violation atol = tol_ord
		@test all(F .>= 0)
		@test A == A_original
		@test G == G_original
	end

	F = copy(F0)
	gram = zeros(2, 2)
	cross = zeros(3, 2)
	gradient = zeros(3)
	violation = BigRiverEssence._nmf_cd_sweep!(
		A, F, zeros(2, 2), gram, cross, gradient, [1, 2], 0.0, 0.0,
	)
	@test F == F0
	@test iszero(violation)
	@test all(iszero, gram)
	@test all(iszero, cross)
end

@testset "_nmf_fit_cd!" begin
	X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
	w0 = [0.5 1.0; 1.5 0.25; 0.75 1.25]
	ht0 = [1.0 0.5; 0.25 2.0]

	expected_w = [
		0.3168316831683168 0.9175659138947603
		2.891089108910891  1.4626766047391258
		4.079207920792079  2.319279118923128
	]
	expected_ht = [
		0.8665537010974804 0.5122298987305688
		0.3299279530133637 1.9050525324340581
	]
	w = copy(w0)
	ht = copy(ht0)
	niter, converged = BigRiverEssence._nmf_fit_cd!(
		X, w, ht, 0.1, 0.3, 0.2, 0.4;
		tol = 0.0, maxiter = 1, shuffle = false,
		rng = MersenneTwister(140),
	)
	@test niter == 1
	@test !converged
	@test w ≈ expected_w atol = tol_ord
	@test ht ≈ expected_ht atol = tol_ord

	w = copy(w0)
	ht = copy(ht0)
	niter, converged = BigRiverEssence._nmf_fit_cd!(
		X, w, ht, 0.1, 0.0, 0.2, 0.0;
		tol = 0.0, maxiter = 1, shuffle = false,
		rng = MersenneTwister(141), update_h = false,
	)
	@test niter == 1
	@test !converged
	@test w ≈ expected_w atol = tol_ord
	@test ht == ht0

	zw = zeros(3, 2)
	zht = zeros(2, 2)
	niter, converged = BigRiverEssence._nmf_fit_cd!(
		zeros(3, 2), zw, zht, 0.0, 0.0, 0.0, 0.0;
		tol = 1e-4, maxiter = 10, shuffle = false,
		rng = MersenneTwister(142),
	)
	@test niter == 1
	@test converged
	@test all(iszero, zw)
	@test all(iszero, zht)

	w1, ht1 = copy(w0), copy(ht0)
	w2, ht2 = copy(w0), copy(ht0)
	r1 = BigRiverEssence._nmf_fit_cd!(
		X, w1, ht1, 0.0, 0.0, 0.0, 0.0;
		tol = 0.0, maxiter = 3, shuffle = true,
		rng = MersenneTwister(143),
	)
	r2 = BigRiverEssence._nmf_fit_cd!(
		X, w2, ht2, 0.0, 0.0, 0.0, 0.0;
		tol = 0.0, maxiter = 3, shuffle = true,
		rng = MersenneTwister(143),
	)
	@test r1 == r2
	@test w1 == w2
	@test ht1 == ht2
end

@testset "_nmf_reconstruction_error" begin
	w = [1.0 2.0; 3.0 4.0; 5.0 6.0]
	h = [0.5 1.5; 2.5 3.5]
	X = w * h
	w_original, h_original, X_original = copy(w), copy(h), copy(X)
	@test iszero(BigRiverEssence._nmf_reconstruction_error(X, w, h))
	@test BigRiverEssence._nmf_reconstruction_error(
		zeros(size(X)), w, h) ≈ norm(w * h) atol = tol_ord

	X_perturbed = copy(X)
	X_perturbed[2, 2] += 1e-12
	@test BigRiverEssence._nmf_reconstruction_error(
		X_perturbed, w, h) ≈ norm(X_perturbed - w * h) atol = eps()
	@test w == w_original
	@test h == h_original
	@test X == X_original
end

@testset "public fit and invariants" begin
	rng = MersenneTwister(150)
	wtrue = rand(rng, 80, 3)
	htrue = rand(rng, 3, 12)
	X = wtrue * htrue
	m = nmf(X; k = 3, init = :nndsvda, rng = MersenneTwister(151),
		tol = 1e-6, maxiter = 3000)

	@test m isa Nmf
	@test size(m.w) == (80, 3)
	@test size(m.h) == (3, 12)
	@test all(isfinite, m.w) && all(isfinite, m.h)
	@test all(m.w .>= 0)
	@test all(m.h .>= 0)
	@test m.reconstruction_error >= 0
	@test m.reconstruction_error ≈ norm(X - m.w * m.h) atol = tol_ord
	@test 1 <= m.niter <= 3000
	@test m.converged
	@test m.alpha_w == 0
	@test m.alpha_h == 0
	@test m.l1_ratio == 0
	@test m.reconstruction_error / norm(X) < 1e-4

	default_rank = nmf([1.0 2.0; 3.0 4.0; 5.0 6.0];
		init = :random, rng = MersenneTwister(152), maxiter = 500)
	@test size(default_rank.w) == (3, 2)
	@test size(default_rank.h) == (2, 2)

	wide_rank = nmf([1.0 2.0; 3.0 4.0]; k = 3,
		init = :random, rng = MersenneTwister(153), maxiter = 20)
	@test size(wide_rank.w) == (2, 3)
	@test size(wide_rank.h) == (3, 2)

	regularized = nmf(rand(MersenneTwister(154), 20, 8); k = 3,
		init = :random, rng = MersenneTwister(155), alpha_w = 0.2,
		l1_ratio = 0.4, maxiter = 500)
	@test regularized.alpha_w == 0.2
	@test regularized.alpha_h == 0.2
	@test regularized.l1_ratio == 0.4

	nonconverged = @test_logs (:warn, r"NMF reached maxiter=1") nmf(
		rand(MersenneTwister(156), 20, 8); k = 3, init = :random,
		rng = MersenneTwister(157), tol = 0.0, maxiter = 1,
	)
	@test nonconverged.niter == 1
	@test !nonconverged.converged
end

@testset "scikit-learn coordinate-descent reference" begin
	# Generated with sklearn.decomposition.NMF 1.9.0 using solver="cd",
	# shuffle=false, tol=1e-10, max_iter=1000, and these exact custom factors.
	# A custom start isolates the Fast HALS solver from differences in SVD routines.
	X = [1.0 1.0; 2.0 1.0; 3.0 1.2; 4.0 1.0; 5.0 0.8; 6.0 1.0]
	w0 = [0.6 0.2; 0.5 0.4; 0.4 0.6; 0.3 0.8; 0.2 1.0; 0.1 1.2]
	h0 = [1.2 0.4; 0.3 1.1]
	expected_w = [
		0.7110319213215501 0.4792680537090550
		1.4220638426603567 0.3879789005371095
		2.1330957639957120 0.4108011887413642
		2.8441276853379693 0.2054005941932187
		3.5551596065857160 0.0
		4.2661915280155820 0.0228222878493277
	]
	expected_h = [
		1.4064066183271926 0.2250250590588947
		0.0                1.7526726295863762
	]

	m = nmf(X; k = 2, init = :custom, w_init = w0, h_init = h0,
		tol = 1e-10, maxiter = 1000)
	@test isapprox(m.w, expected_w; atol = tol_julia)
	@test isapprox(m.h, expected_h; atol = tol_julia)
	@test m.reconstruction_error < tol_julia
	@test m.converged

	wnew = nmf_transform(m, [1.5 0.9; 4.5 1.1]; tol = 1e-10, maxiter = 1000)
	expected_wnew = [
		1.066547881995003 0.3765677564216273
		3.199643645983937 0.2168117383591060
	]
	@test isapprox(wnew, expected_wnew; atol = tol_julia)
end

@testset "LowRankModels.NNMF Julia reference" begin
	# These fixtures are generated by test/Data/NMF/generate_nmf_reference.jl,
	# which calls LowRankModels.NNMF from src/scikitlearn.jl. LowRankModels 1.1.1
	# uses a different optimizer (alternating proximal gradient), so compare the
	# identifiable reconstructed matrix rather than non-unique raw factors.
	reference_dir = joinpath(@__DIR__, "Data", "NMF")
	X = readdlm(joinpath(reference_dir, "X.csv"), ',', Float64)
	reference_w = readdlm(joinpath(reference_dir, "W.csv"), ',', Float64)
	reference_h = readdlm(joinpath(reference_dir, "H.csv"), ',', Float64)
	meta = readdlm(joinpath(reference_dir, "meta.csv"), ',')
	n = Int(meta[findfirst(==("n"), meta[:, 1]), 2])
	p = Int(meta[findfirst(==("p"), meta[:, 1]), 2])
	k = Int(meta[findfirst(==("k"), meta[:, 1]), 2])
	ours = nmf(X; k = k, init = :nndsvda, tol = 1e-7, maxiter = 2000)
	ours_reconstruction = ours.w * ours.h
	reference_reconstruction = reference_w * reference_h
	ours_relative_error = norm(X - ours_reconstruction) / norm(X)
	reference_relative_error = norm(X - reference_reconstruction) / norm(X)

	@test size(X) == (n, p)
	@test size(reference_w) == (n, k)
	@test size(reference_h) == (k, p)
	@test size(reference_w) == size(ours.w)
	@test size(reference_h) == size(ours.h)
	@test all(reference_w .>= 0)
	@test all(reference_h .>= 0)
	@test ours_relative_error < 1e-4
	@test reference_relative_error < tol_julia
	@test abs(ours_relative_error - reference_relative_error) < 1e-4
	@test norm(ours_reconstruction - reference_reconstruction) / norm(X) < 1e-4
end

@testset "regularization and reproducibility" begin
	rng = MersenneTwister(170)
	X = rand(rng, 50, 10)
	w0 = rand(rng, 50, 4)
	h0 = rand(rng, 4, 10)
	unregularized = nmf(X; k = 4, init = :custom, w_init = w0, h_init = h0,
		tol = 1e-4, maxiter = 1000)
	l1fit = nmf(X; k = 4, init = :custom, w_init = w0, h_init = h0,
		alpha_w = 0.01, alpha_h = 0.01, l1_ratio = 1.0,
		tol = 1e-4, maxiter = 1000)
	@test l1fit.alpha_w == 0.01
	@test l1fit.alpha_h == 0.01
	@test l1fit.l1_ratio == 1.0
	@test count(iszero, l1fit.w) + count(iszero, l1fit.h) >=
		count(iszero, unregularized.w) + count(iszero, unregularized.h)

	X = rand(MersenneTwister(171), 40, 8)
	a = nmf(X; k = 3, init = :random, rng = MersenneTwister(172),
		tol = 1e-5, maxiter = 500, shuffle = true)
	b = nmf(X; k = 3, init = :random, rng = MersenneTwister(172),
		tol = 1e-5, maxiter = 500, shuffle = true)
	@test a.w == b.w
	@test a.h == b.h
	@test a.niter == b.niter
end

@testset "nmf_transform and nmf_invtransform" begin
	rng = MersenneTwister(180)
	X = rand(rng, 70, 9)
	m = nmf(X; k = 4, init = :nndsvda, tol = 1e-5, maxiter = 500)
	h_original = copy(m.h)
	wnew = nmf_transform(m, X; tol = 1e-5, maxiter = 500)
	reconstructed = nmf_invtransform(m, wnew)
	@test size(wnew) == (70, 4)
	@test size(reconstructed) == size(X)
	@test all(isfinite, wnew)
	@test all(wnew .>= 0)
	@test norm(X - reconstructed) <= norm(X)
	@test m.h == h_original
	@test reconstructed == wnew * m.h

	manual_model = Nmf(ones(3, 2), [1.0 0.5 0.25; 0.2 0.8 1.2],
		0.0, 1, true, 0.1, 0.1, 0.4)
	Xnew = [1.0 2.0 3.0; 2.0 1.0 0.5]
	manual_w = zeros(2, 2)
	manual_ht = transpose(manual_model.h)
	BigRiverEssence._nmf_fit_cd!(
		Xnew, manual_w, manual_ht, 3 * 0.1 * 0.4, 0.0,
		3 * 0.1 * 0.6, 0.0; tol = 0.0, maxiter = 2,
		shuffle = false, rng = MersenneTwister(181), update_h = false,
	)
	transformed = @test_logs (:warn, r"NMF transform reached maxiter=2") begin
		nmf_transform(manual_model, Xnew; tol = 0.0, maxiter = 2)
	end
	@test transformed ≈ manual_w atol = tol_ord

	@test_throws DimensionMismatch nmf_transform(m, rand(5, 8))
	@test_throws ArgumentError nmf_transform(m, [-1.0 zeros(1, 8)...])
	@test_throws ArgumentError nmf_transform(m, [Inf zeros(1, 8)...])
	@test_throws ArgumentError nmf_transform(m, X; tol = -1)
	@test_throws ArgumentError nmf_transform(m, X; tol = Inf)
	@test_throws ArgumentError nmf_transform(m, X; tol = NaN)
	@test_throws ArgumentError nmf_transform(m, X; maxiter = 0)
	@test_throws DimensionMismatch nmf_invtransform(m, rand(5, 3))
end

@testset "zero matrix" begin
	for init in (:auto, :random, :nndsvd, :nndsvda, :nndsvdar)
		m = nmf(zeros(10, 6); k = 3, init = init,
			rng = MersenneTwister(190))
		@test m.converged
		@test m.niter == 1
		@test iszero(m.reconstruction_error)
		@test all(iszero, m.w)
		@test all(iszero, m.h)
	end
end

@testset "public argument validation" begin
	X = ones(8, 5)
	@test_throws ArgumentError nmf(zeros(0, 5); k = 1)
	@test_throws ArgumentError nmf(zeros(5, 0); k = 1)
	@test_throws ArgumentError nmf([-1.0 2.0; 3.0 4.0]; k = 1)
	@test_throws ArgumentError nmf([1.0 NaN; 2.0 3.0]; k = 1)
	@test_throws ArgumentError nmf([1.0 Inf; 2.0 3.0]; k = 1)
	@test_throws ArgumentError nmf(X; k = 0)
	@test_throws ArgumentError nmf(X; tol = -1)
	@test_throws ArgumentError nmf(X; tol = Inf)
	@test_throws ArgumentError nmf(X; tol = NaN)
	@test_throws ArgumentError nmf(X; maxiter = 0)
	@test_throws ArgumentError nmf(X; alpha_w = -1)
	@test_throws ArgumentError nmf(X; alpha_w = Inf)
	@test_throws ArgumentError nmf(X; alpha_w = NaN)
	@test_throws ArgumentError nmf(X; alpha_h = -1)
	@test_throws ArgumentError nmf(X; alpha_h = Inf)
	@test_throws ArgumentError nmf(X; alpha_h = NaN)
	@test_throws ArgumentError nmf(X; l1_ratio = -0.1)
	@test_throws ArgumentError nmf(X; l1_ratio = 1.1)
	@test_throws ArgumentError nmf(X; l1_ratio = Inf)
	@test_throws ArgumentError nmf(X; l1_ratio = NaN)
	@test_throws ArgumentError nmf(X; init = :unknown)
	@test_throws ArgumentError nmf(X; k = 6, init = :nndsvd)
	@test_throws ArgumentError nmf(X; init = :custom)
	@test_throws ArgumentError nmf(X; k = 2, init = :custom,
		w_init = ones(8, 2))
	@test_throws ArgumentError nmf(X; k = 2, init = :custom,
		h_init = ones(2, 5))
	@test_throws DimensionMismatch nmf(X; k = 2, init = :custom,
		w_init = ones(7, 2), h_init = ones(2, 5))
	@test_throws DimensionMismatch nmf(X; k = 2, init = :custom,
		w_init = ones(8, 2), h_init = ones(2, 4))
	@test_throws ArgumentError nmf(X; k = 2, init = :custom,
		w_init = -ones(8, 2), h_init = ones(2, 5))
	@test_throws ArgumentError nmf(X; k = 2, init = :custom,
		w_init = ones(8, 2), h_init = -ones(2, 5))
	@test_throws ArgumentError nmf(X; k = 2, init = :custom,
		w_init = fill(NaN, 8, 2), h_init = ones(2, 5))
	@test_throws ArgumentError nmf(X; k = 2, init = :custom,
		w_init = ones(8, 2), h_init = fill(Inf, 2, 5))
	@test_throws ArgumentError nmf(X; k = 2, init = :random,
		w_init = ones(8, 2), h_init = ones(2, 5))
end
