using Distributed
addprocs(127)
@everywhere using AutoBZCore
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
using Optim
using HDF5

@everywhere function f(ek, beta, mu)
	return 1/(exp(beta*(ek-mu))+1)
end

#@everywhere gloc_integrand((kx,ky,kz), (; h, f, q, η, ω, beta, mu)) = -(f(h(kx+0.5,ky+0.5,kz+q), beta, mu) - f(h(kx,ky,kz), beta, mu))/(h(kx+0.5,ky+0.5,kz+q) - h(kx,ky,kz) + complex(ω,η))
@everywhere function gloc_integrand((kx, ky, kz), (; h, f, q, η, ω, beta, mu))
    h1 = h(kx + 0.5, ky + 0.5, kz + q)
    h0 = h(kx, ky, kz)
    denom = h1 - h0
    if abs(denom) < 1e-8
        return 0.0
    else
        return -(f(h1, beta, mu) - f(h0, beta, mu)) / denom
    end
end


@everywhere h(kx,ky,kz,t=1.,tp=0.0) = -2*t*(cos(2pi*kx) + cos(2pi*ky) + cos(2pi*kz)) - 4*tp * (cos(2pi*kx)*cos(2pi*ky) + cos(2pi*ky)*cos(2pi*kz) + cos(2pi*kz)*cos(2pi*kx))

@everywhere function chi_w(args)
	res = solve!(args).value
	return res
end

ns = [0.74]
betas = [60,90,120,140,160,180,200]
nk = 1000
nq = 1000
qpoints = range(0.0, 2pi, nq+1)[1:end-1]
err = 1e-8

# ek
ek = Array{Float64}(undef, nk, nk, nk)
k = collect(LinRange(0,1,nk+1)[1:nk])
for (l,kx) in enumerate(k)
	for (m,ky) in enumerate(k)
		for (n,kz) in enumerate(k)
			ek[l,m,n] = h(kx,ky,kz)
		end
	end
end

s = time()

# hdf5 file for saving
print("Creating file to save data", "\n")
save_file = "./rpa_3d_julia_nq1000.hdf5"
fid = h5open(save_file, "w")

# RPA loop for all densities and betas
for (i_n,n) in enumerate(ns[:])
	print("Starting Calculation for n = ", n, "\n")

	# create group for density
	nstrg = "n"*string(Int(n*100))
	create_group(fid, nstrg)
	qmax_b = []
	for (i_b,beta) in enumerate(betas)
		s_beta = time()
		print("RPA for beta = ", beta, "\n")

		# optimize for chemical potential
		function density(x)
			mu = x
			res = sum(f.(ek, beta, mu))
			return abs(res/nk^3 - n/2)
		end

		x0 = 0.0
		res = optimize(x->density(first(x)), [x0])
		mu = Optim.minimizer(res)[1]

		# solve for all q	
		args = []
		for q in qpoints
			bz = load_bz(FBZ(3), 2pi*I(3))
			p = (; h=h, f=f, q=q/2pi, η=0.0, ω=0.0, beta=beta, mu=mu)
			prob = AutoBZProblem(TrivialRep(), IntegralFunction(gloc_integrand), bz, p)
			alg = IAI()
			solver = init(prob, alg; abstol=err)
			push!(args, solver)
		end

		chi = pmap(chi_w, args)
		qmax = qpoints[argmax(real.(chi))]
		push!(qmax_b, qmax)

		# save data
		bstrg = "b"*string(Int(beta))
		g = fid[nstrg]
		g[bstrg] = chi
		e_beta = time()
		print("calc time for ",bstrg, " ", e_beta-s_beta, "\n")
	end
	# save all qmax for given n
        bstrg = "qmax_beta"
        g = fid[nstrg]
	qmax_b = Float64.(qmax_b)
        g[bstrg] = qmax_b
end

e = time()

print("Finished Calculation ...", "\n")
print("Time of all RPA Calculations:", e-s, "\n")
