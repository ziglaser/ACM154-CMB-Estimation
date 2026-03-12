include("ensemble_kalman_sampling.jl")

using PyCall
using LinearAlgebra
using Statistics
using Printf

pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, ".."))  # generate_cosmopower_unlensed_maps.py
pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)                  # nonlensed_EKI.py

np       = pyimport("numpy")
unlensed = pyimport("generate_cosmopower_unlensed_maps")
nonlensed = pyimport("CMB_EKI")

N_BINS = 50
N_OBS  = 1

data = np.load(joinpath(@__DIR__, "..", "..", "data", "cmb_fiducial_dataset.npz"))
f = convert(Array{Float64,3}, pycall(data.__getitem__, PyObject, "f"))

y_obs_ps = zeros(Float64, N_OBS, N_BINS)
for i in 1:N_OBS
    ps = unlensed.compute_power_spectrum(f[:, :, i], n_bins=N_BINS)
    y_obs_ps[i, :] = convert(Vector{Float64}, ps)
end

y_flat = collect(vec(y_obs_ps'))
Gamma = convert(Matrix{Float64}, nonlensed.analytic_gamma_ps(N_OBS))

const PARAM_LO = Float64[67.37 - 40, 0.001, 0.001]
const PARAM_HI = Float64[67.37 + 40, 0.24,  0.04466]

function theory_model(theta::Vector{Float64})::Vector{Float64}
    h0    = theta[1]
    omch2 = theta[2]
    ombh2 = theta[3]
    cl_py = unlensed.generate_cosmopower_theory_spectrum(
        h0=h0, omch2=omch2, ombh2=ombh2,
        noise_level=0.08, n_bins=N_BINS)
    cl = convert(Vector{Float64}, cl_py)
    return repeat(cl, N_OBS)
end

function forward_model(theta::Vector{Float64})::Vector{Float64}
    h0    = theta[1]
    omch2 = theta[2]
    ombh2 = theta[3]
    cmb_map = unlensed.generate_cosmopower_map(seed=1,
        h0=h0, omch2=omch2, ombh2=ombh2,
        noise_level=0.08)
    cl_py = unlensed.compute_power_spectrum(cmb_map, n_bins=N_BINS)
    cl = convert(Vector{Float64}, cl_py)
    return repeat(cl, N_OBS)
end

# ens_func: (N_ens × N_θ) → (N_ens × k) as expected by update_ensemble!
function ens_func(theta_ens::Matrix{Float64})
    N_ens = size(theta_ens, 1)
    g = zeros(Float64, N_ens, N_OBS * N_BINS)
    for j in 1:N_ens
        g[j, :] = forward_model(theta_ens[j, :])
    end
    return g
end

function main()
    theta_mean     = Float64[67.37, 0.1198, 0.02233]
    prior_std      = Float64[20.0, 0.03, 0.005]
    prior_cov_sqrt = diagm(prior_std) 

    N_ens = 500
    Δt = 0.5
    α_reg = 1.0
    update_freq = 1
    N_iter = 10

    tau_mean, tau_std = nonlensed.compute_tau(y_flat, Gamma, theta_mean, N_OBS, stochastic_n=500)
    τ = Float64(tau_mean)
    @printf "Computed τ = %.4g ± %.4g  (from %d stochastic samples)\n" τ Float64(tau_std) 500


    ekiobj = EKIObj("EKI",
                    ["h0", "omch2", "ombh2"],
                    N_ens,
                    theta_mean,
                    prior_cov_sqrt,
                    theta_mean,
                    prior_cov_sqrt,
                    y_flat,
                    Gamma,
                    Δt,
                    α_reg,
                    update_freq,
                    PARAM_LO,
                    PARAM_HI)

    gamma_diag = diag(Gamma)

    iter_times = Float64[]
    for i in 1:N_iter
        t0 = time()
        update_ensemble!(ekiobj, ens_func)
        push!(iter_times, time() - t0)
        m   = vec(mean(ekiobj.θ[end], dims=1))
        std = vec(Statistics.std(ekiobj.θ[end], dims=1))
        @printf "Iter %3d:  h0=%.2f±%.2f  omch2=%.4f±%.4f  ombh2=%.5f±%.5f\n" i m[1] std[1] m[2] std[2] m[3] std[3]

        residual    = forward_model(m) - y_flat
        discrepancy = sqrt(sum(residual.^2 ./ gamma_diag))
        @printf "  discrepancy = %.4g  (τ = %.2g)\n" discrepancy τ
        if discrepancy <= τ
            println("  Discrepancy principle satisfied at iteration $i, stopping.")
            break
        end
    end

    total_time = sum(iter_times)
    avg_time   = total_time / length(iter_times)
    @printf "\nTotal time: %.3fs  |  Time per iteration: %.3fs  (%d iterations)\n" total_time avg_time length(iter_times)

    println("Final ensemble mean: ", vec(mean(ekiobj.θ[end], dims=1)))

    # Save ensemble
    n_snaps = length(ekiobj.θ)
    theta_snaps = [np.array(Matrix{Float64}(ekiobj.θ[i])) for i in 1:n_snaps]
    theta_3d = np.stack(theta_snaps, axis=0)   # shape: (n_snaps, N_ens, N_θ)
    out_path = joinpath(@__DIR__, "..", "..", "data", "eks_theta_history.npy")
    np.save(out_path, theta_3d)
end

main()
