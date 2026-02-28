#README: the ENV variable below should be changed to your local installation of python with camb installed.

using NPZ
using Base.Threads
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"  # Don't use CondaPkg's conda
ENV["JULIA_PYTHONCALL_EXE"] = "/Users/nferree/miniconda3/envs/acm_154/bin/python" #change this to your python bin that has the python package camb installed.
using PythonCall
using CMBLensing
using Plots
using Random
using Distributions


"""
make_single_sim(θpix, Nside) -> (f, f̃)

Creates a single pair of unlensed and lensed CMB temperature realizations.

# Arguments
- `θpix::Real`: Pixel size in arcminutes
- `Nside::Int`: Number of pixels per side in the map
- 'cosmo_params::NamedTuple': cosmological parameters for map generation.
Our inference parameters are ωb (aka ombh2), ωc (aka omch2), and H0. The others are needed for the code to run,
but should be left at their default values.
- 'pixel_noise_level::Float' : standard deviation of white noise in pixel space. Leave at default level.
- 'rng': random number generator for the realizations of CMB, noise, etc.
- 'plot::bool' : if true, the lensed map will be saved as a png.

# Returns
- `f`: Unlensed CMB map in pixel space
- `f̃`: Lensed CMB map in pixel space (obtained by applying the lensing operator to `f`)

"""
function make_single_sim(θpix, Nside, cosmo_params, pixel_noise_level, rng, plot = false) 
    (;ds, f, ϕ, proj) = load_sim(
        θpix  = θpix,
        Nside = Nside,
        fiducial_θ = cosmo_params;
        T     = Float32,
        pol   = :I,
    )

    white_noise = pixel_noise_level * rand(rng, Normal(0,1), Nside, Nside)
    noise_map = FlatMap(white_noise,  θpix =  θpix)
    f̃ = LenseFlow(ϕ) * f + noise_map
    if plot
        plt = heatmap(f̃)
        savefig(plt, "julia_lensed_sim.png")
    end
    return f[:Ix], f̃[:Ix]
end

"""
make_all_sis(n, θpix, Nside) -> None

Creates a dataset of unlensed and lensed CMB temperature realizations. They are saved in a .npz file, all_simulations.npz.
WARNING: these unlensed simulations (f) shouldn't be used for any inputs - if you want those compatible with MCMC and logpdfs,
use the python code. The f are just for visualization purposes. The f_tilde are the lensed maps that should be used.

# Arguments
- `n::Int`: Number of CMB realizations
- `θpix::Real`: Pixel size in arcminutes
- `Nside::Int`: Number of pixels per side in the map. Called npix in the python code
- 'cosmo_params::NamedTuple': cosmological parameters for map generation.
Our inference parameters are ωb (aka ombh2), ωc (aka omch2), and H0. The others are needed for the code to run,
but should be left at their default values.
- 'pixel_noise_level::Float' : standard deviation of white noise in pixel space. Leave at default level.
- 'rng': random number generator for the realizations of CMB, noise, etc.

"""
function make_all_sims(n, θpix, Nside, cosmo_params, pixel_noise_level, rng)
    # Pre-allocate arrays
    f_all = Vector{Any}(undef, n)
    f̃_all = Vector{Any}(undef, n)
        # Generate in parallel
    Threads.@threads for i in 1:n
        f, f̃ = make_single_sim(θpix, Nside, cosmo_params, pixel_noise_level, rng, false)
        f_all[i] = f
        f̃_all[i] = f̃
    end
    
    # Stack into 3D arrays
    f_stacked = cat(f_all..., dims=3)
    f̃_stacked = cat(f̃_all..., dims=3)

    # Save single file
    npzwrite("all_CMB_simulations.npz", Dict(
        "f" => f_stacked, 
        "f_tilde" => f̃_stacked
    ))
end


function main()
    seed = 0 #rng seed
    nsims = 10 #number of data sets
    npix = 64 #number of pixels per side (in a square map)
    pix_size = 8 #pixel size in arcmin. Do not set below 8 - this will cause problems in python compatibility.
    radian_per_arcmin = π/(60*180) #conversion factor
    radian_pixel_size = pix_size * radian_per_arcmin
    rng = MersenneTwister(seed) #rng instance
    jax_to_julia_factor = 10^6 / radian_pixel_size
    pixel_noise_level = 10^-8 * jax_to_julia_factor #standard deviation of pixel noise
    cosmo_params = (ωb = 0.02233, ωc = 0.1198, H0 = 67.37, τ = 0.0540, nₛ = .9652, logA = 3.0381498999763017, θs = nothing); #omega b is ombh2, omegac is omch2
    #θpix, Nside, cosmo_params, pixel_noise_level, rng
    @time make_single_sim(pix_size, npix, cosmo_params, pixel_noise_level, rng, true);
    @time make_all_sims(nsims, pix_size, npix, cosmo_params, pixel_noise_level, rng);
end

main()