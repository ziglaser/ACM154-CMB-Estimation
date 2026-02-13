using NPZ
using Base.Threads
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"  # Don't use CondaPkg's conda
ENV["JULIA_PYTHONCALL_EXE"] = "/Users/nferree/miniconda3/envs/acm_154/bin/python"
using PythonCall
using CMBLensing


"""
make_single_sim(θpix, Nside) -> (f, f̃)

Creates a single pair of unlensed and lensed CMB temperature realizations.

# Arguments
- `θpix::Real`: Pixel size in arcminutes
- `Nside::Int`: Number of pixels per side in the map

# Returns
- `f`: Unlensed CMB map in pixel space
- `f̃`: Lensed CMB map in pixel space (obtained by applying the lensing operator to `f`)

"""
function make_single_sim(θpix, Nside, cosmo_params) #note: this can also be modified to accept fiducial theta
    #fiducial_θ = (;)` — NamedTuple of keyword arguments passed to `camb()` for the fiducial model.
    #the function camb takes the arguments:
    #    ℓmax = 6000, 
    #=r = 0.2, ωb = 0.0224567, ωc = 0.118489, τ = 0.055, Σmν = 0.06,
    θs = 0.0104098, H0 = nothing, logA = 3.043, nₛ = 0.968602, nₜ = -r/8,
    AL = 1,
    k_pivot = 0.002=#
    #instead of this, I think we can do θs = nothing and H0 = as desired
    (;ds, f, ϕ) = load_sim(
        θpix  = θpix,
        Nside = Nside,
        fiducial_θ = cosmo_params;
        T     = Float32,
        pol   = :I,
    )
    f̃ = LenseFlow(ϕ) * f
    return f[:Ix], f̃[:Ix]
end

"""
make_all_sis(n, θpix, Nside) -> None

Creates a dataset of unlensed and lensed CMB temperature realizations. They are saved in a .npz file, all_simulations.npz.

# Arguments
- `n::Int`: Number of CMB realizations
- `θpix::Real`: Pixel size in arcminutes
- `Nside::Int`: Number of pixels per side in the map
"""
function make_all_sims(n, θpix, Nside, cosmo_params)
    # Pre-allocate arrays
    f_all = Vector{Any}(undef, n)
    f̃_all = Vector{Any}(undef, n)
        # Generate in parallel
    Threads.@threads for i in 1:n
        f, f̃ = make_single_sim(θpix, Nside, cosmo_params)
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
    nsims = 100 #number of data sets
    npix = 64 #number of pixels per side (in a square map)
    pix_size = 2 #pixel size in arcmin
    cosmo_params = (ωb = 0.0224567, ωc = 0.118489, H0 = 67.7, θs = nothing);
    @time make_single_sim(pix_size, npix, cosmo_params);
    @time make_all_sims(nsims, pix_size, npix, cosmo_params);