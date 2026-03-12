using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

# generate ensemble
Random.seed!(123)

function MvNormal_sqrt(N_ens, θ_mean::Array{FT,1}, θθ_cov_sqrt) where {FT<:AbstractFloat}
    
    
    N_θ, N_r = size(θθ_cov_sqrt)
    θ = zeros(FT, N_ens, N_θ)
    for i = 1: N_ens
        θ[i,     :] = θ_mean + θθ_cov_sqrt * rand(Normal(0, 1), N_r)
    end
    
    return θ
end
"""
EKIObj{FT<:AbstractFloat, IT<:Int}
Structure that is used in Ensemble Kalman Inversion (EKI)
#Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct EKIObj{FT<:AbstractFloat, IT<:Int}
    "filter_type type"
    filter_type::String
    "vector of parameter names"
    θ_names::Array{String, 1}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of parameters is added)"
    θ::Vector{Array{FT, 2}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
    y_pred::Vector{Array{FT, 1}}
    "vector of observations (length: N_y)"
    y::Array{FT, 1}
    "covariance of the observational noise"
    Σ_η
    "number ensemble size (2N_θ - 1)"
    N_ens::IT
    "size of θ"
    N_θ::IT
    "size of y"
    N_y::IT
    "Covariance matrix square root of the evolution error"
    Z_ω::Union{Array{FT, 2}, Nothing}
    "time step"
    Δt::FT
    "regularization parameter"
    α_reg::FT
    "regularization vector"
    r::Array{FT, 1}
    "update frequency"
    update_freq::IT
    "current iteration number"
    iter::IT
    "optional parameter lower bounds (length N_θ), applied after prediction step"
    param_lo::Union{Vector{FT}, Nothing}
    "optional parameter upper bounds (length N_θ), applied after prediction step"
    param_hi::Union{Vector{FT}, Nothing}
end

# outer constructors
function EKIObj(
    filter_type::String,
    θ_names::Array{String, 1},
    N_ens::IT,
    # initial condition
    θ0_mean::Array{FT},
    θθ0_cov_sqrt::Array{FT,2},
    # prior information
    prior_mean::Array{FT},
    prior_cov_sqrt::Array{FT,2},
    y::Array{FT, 1},
    Σ_η,
    Δt::FT,
    α_reg::FT = 1.0,
    update_freq::IT = 0,
    param_lo::Union{Vector{FT}, Nothing} = nothing,
    param_hi::Union{Vector{FT}, Nothing} = nothing) where {FT<:AbstractFloat, IT<:Int}
    
    ## check EKI hyperparameters
    Z_ω = sqrt(Δt/(1 - Δt) + 1 -  α_reg^2) * prior_cov_sqrt

    if update_freq > 0
        @assert(Δt > 0.0 && Δt < 1)
        @info "Start ", filter_type, " on the mean-field stochastic dynamical system for Bayesian inference "
        @assert(α_reg ≈ 1.0)

    elseif Δt ≈ 1.0 
        @info "Start original ", filter_type, " for optimization "
        @assert(α_reg ≈ 1.0 && update_freq == 0)
        Z_ω = nothing

    else
        @assert(Δt > 0.0 && Δt < 1)
        @info "Start ", filter_type, " on the regularized stochastic dynamical system for optimization "
        @assert(α_reg >= 0.0 && α_reg <= 1.0)
    end
    
    N_θ = size(θ0_mean,1)
    N_y = size(y, 1)
    
    
    # generate initial assemble
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    θ0 = MvNormal_sqrt(N_ens, θ0_mean, θθ0_cov_sqrt)
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    
    # prediction
    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s
    
    r = prior_mean
    iter = 0
    


    EKIObj{FT,IT}(filter_type,
    θ_names, θ,
    y_pred,
    y, Σ_η,
    N_ens, N_θ, N_y,
    Z_ω, Δt, α_reg, r, update_freq, iter,
    param_lo, param_hi)
end

"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(eki::EKIObj{FT}, x::Array{FT,2}, x_mean::Array{FT, 1}, y::Array{FT,2}, y_mean::Array{FT, 1}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = eki.N_ens, size(x_mean,1), size(y_mean,1)
    
    xy_cov = zeros(FT, N_x, N_y)
    
    for i = 1: N_ens
        xy_cov .+= (x[i,:] - x_mean)*(y[i,:] - y_mean)'
    end
    
    return xy_cov/(N_ens - 1)
end


"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov_sqrt(eki::EKIObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = eki.N_ens, size(x,2)
    x_mean = dropdims(mean(x, dims=1), dims=1)
    
    x_cov_sqrt = zeros(FT, N_x, N_ens)
    
    for i = 1: N_ens
        x_cov_sqrt[:, i] .+= (x[i,:] - x_mean)
    end
    
    return x_cov_sqrt/sqrt(N_ens - 1)
end


function trunc_svd(X,  ϵ = 1.0e-6)
    n_row, n_col = size(X)
    svd_X = svd(X)
    rank_X = min(n_row, n_col)
    for i = 1:min(n_row, n_col)
        if svd_X.S[i] <= ϵ*svd_X.S[1]
            rank_X = i - 1
            break
        end
    end

    return svd_X.U[:, 1:rank_X], svd_X.S[1:rank_X], svd_X.Vt[1:rank_X, :]'
end


function update_ensemble!(eki::EKIObj{FT}, ens_func::Function) where {FT<:AbstractFloat}

    eki.iter += 1

    filter_type = eki.filter_type
    N_ens, N_θ, N_y = eki.N_ens, eki.N_θ, eki.N_y
    r, α_reg, update_freq, Δt = eki.r, eki.α_reg, eki.update_freq, eki.Δt
    
    # θ: N_ens x N_θ
    θ = eki.θ[end]
    # compute the observation covariance matrices
    Σ_ν = (1/Δt) * eki.Σ_η
    

    ############# Prediction step
    θ_p = similar(θ)
    # θ mean at previous time step 
    θ_mean = dropdims(mean(θ, dims=1), dims=1)
    if update_freq > 0 && eki.iter % update_freq == 0 # deterministic update for mean-field evoluation
        θ_p_mean =  r + α_reg*(θ_mean  - r) 
        for j = 1:N_ens
            θ_p[j, :] .= θ_p_mean + sqrt(Δt/(1 - Δt) + 1 - α_reg^2 + 1)*(θ[j, :] - θ_mean)
        end

    else # stochastic update

        # generate evolution error
        noise = (eki.Z_ω === nothing ? zeros(N_ens, N_θ) : MvNormal_sqrt(N_ens, zeros(N_θ), eki.Z_ω))

        for j = 1:N_ens
            θ_p[j, :] .= α_reg*θ[j, :] + (1-α_reg)*r + noise[j, :]
        end

    end

    # clamp predicted particles to physical bounds before forward model evaluation
    if eki.param_lo !== nothing
        θ_p .= clamp.(θ_p, eki.param_lo', eki.param_hi')
    end

    θ_p_mean = dropdims(mean(θ_p, dims=1), dims=1)

    ############# Analysis step
    
    # evaluation G(θ)
    g = zeros(FT, N_ens, N_y)
    g .= ens_func(θ_p)
    g_mean = dropdims(mean(g, dims=1), dims=1)
    
    # construct square root matrix for  θ̂ - m̂
    Z_p_t = copy(θ_p)
    for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_mean;    end
    Z_p_t ./= sqrt(N_ens - 1)
    
    # construct square root matrix for  g - g_mean
    Y_p_t = copy(g)  
    for j = 1:N_ens;  Y_p_t[j, :] .-=  g_mean;  end
    Y_p_t ./= sqrt(N_ens - 1)
    
    X = Y_p_t/Σ_ν*Y_p_t'

    svd_X = svd(X)
    P, Γ = svd_X.U, svd_X.S
    
    # compute the mean for EAKI and ETKI at next time step 
    θ_mean = θ_p_mean + Z_p_t' * (P *( (Γ .+ 1.0) .\ (P' * (Y_p_t * (Σ_ν\(eki.y - g_mean))))))
    
    if filter_type == "EKI"
        noise = rand(MvNormal(zeros(N_y), Σ_ν), N_ens) 

        θ = copy(θ_p) 
        for j = 1:N_ens
            θ[j,:] += Z_p_t' * (P *( (Γ .+ 1.0) .\ (P' * (Y_p_t * (Σ_ν\((eki.y - g[j, :] - noise[:, j]))))))) # N_ens x N_θ
        end

        
    elseif filter_type == "EAKI"
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular
        
        F, sqrt_D_p, V =  trunc_svd(Z_p_t') 
        
        # I + Y_p_t/Σ_ν*Y_p_t' = P (Γ + I) P'
        # Y = V' /(I + Y_p_t/Σ_ν*Y_p_t') * V
        Y = V' * P ./ (Γ .+ 1.0)' * P' * V
        
        svd_Y = svd(Y)
        
        U, D = svd_Y.U, svd_Y.S
        
        A = (F .* sqrt_D_p' * U .* sqrt.(D)') * (sqrt_D_p .\ F')
        
        θ = similar(θ_p) 
        for j = 1:N_ens
            θ[j, :] .= θ_mean + A * (θ_p[j, :] - θ_p_mean) # N_ens x N_θ
        end
        
    elseif filter_type == "ETKI"
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular
        # X = Y_p_t/Σ_ν*Y_p_t'
        # svd_X = svd(X)
        
        # P, Γ = svd_X.U, svd_X.S
        
        #Original ETKF is  T = P * (Γ .+ 1)^{-1/2}, but it is biased
        T = P ./ sqrt.(Γ .+ 1)' * P'
        
        # Z_p'
        θ = similar(θ_p) 
        for j = 1:N_ens;  θ[j, :] .=  θ_p[j, :] - θ_p_mean;  end
        # Z' = （Z_p * T)' = T' * Z_p
        θ .= T' * θ 
        for j = 1:N_ens;  θ[j, :] .+=  θ_mean;  end
        
        
        ################# Debug check
        
        # Z_p_t = copy(θ_p)
        # for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_mean;    end
        # Z_p_t ./= sqrt(N_ens - 1)
        
        
        # θθ_p_cov = construct_cov(eki, θ_p, θ_p_mean, θ_p, θ_p_mean)
        # θθ_cov = Z_p_t'*(I - Y_p_t/(Y_p_t'*Y_p_t + Σ_ν)*Y_p_t') *Z_p_t
        # θ_mean_debug = dropdims(mean(θ, dims=1), dims=1)
        # θθ_cov_debug = construct_cov(eki, θ, θ_mean_debug, θ, θ_mean_debug)
        # @info "mean error is ", norm(θ_mean - θ_mean_debug), " cov error is ", norm(θθ_cov - Z_p_t'*T*T'*Z_p_t), norm(θθ_cov - θθ_cov_debug)
        
        
    else
        error("Filter type :", filter_type, " has not implemented yet!")
    end

    # Save results
    push!(eki.θ, θ) # N_ens x N_θ
    push!(eki.y_pred, g_mean)
end

# the evolution error covariance is (Δt/(1-Δt) + 1 - α^2) * C_n or  (Δt/(1-Δt) + 1 - α^2) * C_0
# the observation error covariance is Σ_ν = (1/Δt) * Σ_η
function EKI_Run(s_param, forward::Function, 
    filter_type,
    θ0_mean, θθ0_cov_sqrt,
    prior_mean, prior_cov_sqrt,
    N_ens,
    y, Σ_η,
    Δt,
    α_reg,
    update_freq,
    N_iter)
    

    θ_names = s_param.θ_names
    

    ekiobj = EKIObj(filter_type ,
    θ_names,
    N_ens,
    θ0_mean, θθ0_cov_sqrt,
    prior_mean, prior_cov_sqrt,
    y, Σ_η,
    Δt,
    α_reg, update_freq)
    
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward) 
    
    
    for i in 1:N_iter
        update_ensemble!(ekiobj, ens_func) 
    end
    
    return ekiobj
    
end