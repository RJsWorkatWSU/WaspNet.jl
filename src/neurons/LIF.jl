"""
    LIF{T<:Number}<:AbstractNeuron 

Contains the necessary parameters for describing a Leaky Integrate-and-Fire (LIF) neuron as well as the current membrane potential of the neuron.

# Fields
- `τ::T`: Neuron time constant (ms)
- `R::T`: Neuronal model resistor (MOhms)
- `θ::T`: Threshold voltage (mV) - when state exceeds this, firing occurs.
- `vSS::T`: Steady-state voltage (mV) - in the absence of input, this is the resting membrane potential.
- `v0::T`: Reset voltage (mV) - immediately after firing, state is set to this.
- `state::T`: Current membrane potential (mV)

Different relative orders of threshold voltage, resting voltage, and reset voltage will produce different dynamics.
The default values of resting > threshold >> reset allows for a baseline firing rate that can be modulated up or down.
@with_kw struct LIF{T<:Number}<:AbstractNeuron 
    τ::T = 8.         
    R::T = 10.      
    θ::T = -55.     
    vSS::T = -50.
    v0::T = -100. 
    state::T = -100.
    output::T = 0.     
end
"""

struct LIF<:AbstractNeuron     
    τ::Float64 # 8.    
    R::Float64 # 10.
    θ::Float64 # -55.
    vSS::Float64    
    v0::Float64#  -100.
    state::Float64#  -100. # state is just the instantaneous membrane potential.
    output::Int64 # 0. # output is spike or no spike.
end
# define default LIF.
LIF() = LIF(8. ,10.,-55,-50,-100.,-100.,0.)


"""
    update!(neuron::LIF, input_update, dt, t)

Evolve an `LIF` neuron subject to a membrane potential step of size `input_update` a time duration `dt` starting from time `t`

# Inputs
- `input_update`: Membrane input charge (pC)
- `dt`: timestep duration (s)
- `t`: global time (s)
"""
function update(neuron::LIF, input_update, dt, t)
    output = 0.
    # If an impulse came in, add it
    state = neuron.state + input_update * neuron.R / neuron.τ

    # Euler method update
    state += 1000 * (dt/neuron.τ) * (-state + neuron.vSS)

    # Check for thresholding
    if state >= neuron.θ
        state = neuron.v0
        output = 1. # Binary output
    end

    return (output, LIF(neuron.τ, neuron.R, neuron.θ, neuron.vSS, neuron.v0, state, output))
end

function reset(neuron::LIF)
    return LIF(neuron.τ, neuron.R, neuron.θ, neuron.vSS, neuron.v0, neuron.v0, 0.)
end
# similar to cellModel-GLIF1.jl, but more performant as it uses a single
# membrane time constant and no resting membrane potential.

##
# From Ben Arthurs awesome code.
# https://github.com/SpikingNetwork/TrainSpikingNet.jl/blob/master/src/cellModel-LIF.jl
##

function cellModel_timestep!(i::Number, v, X, u, args)
    v[i] += args.dt * args.invtau_mem[i] * (X[i] + u[i] - v[i])
end
function cellModel_timestep!(bnotrefrac::AbstractVector, v, X, u, args)
    @. v += bnotrefrac * args.dt * args.invtau_mem * (X + u - v)
end

function cellModel_spiked(i::Number, v, args)
    v[i] > args.thresh[i]
end
function cellModel_spiked!(bspike::AbstractVector, bnotrefrac::AbstractVector, v, args)
    @. bspike = bnotrefrac & (v > args.thresh)
end

function cellModel_reset!(i::Number, v, args)
    v[i] = args.vre
end
function cellModel_reset!(bspike::AbstractVector, v, args)
    @. v = ifelse(bspike, args.vre, v)
end