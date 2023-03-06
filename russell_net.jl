"""
# Example
Here we step through an example to demonstrate typical usage of `WaspNet`. We begin by building a new `LIF Neuron` type which encapsulates the update rules for a neuron. Then, we make two `Layer`s of neurons to communicate back and forth; one `Layer` is strictly feed-forward, the other is a recurrent layer accepting inputs both from itself and from the preceding `Layer`. Finally, we build a `Network` out of these `Layer`s and simulate it to observe the evolution of the `Network` as a whole.
## Getting Started
The following code has been tested in Julia 1.4.2 and executes without errors. Any dependencies will be called out as necessary. 

To start, the only necessary dependency will be `WaspNet` itself, so we start by importing it. Other useful packages will be `Random` and `BlockArrays`
"""

using WaspNet
using OhMyREPL
using Random
using BlockArrays
using SparseArrays
include("genPotjansConnectivity.jl")
"""
## Constructing a New Neuron
The simlpest unit in `WaspNet` is the `Neuron` which translates an input signal from preceding neurons into the evolution of an internal state and ultimately spikes which are sent to neurons down the line.

A concrete `AbstractNeuron` needs to cover 3 things:
 - A new `struct` which is a subtype of `AbstractNeuron`; optionally mutable
 - An `update` method to implement the dynamics of the neuron
 - A `reset` method which restores the neuron to its default state.

We will implement the [Leaky Integrate-&-Fire](https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire) neuron model here, but a slightly different implementation is available in `WaspNet/src/neurons/lif.jl` or with `WaspNet.LIF`. 

A concrete `AbstractNeuron` implementation currently must include two specific fields: `state` and `output`. `state` holds the current state of the neuron in a `Vector` and `output` holds the output of the neuron after its last update; for a spiking neuron, update is either a `0` or a `1` to denote whether a spike did or did not occur. Additional fields should be implemented as needed to parameterize the neuron. 
#Additionally, we need to define how to evolve our neuron given a time step. This is done by adding a method to `WaspNet.update!` or `WaspNet.update`, a function which is global across all `WaspnetElements`. To `update` a neuron, we provide the `neuron` we need to update, the `input_update` to the neuron, the time duration to evolve `dt`, and the current global time `t`. In the LIF case, the `input_update` is a voltage which must be added to the membrane potential of the neuron resulting from spikes in neurons which feed into the current neuron. `reset` simply restores the state of the neuron to its some state.

#We use an [Euler update](https://en.wikipedia.org/wiki/Euler_method) for the time evolution because of its simplicity of implementation.

#Note that both `update` and `reset` are defined *within* `WaspNet`; that is, we actually define the methods `WaspNet.update` and `WaspNet.reset`. If defined externally, these methods are not visible to other methods from within `WaspNet`.
"""

#Now we want to instantiate our `LIF` neuron, update it a few times to see the state of the neuron change

#neuronLIF = WaspNet.LIF(8., 10.E2, 30., 40., -55., -55., 0.)

#println(neuronLIF.state)
## -55.0
#(output, neuronLIF) = update(neuronLIF, 0., 0.001, 0.)
#println(neuronLIF.state)
# -49.993125

#neuronLIF = reset!(neuronLIF)
#println(neuronLIF.state)


N1 = [WaspNet.LIF() for _ in 1:5]
W10 = reshape(collect(1:20)*1., (5,4))
W11 = reshape(collect(1:25)*1., (5,5))
W12 = reshape(collect(1:30)*1., (5,6))
W1 = BlockArray(hcat(W10, W11, W12), [5],[4,5,6])
L1 = Layer(N1, W1, [0, 1, 2])

N2 = [WaspNet.LIF() for _ in 1:6]
W20 = reshape(collect(1:24)*1., (6,4))
W21 = reshape(collect(1:30)*1., (6,5))
W22 = reshape(collect(1:36)*1., (6,6))
W2 = BlockArray(hcat(W20, W21, W22), [6], [4,5,6])
L2 = Layer(N2, W2, [0,1,2])

net = Network([L1, L2], 4)
prune_layers = [1,2]
prune_neurons = [[3], [2, 4]]
pruned = WaspNet.prune(net, prune_layers, prune_neurons)
#@show(W10)


function get_Ncell(scale=1.0::Float64)
	ccu = Dict{String, Int32}("23E"=>20683,
		    "4E"=>21915, 
		    "5E"=>4850, 
		    "6E"=>14395, 
		    "6I"=>2948, 
		    "23I"=>5834,
		    "5I"=>1065,
		    "4I"=>5479)
	ccu = Dict{String, Int32}((k,ceil(Int64,v*scale)) for (k,v) in pairs(ccu))
	Ncells = Int32(sum([i for i in values(ccu)])+1)
	Ne = Int32(sum([ccu["23E"],ccu["4E"],ccu["5E"],ccu["6E"]]))
    Ni = Int32(Ncells - Ne)
    Ncells, Ne, Ni, ccu

end

function potjans_layer()
    scale =1.0/30.0
    Ncells,Ne,Ni, ccu = get_Ncell(scale)    
    pree = prie = prei = prii = 0.1
    K = round(Int, Ne*pree)
    sqrtK = sqrt(K)
    pree = prie = prei = prii = 0.1
    g = 1.0
    tau_meme = 10   # (ms)
    je = 2.0 / sqrtK * tau_meme * g
    ji = 2.0 / sqrtK * tau_meme * g 
    jee = 0.15je 
    jei = je 
    jie = -0.75ji 
    jii = -ji
    genStaticWeights_args = (;Ncells,jee,jie,jei,jii,ccu,scale)
    _, w0Weights, _ = genStaticWeights(genStaticWeights_args)
    w0Weights
end
w0Weights = sparse(potjans_layer())

#@show(w0Weights)
# -55.0

#We can also `simulate!` a neuron, chaining together multiple `update` calls and returning the outputs (spikes) and optionally the internal state of the neuron as well. The following code simulates our `LIF` neuron for 250 ms with a 0.1 ms time step. The input to the neuron is a function of one parameter, `t`, defined by `(t) -> 0.4*exp(-4t)`.
#assigned_time = t = 0.4.*exp(-4t)
#@show(assigned_time)
#LIFsim = simulate!(neuronLIF,assigned_time, 0.0001, 0.250, track_state=true)

#`LIFsim` is a `SimulationResult` instance with three fields: `LIFsim.outputs`, `LIFsim.states`, and `LIFsim.times`. `times` holds all of the times at which the neuron was simualted, `outputs` holds the output of the neuron at each time step, and `states` hold the state of the neuron at each time step.
## Combining Neurons into a Layer
#In `WaspNet`, a collection of neurons is called a `Layer` or a population. A `Layer` is homogeneous insofar as all of the `Neuron`s in a given `Layer` must be of the same type, although their individual parameters may differ. The chief utility of a `Layer` is to handle the computation of the inputs into its constituent `Neuron`s; which is handled through a multiplication of the input spike vector by a corresponding weight matrix, `W`.

#The following code constructs a feed-forward `Layer` with `N` `LIF` neurons inside of it with an incoming weight matrix `W` to handle 2 inputs. 
@show(w0Weights[:,1])
N = length(w0Weights);
PotLayer = zeros(N,N)
for i in 1:N
    for j in 1:N
        PotLayer[i,j] = WaspNet.LIF(8., 10.E2, 30., 40., -55., -55., 0.)
    end
end
neurons = [WaspNet.LIF(8., 10.E2, 30., 40., -55., -55., 0.) for _ in 1:N];
#weights = randn(MersenneTwister(13371485), N,2);
layer = Layer(neurons, w0Weights[:,1]);

#We can also `update!` a `Layer` by driving it with some input as we did for our `LIF` neuron above. Not that `input` here is actually an `Array{Array{<:Number, 1}, 1}` and not just `Array{<:Number, 1}`. The purpose of this is to handle recurrent or non-feed-forward connections; we will discuss this more in [Constructing Networks from Layers](@ref).

reset!(layer)
update!(layer, [[0.5, 0.8]], 0.001, 0)
println(WaspNet.get_neuron_states(layer))
# [-49.541978928637135, -49.60578871324857, ..., -50.84036022383181]

#And we can `simulate!` with the same syntax as before

layersim = simulate!(layer, (t) -> [randn(2)], 0.001, 0.25, track_state=true);

#Now `layersim.outputs` and `layersim.states` will be of size `NxT` where there are `N` neurons in the `Layer` and `T` time steps.

#There are several `Layer` constructors and `update!` methods; for more information, see [Reference](@ref) or type `?Layer` or `?update!` in the REPL.
## Constructing Networks from Layers
#Once we have `Layer`s available, we need a way to communicate spikes between them. The `Network` solves exactly that problem: it orchestrates communication of spiking (or output signals in general) between `Layer`s, routing the appropriate outputs between `Layer`s. 

#We'll start by constructing a new first `Layer` for our `Network` similar to how we did before with the added parameter of `Nin`, the number of inputs we're feeding into the first `Layer` of the `Network`.

Nin = 2
N1 = 3
neurons1 = [WaspNet.LIF(8., 10.E2, 30., 40., -55., -55., 0.) for _ in 1:N1]
weights1 = randn(MersenneTwister(13371485), N1, Nin)
layer1 = Layer(neurons1, weights1);

#Now we'll make our second `Layer`. This `Layer` is special: it will take feed-forward inputs from the first `Layer`, but also a recurrent connection to itself. This means that we need to specify `W` slightly differently, and we also need to supply a new field, `conns`. To handle non-feed-forward connections in a `K`-layer, `W` must be declared as a `1x(K+1)` `BlockArray`. 
#For our case, `K=2`. Thus, the first block in `W` holds the input weights corresponding to the `Network` input, the second block holds the weights for the first `Layer`, and the third block holds the weights for the second `Layer` feeding back into itself. Similarly we must supply `conns`, an array stating which `Layer`s the current `Layer` connects to. Entries in `conns` are indexed such that `0` corresponds to the `Network` input, `1` corresponds to the output of the first `Layer` and so on. 
N2 = 4;
neurons2 = [WaspNet.LIF(8., 10.E2, 30., 40., -55., -55., 0.) for _ in 1:N2]

W12 = randn(N2, N1) # connections from layer 1
W22 = 5*randn(N2, N2) # Recurrent connections
weights2 = [W12, W22]

conns = [1, 2]

layer2 = Layer(neurons2, weights2, conns);
#To form a `Network`, we specify the constituent `Layer`s
mynet = Network([layer1, layer2], Nin)
#`update!`, `reset!` work just as they did for `Layer`s and `Neuron`s
reset!(mynet)
update!(mynet, 0.4*ones(Nin), 0.001, 0)
println(WaspNet.get_neuron_states(mynet))
# [-49.61444931320574, -50.42051080817629, ..., -49.993125]
#As does `simulate!`
reset!(mynet)
netsim = simulate!(mynet, (t) -> 0.4*ones(Nin), 0.001, 1, track_state=true)
#@show(W10)
#@show(neurons1)
#@show(netsim)