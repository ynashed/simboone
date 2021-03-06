import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
Pkg.instantiate()

module SimIonization

@info("Loading Zygote...")
using Zygote, LinearAlgebra, HDF5

# Known parameters for genearting ground truth output
g_density           = 1.38      # g/cm^3
g_alpha             = 0.847
g_beta              = 0.2061
g_efield            = 0.500     # V/sm
g_lifetime          = 6000      # ms
g_energy_threshold  = 0.06      # MeV threshold to ignore drift
g_dedx_threshold    = 0.0001    # MeV/cm threshold to ignore ...
g_vdrift            = 0.153812  # cm/us

mutable struct Variables
    # These values will be implicitly learned
    E::Matrix
    x::Matrix
    de_dx::Matrix
end
Variables(N) = Variables(rand(N, 1), rand(N, 1), rand(N, 1))

function forward_model(input::Variables)
    de_dx_scaled = g_beta .* input.de_dx
    # apply recombination
    x_0 = input.E .* log.(g_alpha .+ de_dx_scaled) ./ de_dx_scaled
    # apply lifetime
    x_0 = x_0 .* exp.( -1. .* input.x ./ g_vdrift / g_lifetime)

    return x_0
end

function loss(input::Variables, expected_output)
    return norm(forward_model(input) .- expected_output, 2)
end

# Let's define an update rule that will allow us to modify the weights
# of our model a tad bit according to the gradients
function sgd_update!(input::Variables, grads, η)
    input.E .+= η .* -grads.E
    input.x .+= η .* -grads.x
    input.de_dx .+= η .* -grads.de_dx
    clamp!(input.de_dx, eps(Float64), typemax(Float64))
end

function fit(GT; iterations=100, η=0.001)
    @info("Initializing variables...")
    vars = Variables(size(GT, 1))

    @info("Running train loop for $(iterations) iterations")
    losses = zeros(iterations)
    for idx in 1:iterations
        grads = Zygote.gradient(m -> loss(m, GT), vars)[1][]
        losses[idx] = loss(vars, GT)
        sgd_update!(vars, grads, η)
        @info("Loss at $(idx) iterations is $(losses[idx])")
    end

    return vars, losses
end

using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--data_file"
            arg_type = String
            required = true
            help = "Path to .h5 data file"
        "--sample_id"
            help = "Select a detection sample from the data file to test"
            arg_type = Int
            default = 1
        "--num_step"
            help = "Number of optimization steps"
            arg_type = Int
            default = 20000
        "--lr"
            help = "Step size/Learning rate for optimization"
            arg_type = Float64
            default = 1e-4
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    fid = h5open(parsed_args["data_file"], "r")
    vox_attr = read(fid["vox_attr"])
    vox = read(fid["voxels"])[parsed_args["sample_id"]]
    vox = transpose(reshape(vox, size(vox_attr, 2), :))

    E = vox[:, [4]]; x = vox[:, [1]]; de_dx = vox[:, [5]];
    clamp!(de_dx, 0.8, 300) #KT said that's a good range
    gt = forward_model(Variables(E, x.*1e-4, de_dx.*1e-2)) #scale x and de_dx

    (vars, losses) = fit(gt, iterations=parsed_args["num_step"], η=parsed_args["lr"])

    println("Energy l2 error: $(norm(E .- vars.E, 2))")
    println("Position(x) l2 error: $(norm(x .- vars.x, 2))")
    println("de_dx l2 error: $(norm(de_dx .- vars.de_dx, 2))")

    close(fid)
end

main()

end #end of module
