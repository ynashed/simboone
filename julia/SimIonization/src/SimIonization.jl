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
g_lifetime          = 0.6       # ms
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
    # apply recombination
    x_0 = input.E .* log.(g_alpha .+ g_beta .* input.de_dx) ./ (g_beta .* input.de_dx)
    # apply lifetime
    x_1 = input.x .* exp.( -1. .* input.x ./ g_vdrift / (g_lifetime * 10000))

    return [x_0 x_1]
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
    input.de_dx = clamp.(input.de_dx, eps(Float64), typemax(Float64))
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
            default = 0
        "--num_step"
            help = "Number of optimization steps"
            arg_type = Int
            default = 20000
        "--lr"
            help = "Step size/Learning rate for optimization"
            arg_type = Real
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
    vox = fid["voxels"][parsed_args["sample_id"]]
    @info(vox)
    # KH: 100 micron per index, so divide by 100 to turn index=>cm
    # CL: somehow optimization works much better if divided by 100000 (so x coordinates are normalized to be the same value range as energy)
    gt = forward_model(Variables(vox[:, 4], vox[:, 1].*1e-5, vox[:, 5].*1e-2))

    (vars, losses) = fit(gt, iterations=parsed_args["num_step"], η=parsed_args["lr"])

    println("Energy l2 error: $(norm(vox[:, 4] .- vars.E, 2))")
    println("Position(x) l2 error: $(norm(vox[:, 1] .- vars.x, 2))")
    println("de_dx l2 error: $(norm(vox[:, 1] .- vars.x, 2))")

    close(fid)
end

main()

end #end of module
