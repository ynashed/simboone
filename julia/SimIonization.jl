module SimIonization

# Known parameters for genearting ground truth output
g_density           = 1.38      # g/cm^3
g_alpha             = 0.847
g_beta              = 0.2061
g_efield            = 0.500     # V/sm
g_lifetime          = 0.6       # ms
g_energy_threshold  = 0.06      # MeV threshold to ignore drift
g_dedx_threshold    = 0.0001    # MeV/cm threshold to ignore ...
g_vdrift            = 0.153812  # cm/us

function forward_model(E, x, de_dx)
    # apply recombination
    x_0 = E .* log.(g_alpha .+ g_beta .* de_dx) ./ (g_beta .* de_dx)
    # apply lifetime
    # CL: scale lifetime back to the correct value range
    x_1 = x .* exp.( -1. .* x ./ g_vdrift / (g_lifetime * 10000))

    return x_0, x_1
end

function simulate(N=1000, C=3)
    X = rand(N, C)
    return forward_model(X[:,[1]], X[:,[2]], X[:,[3]])
end

end #end of module
