using Random
using Dimacs
using SparseArrays # Required by Dimacs.jl
using GZip # For decompressing .min.gz to .min

# Include the GridGraphs module
include("../src/graph_generators/GridGraphs.jl")
using .GridGraphs

# --- Configuration ---
h_fixed = 5
w_values = round.(Int, 10 .^ (range(log10(5), stop = log10(1_000_000), length = 20)))
max_cap = 100
max_cost = 10

problems_dir = "data/problems/unifwidegrid"
specs_dir = "data/specs"
spec_filename = joinpath(specs_dir, "unifwidegrid.inspec")

# Ensure output directories exist
mkpath(problems_dir)
mkpath(specs_dir)

# Initialize random number generator for reproducibility
rng = MersenneTwister(1234)

# Prepare spec file content
spec_lines = ["name,input_file,bytes"]

println("Generating grid problems...")

for w in w_values
    problem_name = "grid_h$(h_fixed)_w$(w)"
    output_file_path_gz = joinpath(problems_dir, "$(problem_name).min.gz")
    output_file_path_min = joinpath(problems_dir, "$(problem_name).min")

    println("  Generating $(problem_name)...")
    mcfp_net = GridGraphs.generate_grid_graph_mcfp(h_fixed, w, max_cap, max_cost, rng)

    # Write to DIMACS .min.gz file
    Dimacs.WriteDimacs(output_file_path_gz, mcfp_net)

    # Decompress the .min.gz file to create a plain .min file
    GZip.open(output_file_path_gz) do gz_file
        plain_content = read(gz_file, String)
        open(output_file_path_min, "w") do min_file
            write(min_file, plain_content)
        end
    end

    file_bytes = filesize(output_file_path_gz)
    push!(
        spec_lines,
        "$(problem_name),data/problems/unifwidegrid/$(problem_name).min.gz,$(file_bytes)",
    )
end

# Write the spec file
open(spec_filename, "w") do io
    for line in spec_lines
        println(io, line)
    end
end

println("Generated $(length(w_values)) grid problems and spec file: $(spec_filename)")
