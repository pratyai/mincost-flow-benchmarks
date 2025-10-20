using Downloads
using GZip
using Printf
using SparseArrays # Required for Dimacs.jl, though not directly used for download
using Dimacs # To potentially use Dimacs.WriteDimacs if needed, but not for this task

# --- Configuration ---
base_url = "http://lime.cs.elte.hu/~kpeter/data/mcf/gridgraph/grid_wide_"
problems_dir = "data/problems/lemon_gridwide"
specs_dir = "data/specs"
spec_filename = joinpath(specs_dir, "lemon_gridwide.inspec")

# Ensure output directories exist
mkpath(problems_dir)
mkpath(specs_dir)

# Prepare spec file content
spec_lines = ["name,input_file,bytes"]

println("Downloading lemon_gridwide problems...")

# Loop for XX from 08 to 20
for xx_val = 8:20
    xx_str = @sprintf("%02d", xx_val) # Format as 08, 09, 10, etc.
    # Loop for Y from 'a' to 'e'
    for y_char_code = Int('a'):Int('e')
        y_char = Char(y_char_code)

        problem_name = "grid_wide_$(xx_str)$(y_char)"
        filename_gz = "$(problem_name).min.gz"
        output_file_path_gz = joinpath(problems_dir, filename_gz)
        download_url = "$(base_url)$(xx_str)$(y_char).min.gz"

        println("  Downloading $(download_url) to $(output_file_path_gz)...")

        try
            Downloads.download(download_url, output_file_path_gz)

            # Get file size
            file_bytes = filesize(output_file_path_gz)
            push!(
                spec_lines,
                "$(problem_name),data/problems/lemon_gridwide/$(filename_gz),$(file_bytes)",
            )
        catch e
            println("    Error downloading $(download_url): $(e)")
        end
    end
end

# Write the spec file
open(spec_filename, "w") do io
    for line in spec_lines
        println(io, line)
    end
end

println("Generated lemon_gridwide problems and spec file: $(spec_filename)")
