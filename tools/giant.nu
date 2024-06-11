exa ../*.inspec ../*/*.outspec | julia --project=. produce_giant_combined_spec.jl | save -f ../giant.csv;
