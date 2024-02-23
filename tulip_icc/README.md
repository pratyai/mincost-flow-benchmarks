# Tulip + Combinatorial Multigrid Benchmarking

The exact flow solution recovery is not covered here. If verifying or exact solution recovery is needed, have the tool write the solutions into a directory (`-s` flag), and then verify that solution against the input problem and the true solution.

## How to run

```sh
julia --project=. src/main.jl -i <YOUR INPUT SPEC> [-o <YOUR OUTPUT SPEC>] [-s <YOUR SOLUTION DIRECTORY>] [-t warmup/warmup.inspec]
```
