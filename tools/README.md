## How to prduce input spec for a subset [LEMON](https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData) problems?

Some examples:

```sh
julia --project=. produce_lemon_netgen_input_spec.jl <LIST OF PROBLEM NAMES / PATHS> [-p <YOUR DATA DIRECTORY FOR DOWNLOADING>] [-o <SPEC FILE TO WRITE IN>]
```

```sh
bat lemon_problems.txt | grep a | head -n <SOME NUMBER OF ENTRIES> | xargs julia --project=. produce_lemon_netgen_input_spec.jl -p /tmp/data -o ../small.inspec
```
