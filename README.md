# A flexible structure for creating benchmarks for mincost-flow solvers.

## Overview

Weexpect to compare the performances of solver implementations that can have very different methods and relevant metrics. More importantly, these implementations can be from very different languages too, which can make it very difficult to integrate with a unified test harness.

So, we want to have a framework of performance measurement that can interface over simple files. Then we can write simple programs in the most convenient way (e.g. usually in the native language of the implementation that we are measuring), which can read a structured input and produce a structured output containing the various metrics collected during the benchmarking process.

These strucutured input and output files should be human readable, but also easy to work with simple scripts or programs --- we will assume CSV format here. But if necessary, they can contain URLs to other files that may have more complicated or unreadable strucuture (e.g. matrix or vector files if we need to inspect them).

We also need the structure of these files to be sufficiently flexible, so that very different kinds of solvers or experiments can report different kinds of metrics which can still be joined, when applicable, to make a comparison.

## Approximate vs. Exact Solution

We will assume that the following statements hold for approximate solvers:

- It will be possible to recover an exact solution from a sufficiently good approximate solution.
- The approximate solution and the exact solution recovery can be cleanly separated into two phases that do not rely on each other.
- We can separately evaluate the approximate solution against the original problem and/or a known true solution.

These allow us to write benchmarks that targets only the different implementations and tunings of the approximate solver. Evalutation of these solutions and exact solution recovery will be done by a separate tool that possibly need to be written only once, and without the performance concerns.

## Input Spec

### Basic

The following columns are required:

- Name (`name`: `string`): A short name identifying the particular problem described in this entry.
- Format (`format`: `string`): A keyword describing the format of the rest of this entry. We will assume [DIMACS](https://lpsolve.sourceforge.net/5.5/DIMACS_mcf.htm) format here.

### DIMACS format

The following columns are required for DIMACS format.

- Input File (`input_file`: `string`)

There can be other optional columns as well that may be important for a particular implementation or algorithm (e.g. maximum number of iterations, tolerance, various tunable parameters etc.).

## Output Spec

### Basic

The following columns are required:

- Name (`name`: `string`): A short name identifying the particular problem described in this entry.

Every benchmarking experiemnts can add additional columns containing scalar metrics or an URL to different files with complicated structures. There is no other real requirement than this. But for making a good comparisons, the different experiments should agree upon the semantics of the column (identified by its header) beforehand.

### Tulip + ApproxSDDM (mostly default implementation)

For example, our Tulip + ApproxSDDM solver, which is what we are mostly intereted in for the moment, should also produce the following columns:

- Status (`status`: `string`): The solution status string from Tulip.
- Solution Time in Seconds (`time_s`: `float`): The solution time reported by Tulip.
- Number of Iterations (`iters`: `int`): The number of iterations reported by Tulip.
- [Optional] Solution file (`solution_file`: `string`): If we requested for the solution to be stored in a file, the path of that file.
