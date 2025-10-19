module SolverCommon

using Tulip
using Dimacs
using SparseArrays
using Random

function create_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}) where {Tv<:Number}
    Random.seed!(1)
    local A = sparse(Int8.(netw.G.IncidenceMatrix))
    local b = Tv.(netw.Demand)
    local u = Tv.(netw.Cap)
    local c = Tv.(netw.Cost)
    local n, m = size(A)

    local lp = Tulip.Model{Tv}()
    local pb = lp.pbdata
    Tulip.load_problem!(
        pb,
        "mcfp",  # some arbitrary name
        true,  # true := minimize
        c,  # objective vector := cost
        zero(Tv),  # no constant term in the cost
        A,  # equality constraint matrix
        b,  # equality constraints := up = down = b
        b,
        zero(u),  # box constraints := lower bound = 0, upper bound = u
        u,
        repeat([""], n),  # leave empty strings as variable and constraint names
        repeat([""], m),
    )
    return lp
end

end # module SolverCommon
