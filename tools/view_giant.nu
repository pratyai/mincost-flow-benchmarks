(csvsql --query
  "select labels, solver, count(*),
    min(iters), max(iters),
    min(sddm_calls_per_iter), max(sddm_calls_per_iter),
    min(time_s_per_arc_per_iter), max(time_s_per_arc_per_iter),
    min(fact_s_per_arc_per_iter), max(fact_s_per_arc_per_iter),
    min(solv_s_per_arc_per_sddm_call), max(solv_s_per_arc_per_sddm_call)
    from giant group by labels, solver"
  ../giant.csv | csvlens)
