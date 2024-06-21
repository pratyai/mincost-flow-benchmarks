 (csvsql --query
  "select labels, solver, count(1) as count,
    min(iters), max(iters),
    min(sddm_calls_per_iter), max(sddm_calls_per_iter),
    min(time_s_per_arc_per_iter), max(time_s_per_arc_per_iter),
    min(fact_s_per_arc_per_iter), max(fact_s_per_arc_per_iter)
    from giant
    where iters < 200
    and solver in ('tulip_approxchol', 'tulip_cmg', 'tulip_hypre')
    and labels is not null
    group by labels, solver"
  ../giant.csv | csvlens)
