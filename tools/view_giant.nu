 (csvsql --query
  "select probclass, solver, count(1) as count,
    cast(min(iters) as int) as min_iters, cast(max(iters) as int) as max_iters,
    round(min(sddm_calls_per_iter), 1) as min_sddm, round(max(sddm_calls_per_iter), 1) as max_sddm,
    round(min(time_s_per_arc_per_iter) * 1e6, 1) as min_time, round(max(time_s_per_arc_per_iter) * 1e6, 1) as max_time,
    round(min(fact_s_per_arc_per_iter) * 1e6, 1) as min_fact, round(max(fact_s_per_arc_per_iter * 1e6), 1) as max_fact
    from giant
    where iters < 200
    and solver in ('tulip_approxchol', 'tulip_cmg', 'tulip_hypre')
    and probclass is not null
    group by probclass, solver"
  ../giant.csv | csvlens)
