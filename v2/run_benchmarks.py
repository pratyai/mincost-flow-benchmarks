import argparse
import csv
import os
import sqlite3
import sys
import time
import tomllib
import numpy as np
import scipy.sparse as sp
import ecos
from read_dimacs import read_dimacs

def get_or_create_config_id(cursor, config_path):
    with open(config_path, "r") as f:
        config_text = f.read()
    
    cursor.execute("SELECT id FROM configs WHERE config_text = ?", (config_text,))
    row = cursor.fetchone()
    if row:
        return row[0], tomllib.loads(config_text)
    
    data = tomllib.loads(config_text)
    solver_name = data.get("solver", "unknown")
    
    # Check if this script supports the config
    # Python runner only handles python-specific solver names
    if solver_name != "ecos-py":
        return None, None

    # Insert config
    cursor.execute("""
        INSERT INTO configs (config_text, solver_name, precision)
        VALUES (?, ?, ?)
    """, (config_text, solver_name, data.get("precision", "Float64")))
    return cursor.lastrowid, data

def get_or_create_problem_id(cursor, name, input_file):
    cursor.execute("SELECT id FROM problems WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    
    real_path = input_file
    
    if os.path.exists(real_path):
        # We need to read it to get stats if missing, but usually Julia does this.
        # I'll just skip stats populate for now to save time, or do it?
        # Let's read it, read_dimacs is fast enough?
        # For large files, reading twice (once here, once for solve) is bad.
        # I'll default to NULLs if new problem.
        file_size = os.path.getsize(real_path)
    else:
        file_size = None

    cursor.execute("""
        INSERT INTO problems (name, input_file, bytes)
        VALUES (?, ?, ?)
    """, (name, input_file, file_size))
    return cursor.lastrowid

def solve_ecos(net, config):
    c = net.cost
    bounds = list(zip(np.zeros_like(net.cap), net.cap))
    A = net.G.incidence_matrix
    b = net.demand
    
    # Setup ECOS matrices
    num_vars = len(c)
    I = sp.eye(num_vars, format="csc")
    G = sp.vstack([-I, I], format="csc").astype(float)
    h_ecos = np.concatenate(
        [[-bnd[0] for bnd in bounds], [bnd[1] for bnd in bounds]]
    ).astype(float)

    A_ecos = sp.csc_matrix(A).astype(float)
    b_ecos = b.astype(float)
    
    verbose = config.get("verbose", False) or config.get("parameters", {}).get("verbose", False)
    
    # Extract ECOS specific params
    kwargs = {}
    params = config.get("parameters", {})
    for k in ["feastol", "abstol", "reltol", "maxit"]:
        if k in params:
            kwargs[k] = float(params[k]) if k != "maxit" else int(params[k])

    t_start = time.time()
    try:
        sol = ecos.solve(
            c, G, h_ecos, {"l": 2 * num_vars}, A_ecos, b_ecos, verbose=verbose, **kwargs
        )
        solve_time = time.time() - t_start
        
        flag = sol["info"]["exitFlag"]
        success = flag in [0, 10]
        
        status_map = {
            0: "Optimal",
            10: "Optimal", # Inaccurate
            1: "PrimalInfeasible",
            2: "DualInfeasible",
            -1: "MaxIter",
            -2: "NumericalError"
        }
        status = status_map.get(flag, f"Unknown({flag})")
        
        iters = sol["info"]["iter"]
        objective_value = sol["info"]["pcost"] if success else None
        
        return {
            "status": status,
            "time_s": solve_time,
            "iters": iters,
            "objective_value": objective_value,
            "solution": sol.get("x")
        }
    except Exception as e:
        print(f"ECOS Exception: {e}")
        return {
            "status": "Error",
            "time_s": time.time() - t_start,
            "iters": 0,
            "objective_value": None,
            "solution": None
        }

def setup_database_schema(db):
    cursor = db.cursor()
    cursor.execute("PRAGMA busy_timeout = 30000;")
    cursor.execute("PRAGMA journal_mode = WAL;")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS problems (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        input_file TEXT,
        bytes INTEGER,
        num_vertices INTEGER,
        num_edges INTEGER,
        true_optimal REAL,
        lemon_time_s REAL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS configs (
        id INTEGER PRIMARY KEY,
        config_text TEXT UNIQUE,
        solver_name TEXT,
        precision TEXT,
        ipm_preg_min REAL,
        ipm_dreg_min REAL,
        ipm_iterations_limit INTEGER,
        pcg_maxits INTEGER,
        pcg_tol REAL,
        approxchol_type TEXT,
        approxchol_stag_test INTEGER,
        approxchol_split INTEGER,
        approxchol_merge INTEGER,
        cholmod_nested_dissection BOOLEAN
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY,
        name TEXT,
        status TEXT,
        solver_name TEXT,
        config_id INTEGER,
        problem_id INTEGER,
        time_s REAL,
        iters INTEGER,
        solution_file TEXT,
        fact_s REAL,
        solv_s REAL,
        sddm_calls INTEGER,
        optimal_value REAL,
        FOREIGN KEY (config_id) REFERENCES configs(id),
        FOREIGN KEY (problem_id) REFERENCES problems(id)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS solver_history (
        id INTEGER PRIMARY KEY,
        run_id INTEGER,
        ipm_iter INTEGER,
        solve_in_iter INTEGER,
        relative_residual_norm REAL,
        absolute_residual_norm REAL,
        pcg_iterations INTEGER,
        FOREIGN KEY (run_id) REFERENCES runs(id)
    )
    """)
    db.commit()

def run_benchmarks(args):
    print(f"Opening database: {args.output_db}")
    db = sqlite3.connect(args.output_db)
    setup_database_schema(db)
    cursor = db.cursor()
    
    # Process configs
    valid_configs = []
    for cf in args.config_files:
        cid, cdata = get_or_create_config_id(cursor, cf)
        if cid:
            valid_configs.append((cid, cdata, cf))
        else:
            print(f"Skipping non-ecos config: {cf}")
    
    if not valid_configs:
        print("No valid configs for ECOS found.")
        return

    # Process specs
    for spec_file in args.input_spec_files:
        print(f"Processing spec: {spec_file}")
        with open(spec_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                input_file = row['input_file']
                
                if not os.path.exists(input_file):
                    # Check if relative to project root (assuming script run from root)
                    if not os.path.exists(input_file):
                        print(f"Warning: {input_file} not found.")
                        continue
                
                pid = get_or_create_problem_id(cursor, name, input_file)
                
                net = None # Lazy load
                
                for cid, config, cfile in valid_configs:
                    # Check if run exists
                    cursor.execute(
                        "SELECT id FROM runs WHERE config_id=? AND problem_id=? AND solver_name='py_ecos'", 
                        (cid, pid)
                    )
                    if cursor.fetchone():
                        print(f"Skipping {name} with {cfile} (already exists)")
                        continue
                    
                    if net is None:
                        print(f"Loading {name}...")
                        net = read_dimacs(input_file)
                    
                    print(f"Running {name} with {cfile} (Python ECOS)...")
                    res = solve_ecos(net, config)
                    print(f"  Result: {res['status']}, Time: {res['time_s']:.4f}s")
                    
                    # Insert run
                    cursor.execute("""
                        INSERT INTO runs 
                        (name, status, solver_name, config_id, problem_id, time_s, iters, optimal_value, solution_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        name, res["status"], "py_ecos", cid, pid, res["time_s"], res["iters"], res["objective_value"], "null"
                    ))
                    db.commit()
                    
    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python ECOS Benchmark Runner")
    parser.add_argument("-i", "--input_spec_files", action="append", required=True)
    parser.add_argument("-c", "--config_files", action="append", required=True)
    parser.add_argument("-o", "--output_db", required=True)
    
    args = parser.parse_args()
    run_benchmarks(args)
