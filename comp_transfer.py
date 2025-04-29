import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
import seaborn as sns


X_BORDER = 1.0
T_BORDER = 1.0
X_MIN = 0.0
C_VAL = 0.5


def velocity(x, t): return np.full_like(x, C_VAL) if isinstance(x, np.ndarray) else C_VAL
def source_term(x, t): return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
def initial_condition(x): return np.sin(np.pi * np.asarray(x))**3
def left_boundary_condition(t): return np.zeros_like(t) if isinstance(t, np.ndarray) else 0.0
def analytical_solution(x, t, c=C_VAL):
    x = np.asarray(x); u = np.zeros_like(x, dtype=float)
    x_char = x - c * t; condition = (x >= c * t)
    u[condition] = np.sin(np.pi * x_char[condition])**3
    return u
def right_boundary_condition(t, c=C_VAL, x_max=X_BORDER): return analytical_solution(x_max, t, c)


# --- Helper function for grid and parameter initialization ---
def init_grid_and_params(u_initial, x_min, x_max, t_max, Nx, Nt, c_func):
    h = (x_max - x_min) / (Nx - 1)
    tau = t_max / (Nt - 1) if Nt > 1 else t_max
    x_grid = np.linspace(x_min, x_max, Nx)
    t_grid = np.linspace(0, t_max, Nt)
    u = np.zeros((Nt, Nx))
    u[0, :] = u_initial(x_grid)
    c_val = c_func(0, 0)
    r_val = c_val * tau / h
    return h, tau, r_val, x_grid, t_grid, u, c_val

def left_upwind_scheme_vectorized(c_func, f_func, u_initial, u_left_bc, u_right_bc,
                                   x_min, x_max, t_max, Nx, Nt):
    h, tau, r_val, x_grid, t_grid, u, c_val = init_grid_and_params(u_initial, x_min, x_max, t_max, Nx, Nt, c_func)
    if c_val < 0: print("Left Upwind Warning: Unstable for c < 0.")
    for j in range(Nt - 1):
        u[j + 1, 0] = u_left_bc(t_grid[j + 1])
        u[j + 1, 1:] = u[j, 1:] - r_val * (u[j, 1:] - u[j, :-1])
    return u, x_grid, t_grid

def implicit_scheme_b_vectorized_precomp(c_func, f_func, u_initial, u_left_bc, u_right_bc,
                                           x_min, x_max, t_max, Nx, Nt):
    h, tau, r_val, x_grid, t_grid, u, c_val = init_grid_and_params(u_initial, x_min, x_max, t_max, Nx, Nt, c_func)
    denom = 1.0 + r_val
    if np.abs(denom) < 1e-14:
        print(f"Implicit B Warning: Very small denominator {denom:.2e}.")
        u[1:, :] = np.nan; return u, x_grid, t_grid
    for j in range(Nt - 1):
        t_next = t_grid[j+1]; f_next_row = f_func(x_grid, t_next)
        u[j + 1, 0] = u_left_bc(t_next)
        for n in range(1, Nx):
            rhs = u[j, n] + tau * f_next_row[n]
            u[j + 1, n] = (rhs + r_val * u[j + 1, n - 1]) / denom
            if np.isnan(u[j+1, n]):
                if n < Nx - 1: u[j+1, n+1:] = np.nan
                break
    return u, x_grid, t_grid

def lax_friedrichs_scheme_vectorized(c_func, f_func, u_initial, u_left_bc, u_right_bc,
                                      x_min, x_max, t_max, Nx, Nt):
    h, tau, r_val, x_grid, t_grid, u, c_val = init_grid_and_params(u_initial, x_min, x_max, t_max, Nx, Nt, c_func)
    if abs(r_val) > 1.0 + 1e-9: print(f"Lax-Friedrichs Warning: Stability |r|={abs(r_val):.4f} > 1")
    for j in range(Nt - 1):
        u[j + 1, 0] = u_left_bc(t_grid[j + 1])
        u[j + 1, -1] = u_right_bc(t_grid[j + 1])
        u_jp1_interior = (0.5 * (u[j, 2:] + u[j, :-2]) - 0.5 * r_val * (u[j, 2:] - u[j, :-2]))
        u[j + 1, 1:-1] = u_jp1_interior
    return u, x_grid, t_grid

def lax_wendroff_scheme_vectorized(c_func, f_func, u_initial, u_left_bc, u_right_bc,
                                    x_min, x_max, t_max, Nx, Nt):
    h, tau, r_val, x_grid, t_grid, u, c_val = init_grid_and_params(u_initial, x_min, x_max, t_max, Nx, Nt, c_func)
    if abs(r_val) > 1.0 + 1e-9: print(f"Lax-Wendroff Warning: Stability |r|={abs(r_val):.4f} > 1")
    for j in range(Nt - 1):
        u[j + 1, 0] = u_left_bc(t_grid[j + 1])
        u[j + 1, -1] = u_right_bc(t_grid[j + 1])
        term1 = u[j, 1:-1]
        term2 = - 0.5 * r_val * (u[j, 2:] - u[j, :-2])
        term3 = + 0.5 * r_val**2 * (u[j, 2:] - 2 * u[j, 1:-1] + u[j, :-2])
        u_jp1_interior = term1 + term2 + term3
        u[j + 1, 1:-1] = u_jp1_interior
    return u, x_grid, t_grid


def ftcs_scheme_vectorized(c_func, f_func, u_initial, u_left_bc, u_right_bc,
                            x_min, x_max, t_max, Nx, Nt):
    """
    Implements the Explicit Four-Point scheme (Forward Time Centered Space - FTCS).
    Formula: u_m^{n+1} = u_m^n - c*tau/2h * (u_{m+1}^n - u_{m-1}^n)

    WARNING: This scheme is UNCONDITIONALLY UNSTABLE for the linear
             advection equation (du/dt + c du/dx = 0) and will likely diverge rapidly.
             It is included here for illustrative purposes based on the request.
    """
    h, tau, r_val, x_grid, t_grid, u, c_val = init_grid_and_params(u_initial, x_min, x_max, t_max, Nx, Nt, c_func)

    # --- !!! CRITICAL WARNING !!! ---
    print("\nFTCS Scheme Warning: This scheme is UNCONDITIONALLY UNSTABLE for the pure advection equation.")
    print("Expect divergence (NaNs or large numbers) very quickly.")
    # Although theoretically unstable, sometimes for very small tau it might run for a short time.

    for j in range(Nt - 1):
        # Apply boundary conditions for the next time step
        u[j + 1, 0] = u_left_bc(t_grid[j + 1])
        u[j + 1, -1] = u_right_bc(t_grid[j + 1]) # Using analytical/given BC

        # Calculate interior points using the FTCS formula
        # u_m^{n+1} = u_m^n - (r/2) * (u_{m+1}^n - u_{m-1}^n)
        # Note: f_func is ignored here as the simple FTCS doesn't naturally include it
        # without modification, and the source term is zero anyway.
        u_jp1_interior = u[j, 1:-1] - 0.5 * r_val * (u[j, 2:] - u[j, :-2])
                          # + tau * f_func(x_grid[1:-1], t_grid[j]) # Add if source term needed

        u[j + 1, 1:-1] = u_jp1_interior

        # --- Check for immediate divergence ---
        if not np.all(np.isfinite(u[j+1, 1:-1])):
             print(f"FTCS Divergence detected at step j={j}, t={t_grid[j+1]:.4f}")
             # Set remaining steps to NaN to prevent further computation/warnings
             u[j+2:, :] = np.nan
             break # Stop the time loop

    return u, x_grid, t_grid


# --- Convergence Analysis ---

def linear_regr(log_h, a, b):
    return a * log_h + b

# --- UPDATED Convergence Analysis ---
def calculate_convergence(scheme_func, analytical_sol, c_func, f_func,
                         u_initial, u_left_bc, u_right_bc,
                         x_min, x_max, t_max, N_steps=7, Nx_base=41, # Increased N_steps slightly
                         r_stable=0.2, c_const=C_VAL): # <<< Reduced r_stable
    """Calculates and plots convergence order, fixing Courant number, fitting all points."""
    if not hasattr(scheme_func, 'name'):
        # Set a readable name if not present
        scheme_func.name = scheme_func.__name__.replace("_", " ").title()

    print(f"\n--- Convergence Analysis for {scheme_func.name} (Target r={r_stable}) ---")
    errors_max = [] # Max norm (L-infinity)
    errors_l2 = []  # L2 norm
    h_values = []

    Nx_values = np.unique(np.geomspace(Nx_base, Nx_base * 2**(N_steps-1), N_steps, dtype=int))
    print(f"Testing Nx values: {Nx_values}")

    if not callable(analytical_sol): print("Analytical solution function not provided."); return None
    if abs(c_const) < 1e-14: print("Convergence test requires non-zero velocity."); return None

    f_test = f_func
    warnings.filterwarnings('ignore', 'invalid value encountered in log')
    warnings.filterwarnings('ignore', 'divide by zero encountered in log') # Handle log(0) if error is tiny

    for Nx_test in tqdm(Nx_values, desc=f"Converge {scheme_func.name}"):
        h = (x_max - x_min) / (Nx_test - 1)
        if h < 1e-14:
            print(f"Skipping Nx={Nx_test} due to small h.")
            continue

        tau_target = r_stable * h / abs(c_const)
        Nt_test = int(np.ceil(t_max / tau_target)) + 1
        if Nt_test < 2: Nt_test = 2

        actual_tau = t_max / (Nt_test - 1) if Nt_test > 1 else t_max
        actual_r = abs(c_const * actual_tau / h) if abs(h) > 1e-14 else float('inf')
        # print(f" Nx={Nx_test}, h={h:.3e} -> Nt={Nt_test}, tau={actual_tau:.3e}, actual_r={actual_r:.4f}")

        if Nt_test > 1000 * Nx_test: # Increased safeguard limit slightly due to smaller r_stable
             print(f"Warning: Nt={Nt_test} excessively large for Nx={Nx_test}. Skipping.")
             continue

        try:
            solution_num, x_grid, t_grid = scheme_func(
                c_func, f_test, u_initial, u_left_bc, u_right_bc,
                x_min, x_max, t_max, Nx_test, Nt_test
            )

            if solution_num is None or len(t_grid) != Nt_test: print(f"Sim failed/incomplete Nx={Nx_test}. Skip."); continue
            if np.any(np.isnan(solution_num[-1, :])): print(f"NaN detected Nx={Nx_test}. Skip."); continue

            t_final = t_grid[-1]
            solution_an = analytical_sol(x_grid, t_final, c=c_const)
            error_vec = solution_num[-1, :] - solution_an

            max_error = np.max(np.abs(error_vec))
            l2_error = np.sqrt(h * np.sum(error_vec**2)) # L2 norm calculation

            # Check errors before logging
            valid_max = np.isfinite(max_error) and max_error > 1e-15 # Need positive error for log
            valid_l2 = np.isfinite(l2_error) and l2_error > 1e-15   # Need positive error for log

            if not valid_max or max_error > 1e4: print(f"Invalid max_err ({max_error:.2e}) Nx={Nx_test}. Skip."); continue
            if not valid_l2 or l2_error > 1e4: print(f"Invalid l2_err ({l2_error:.2e}) Nx={Nx_test}. Skip."); continue

            errors_max.append(max_error)
            errors_l2.append(l2_error)
            h_values.append(h)

        except OverflowError: print(f"Overflow Nx={Nx_test}. Skip."); continue
        except Exception as e: print(f"Error Nx={Nx_test}: {e}. Skip."); continue

    warnings.filterwarnings('default', 'invalid value encountered in log')
    warnings.filterwarnings('default', 'divide by zero encountered in log')

    # --- Curve Fitting and Plotting (Using all valid points) ---
    results = {}
    for norm_name, errors in [("Max Norm (Linf)", errors_max), ("L2 Norm", errors_l2)]:
        print(f"\n-- Fitting for {norm_name} --")
        if len(errors) < 3:
            print(f"Not enough valid points ({len(errors)} < 3) for {norm_name}.")
            results[norm_name] = None
            continue

        log_h = np.log(np.array(h_values))
        log_error = np.log(np.array(errors))

        valid_indices = np.isfinite(log_h) & np.isfinite(log_error)
        log_h_valid = log_h[valid_indices]
        log_error_valid = log_error[valid_indices]

        if len(log_h_valid) < 3:
            print(f"Not enough valid points ({len(log_h_valid)} < 3) after log/filter for {norm_name}.")
            results[norm_name] = None
            continue

        try:
            # *** Fit using ALL valid points ***
            popt, pcov = curve_fit(linear_regr, log_h_valid, log_error_valid)
            convergence_order = popt[0]
            perr = np.sqrt(np.diag(pcov))
            slope_std_err = perr[0]

            print(f"Estimated Order ({norm_name}): {convergence_order:.3f} +/- {slope_std_err:.3f}")
            results[norm_name] = convergence_order # Store result

            plt.figure(figsize=(10, 6))
            plt.scatter(log_h_valid, log_error_valid, marker='o', label=f'log({norm_name}) vs log(h)')

            # *** Plot fit line over the full range of valid data ***
            x_plot_fit = np.linspace(min(log_h_valid), max(log_h_valid), 100)
            plt.plot(x_plot_fit, linear_regr(x_plot_fit, popt[0], popt[1]), color='red', linestyle='--',
                     label=f'Linear Fit (Slope={convergence_order:.3f}) on all {len(log_h_valid)} points')

            # Add reference lines
            theo_order_guess = 1 if any(s in scheme_func.name for s in ['Upwind', 'Implicit', 'Friedrichs']) else (2 if 'Wendroff' in scheme_func.name else 0)
            if abs(round(convergence_order) - theo_order_guess) < 0.6 : # Plot reference if close
                 plt.plot(x_plot_fit, linear_regr(x_plot_fit, theo_order_guess, popt[1]), color='gray', linestyle=':', label=f'Slope = {theo_order_guess} Ref')
            elif theo_order_guess == 2 and abs(round(convergence_order)-1) < 0.6: # Maybe show slope 1 if LW is behaving like 1st order
                 plt.plot(x_plot_fit, linear_regr(x_plot_fit, 1, popt[1]), color='gray', linestyle=':', label='Slope = 1 Ref')


            plt.xlabel("log(h) (Spatial Step Size)")
            plt.ylabel(f"log(Error) ({norm_name})")
            plt.title(f"Convergence ({norm_name}) for {scheme_func.name}")
            plt.grid(True); plt.legend(); plt.show()

        except Exception as e:
            print(f"Error during curve fitting for {norm_name}: {e}")
            results[norm_name] = None
            # Plot data even if fit fails
            plt.figure(figsize=(10, 6))
            plt.scatter(log_h_valid, log_error_valid, marker='o', label=f'log({norm_name}) vs log(h)')
            plt.xlabel("log(h)"); plt.ylabel(f"log(Error) ({norm_name})")
            plt.title(f"Convergence Data ({norm_name}) for {scheme_func.name} (Fit Failed)")
            plt.grid(True); plt.legend(); plt.show()

    # Return the order obtained from the L2 norm if available, otherwise Max norm
    return results.get("L2 Norm", results.get("Max Norm (Linf)"))


# --- Main Execution --- (Adjust r_stable for the main call)

def plot_solution_and_heatmap(name, u_num, x_g, t_g, nt_run):
    if u_num is None or x_g is None or t_g is None:
        print(f"Skip plots {name}: sim failed.")
        return
    if np.any(np.isnan(u_num)):
        print(f"NaN in {name} result. Plotting may be limited or fail.")
    t_final = t_g[min(len(t_g)-1, nt_run-1)]
    u_ana_final = analytical_solution(x_g, t_final, c=C_VAL)
    r_actual_sim = C_VAL * (t_g[1]-t_g[0]) / (x_g[1]-x_g[0]) if len(t_g)>1 and len(x_g)>1 and abs(x_g[1]-x_g[0])>1e-14 else 0

    plt.figure(figsize=(12, 7))
    num_slices = 6
    actual_steps = len(t_g)
    time_indices = np.linspace(0, actual_steps - 1, num_slices, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, num_slices))
    for i, idx in enumerate(time_indices):
        t_plot = t_g[idx]
        label = f't={t_plot:.2f}' + (' (I)' if i==0 else '') + (' (F)' if idx==actual_steps-1 else '')
        if np.all(np.isfinite(u_num[idx, :])):
            plt.plot(x_g, u_num[idx, :], color=colors[i], ls='-', marker='.', ms=3, lw=1.5, label=label)
        else:
            print(f"Skipping plot slice for t={t_plot:.2f} due to non-finite values.")
    plt.plot(x_g, initial_condition(x_g), 'k:', lw=1.5, label='Ana t=0')
    plt.plot(x_g, u_ana_final, 'r--', lw=2, label=f'Ana t={t_final:.2f} (or last valid)')
    plt.xlabel("x"); plt.ylabel("u(x, t)")
    plt.title(f"{name} Evol (c={C_VAL}, Nx={Nx}, Nt={nt_run}, r~{r_actual_sim:.3f})")
    plt.grid(True); plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    valid_u = u_num[np.isfinite(u_num)]
    min_val = min(initial_condition(x_g).min(), u_ana_final.min(), valid_u.min() if len(valid_u)>0 else 0)
    max_val = max(initial_condition(x_g).max(), u_ana_final.max(), valid_u.max() if len(valid_u)>0 else 1)
    plt.ylim(min_val-0.1, max_val+0.1); plt.tight_layout(); plt.show()

    if np.any(np.isfinite(u_num)):
        plt.figure(figsize=(10, 7))
        last_valid_idx = actual_steps - 1
        while last_valid_idx > 0 and not np.all(np.isfinite(u_num[last_valid_idx, :])):
            last_valid_idx -= 1
        if last_valid_idx >= 0:
            im = plt.imshow(u_num[:last_valid_idx+1, :].T, extent=[X_MIN, X_BORDER, t_g[0], t_g[last_valid_idx]],
                            origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
            plt.colorbar(im, label='u(x, t)'); plt.xlabel("x"); plt.ylabel("t")
            plt.title(f"{name} Heatmap (Up to t={t_g[last_valid_idx]:.2f}) (c={C_VAL})")
            plt.tight_layout(); plt.show()
        else:
            print(f"Skipping heatmap for {name} as no valid time steps found.")
    else:
        print(f"Skipping heatmap for {name} due to all NaN results.")


if __name__ == "__main__":
    # ... (предыдущие настройки Nx, Nt_base, h_sim, target_r_explicit, etc.) ...
    Nx = 101
    Nt_base = 101
    h_sim = (X_BORDER - X_MIN) / (Nx - 1)
    print(f"Base Simulation settings: Nx={Nx}, h={h_sim:.4f}")
    target_r_explicit = 0.2 # Используем малое r
    # ... (расчет Nt_explicit, r_sim_explicit как раньше) ...
    if abs(C_VAL) > 1e-12:
        tau_explicit_target = target_r_explicit * h_sim / abs(C_VAL); Nt_explicit = int(np.ceil(T_BORDER / tau_explicit_target)) + 1
        if Nt_explicit < 2: Nt_explicit = 2
        tau_sim_explicit = T_BORDER / (Nt_explicit - 1) if Nt_explicit > 1 else T_BORDER
        r_sim_explicit = C_VAL * tau_sim_explicit / h_sim if abs(h_sim) > 1e-14 else float('inf')
        print(f"Explicit scheme settings: Target r={target_r_explicit:.3f} -> Nt={Nt_explicit}, tau={tau_sim_explicit:.4f}, Actual r={r_sim_explicit:.4f}")
    else: Nt_explicit = Nt_base; r_sim_explicit = 0.0; print(f"Warn: C=0. Using Nt={Nt_explicit}, r=0.")
    Nt_implicit = Nt_base

    schemes_to_run = {
        "Left Upwind": (left_upwind_scheme_vectorized, Nt_explicit),
        "Implicit Scheme B": (implicit_scheme_b_vectorized_precomp, Nt_implicit),
        "Lax-Friedrichs": (lax_friedrichs_scheme_vectorized, Nt_explicit),
        "Lax-Wendroff": (lax_wendroff_scheme_vectorized, Nt_explicit),
        "FTCS (Unstable)": (ftcs_scheme_vectorized, Nt_explicit),
    }

    if Nx < 2:
        raise ValueError("Nx must be at least 2.")
    results = {}
    # --- Цикл запуска симуляций и построения графиков ---
    for name, (func, nt_run) in schemes_to_run.items():
        print(f"\nRun {name} Nx={Nx}, Nt={nt_run}...");
        try:
            results[name] = func(velocity, source_term, initial_condition, left_boundary_condition, right_boundary_condition, X_MIN, X_BORDER, T_BORDER, Nx, nt_run)
        except Exception as e:
            print(f"!!! Fail {name}: {e}")
            results[name] = (None, None, None)
            continue
        u_num, x_g, t_g = results[name]
        plot_solution_and_heatmap(name, u_num, x_g, t_g, nt_run)

    # --- Анализ сходимости ---
    print("\n" + "="*30 + "\nStarting Convergence Analysis (Including FTCS)\n" + "="*30)
    convergence_results = {}
    for name, (func, _) in schemes_to_run.items():
        scheme_function = func
        if name not in results or results[name][0] is None:
            print(f"Skip converge {name}: base sim failed or not run.")
            continue

        convergence_results[name] = calculate_convergence(
                                         scheme_function, analytical_solution, velocity, source_term,
                                         initial_condition, left_boundary_condition, right_boundary_condition,
                                         X_MIN, X_BORDER, T_BORDER, N_steps=7, Nx_base=41,
                                         r_stable=target_r_explicit, c_const=C_VAL)

    print("\n--- Convergence Summary ---")
    for name, order in convergence_results.items():
        if order is not None:
            theo_order = 1 if any(s in name for s in ['Upwind', 'Implicit', 'Friedrichs']) else (2 if 'Wendroff' in name else -1)
            print(f"{name}: Estimated Order (from L2/MaxInt) = {order:.3f} (Theoretical: ~{theo_order})")
        elif "FTCS" in name:
             print(f"{name}: Convergence analysis skipped (unstable scheme).")
        else:
            print(f"{name}: Convergence analysis failed.")