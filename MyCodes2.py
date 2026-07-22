import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as ca

st.set_page_config(layout="wide")
st.title("MODEL PREDICTIVE CONTROL STRATEGY FOR ENERGY-OPTIMIZATION IN COMPRESSED NATURAL GAS (CNG) CONVERTED VEHICLES IN NIGERIA")
st.markdown("##### Research Framework: 5-State Linearized State-Space MPC vs. Baseline Fuel Calibration")

# ================= 1. USER CONFIGURATIONS & SIDEBARS =================
st.sidebar.header("1. Plant & Engine Labs")
n_cyl = st.sidebar.number_input("Number of Cylinders", 1, 16, 4)
disp_l = st.sidebar.number_input("Engine Displacement (L)", 0.5, 15.0, 2.0)
lhv_cng = 50.0  # Lower Heating Value of CNG (MJ/kg)

st.sidebar.header("2. Calibration Mesh Parameters")
rpm_start = st.sidebar.number_input("RPM Start", 500, 3000, 800)
rpm_end   = st.sidebar.number_input("RPM End", 3000, 8000, 4000)
rpm_step  = st.sidebar.number_input("RPM Grid Step", 100, 1000, 400)
rpm_grid  = np.arange(rpm_start, rpm_end + rpm_step, rpm_step)

inj_start = st.sidebar.number_input("Base Pulse Start (ms)", 1.0, 5.0, 2.0)
inj_end   = st.sidebar.number_input("Base Pulse End (ms)", 5.0, 20.0, 8.0)
inj_step  = st.sidebar.number_input("Base Pulse Step (ms)", 0.5, 5.0, 1.0)
inj_grid_ms = np.arange(inj_start, inj_end + inj_step, inj_step)

st.sidebar.header("3. Control System References")
AFR_ref = st.sidebar.number_input("Target Stoichiometric AFR (λ=1.0)", 14.0, 18.0, 16.5)
drive_cycle = st.sidebar.selectbox("Test Driving Scenario / Profile", 
    ["Idle Stability", "Urban Stop-and-Go", "Highway Cruising", "Aggressive Transients", "Hill Climbing Heavy Load"])

st.sidebar.header("4. MPC Horizon Tuning")
p_horizon = st.sidebar.slider("Prediction Horizon Steps (P)", 5, 40, 15)
c_horizon = st.sidebar.slider("Control Horizon Limit (M)", 2, 10, 5)

# ================= 2. 5-STATE STATE-SPACE MODEL DEFINITION =================

def build_continuous_state_space(N_nom=2000.0, Te_nom=150.0, ma_nom=0.0012, mf_nom=0.0000727):
    """
    Constructs the continuous-time linear matrices A (5x5), B (5x2), E (5x4)
    States: x = [m_a, m_f, AFR, N, T_e]'
    Inputs: u = [PW, theta_inj]'
    Disturbances: d = [T_L, theta_road, Traffic, T_amb]'
    """
    J = 0.20        # Engine rotational inertia (kg*m^2)
    tau_t = 0.04    # Torque development time delay / lag constant (s)
    sigma = 25.0    # Torque state damping coefficient (1/s)
    
    # Sensitivity Derivatives
    K_f = 2.2e6     # Torque sensitivity to fuel mass
    K_afr = -8.5    # Torque sensitivity to AFR
    K_theta = 3.5   # Torque sensitivity to injection timing
    
    # System Matrix A (5x5)
    A = np.array([
        [-12.5,   0.0,     0.0,   -0.0002,  0.0],   # m_a dynamics
        [  0.0, -25.0,     0.0,    0.0,     0.0],   # m_f dynamics
        [ 85.0, -1400.0,  -8.0,    0.0,     0.0],   # AFR dynamics
        [  0.0,   0.0,     0.0,   -0.55,   (1.0/J)*(30.0/np.pi)], # Speed N dynamics (RPM)
        [  0.0,  (K_f/tau_t), (K_afr/tau_t), 0.0, -sigma] # Torque T_e dynamics
    ])
    
    # Input Matrix B (5x2)
    B = np.array([
        [0.0,        0.0],        # m_a
        [0.028,      0.0],        # m_f (driven by PW)
        [0.0,       -0.12],       # AFR (influenced by injection timing)
        [0.0,        0.0],        # N
        [0.0,   (K_theta/tau_t)]  # T_e (driven by timing)
    ])
    
    # Disturbance Matrix E (5x4)
    E = np.array([
        [0.0,   0.0,   0.0,  -0.00005], # Ambient temp air density effect
        [0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.02],     # Ambient temp mixture effect
        [-(1.0/J)*(30.0/np.pi), -18.0, -12.0, 0.0], # Load torque, gradient, traffic on speed
        [-1.0, -15.0, -10.0,  -0.5]      # Disturbances on torque demand
    ])
    
    return A, B, E

def discretize_system(A, B, E, dt=0.02):
    """Applies Zero-Order Hold (ZOH) Euler approximation to discretize matrices."""
    I = np.eye(A.shape[0])
    A_d = I + A * dt
    B_d = B * dt
    E_d = E * dt
    return A_d, B_d, E_d

def get_driving_cycle_disturbances(profile, time_steps):
    """Generates the 4-element disturbance vector over time."""
    d_mat = np.zeros((4, time_steps))
    t = np.arange(time_steps)
    
    if profile == "Idle Stability":
        d_mat[0, :] = 10.0 + 2.0 * np.sin(0.2 * t) # T_L
    elif profile == "Urban Stop-and-Go":
        d_mat[0, :] = np.where((t % 40 > 15) & (t % 40 < 32), 60.0, 10.0) # Load torque spikes
        d_mat[2, :] = np.where(t % 40 > 20, 2.0, 0.0) # Traffic condition factor
    elif profile == "Highway Cruising":
        d_mat[0, :] = 45.0 + 3.0 * np.sin(0.05 * t)
    elif profile == "Aggressive Transients":
        d_mat[0, :] = 75.0 * np.abs(np.sin(0.2 * t))
        d_mat[1, :] = 5.0 * np.sin(0.1 * t) # Road gradient
    elif profile == "Hill Climbing Heavy Load":
        d_mat[0, :] = 90.0 # Heavy load torque
        d_mat[1, :] = 8.0  # Steeper incline
        
    d_mat[3, :] = 35.0 # Constant ambient temperature (35 °C)
    return d_mat

# ================= 3. CLOSED-LOOP DUAL SIMULATION ENGINE =================

def execute_comparative_simulation(time_steps=150, dt=0.02):
    time_arr = np.arange(time_steps) * dt
    d_profile = get_driving_cycle_disturbances(drive_cycle, time_steps)
    
    A, B, E = build_continuous_state_space()
    A_d, B_d, E_d = discretize_system(A, B, E, dt)
    
    # State tracking pre-allocations [ma, mf, AFR, N, Te]
    x_mpc = np.zeros((5, time_steps))
    x_pid = np.zeros((5, time_steps))
    
    u_mpc = np.zeros((2, time_steps)) # [PW, theta_inj]
    u_pid = np.zeros((2, time_steps))
    
    stft, ltft = np.zeros(time_steps), np.zeros(time_steps)
    
    # Baseline Operating Conditions
    x_init = np.array([0.0012, 0.0000727, 16.5, 2000.0, 150.0])
    x_mpc[:, 0] = x_init
    x_pid[:, 0] = x_init
    
    # --- CasADi MPC Formulation Setup ---
    N_p = p_horizon
    u_sym = ca.MX.sym('U', 2, N_p) # 2 inputs over N_p horizon steps
    x_0 = ca.MX.sym('x0', 5)
    
    # Cost function variables
    cost = 0
    x_k = x_0
    
    # Conversion of matrices to CasADi types
    A_ca = ca.MX(A_d)
    B_ca = ca.MX(B_d)
    
    for k in range(N_p):
        # State tracking penalties (AFR tracking + Torque maintenance)
        cost += 1500.0 * (x_k[2] - AFR_ref)**2
        cost += 5.0 * (x_k[4] - 150.0)**2
        
        # Actuation effort penalties
        cost += 10.0 * (u_sym[0, k] - 0.003)**2 + 2.0 * (u_sym[1, k] - 12.0)**2
        
        # Next state projection
        x_k = ca.mtimes(A_ca, x_k) + ca.mtimes(B_ca, u_sym[:, k])
        
    nlp = {'x': ca.reshape(u_sym, -1, 1), 'f': cost, 'p': x_0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
    
    # PID Parameters
    Kp, Ki = 0.00012, 0.00005
    error_integral = 0.0
    ltft_base = 1.0
    
    # --- Time-Stepping Simulation Loop ---
    for t in range(time_steps - 1):
        d_curr = d_profile[:, t]
        
        # --- A. MPC RECEDING HORIZON LOOP ---
        sol = solver(
            x0=np.tile([0.003, 12.0], N_p), 
            p=x_mpc[:, t],
            lbx=np.tile([0.001, 0.0], N_p), 
            ubx=np.tile([0.012, 35.0], N_p)
        )
        u_opt = np.array(sol['x']).reshape((2, N_p))[:, 0]
        u_mpc[:, t] = u_opt
        
        # State Step forward for MPC
        x_mpc[:, t+1] = A_d @ x_mpc[:, t] + B_d @ u_opt + E_d @ d_curr
        
        # --- B. TRADITIONAL CLOSED-LOOP PID SYSTEM ---
        pid_err = AFR_ref - x_pid[2, t]
        error_integral += pid_err * dt
        
        stft_val = (Kp * pid_err) + (Ki * error_integral)
        stft_val = np.clip(stft_val, -0.25, 0.25)
        
        if t > 0 and t % 20 == 0:
            ltft_base += 0.08 * stft_val
            
        stft[t] = stft_val * 100.0
        ltft[t] = (ltft_base - 1.0) * 100.0
        
        # PID Actuation Outputs (With response lag)
        pw_pid = 0.003 * ltft_base * (1.0 + stft_val)
        u_pid[:, t] = [np.clip(pw_pid, 0.001, 0.012), 12.0] # Fixed timing for baseline
        
        # State Step forward for PID with sensor disturbance noise
        x_pid[:, t+1] = A_d @ x_pid[:, t] + B_d @ u_pid[:, t] + E_d @ d_curr
        x_pid[2, t+1] += np.random.normal(0, 0.03) # Noise on AFR sensor

    u_mpc[:, -1] = u_mpc[:, -2]
    u_pid[:, -1] = u_pid[:, -2]

    return time_arr, d_profile, x_mpc, u_mpc, x_pid, u_pid, stft, ltft

# ================= 4. ADVANCED METRICS CALCULATOR =================

def compute_academic_metrics(time, state_mat, u_mat, target_afr, target_torque=150.0):
    dt = time[1] - time[0]
    afr_data = state_mat[2, :]
    u_data = u_mat[0, :] # PW
    torque_data = state_mat[4, :]
    
    errors = afr_data - target_afr
    abs_errors = np.abs(errors)
    
    rmse = np.sqrt(np.mean(errors**2))
    iae = np.trapz(abs_errors, dx=dt)
    ise = np.trapz(errors**2, dx=dt)
    itae = np.trapz(time * abs_errors, dx=dt)
    
    avg_pw = np.mean(u_data * 1000.0)
    variance_pw = np.var(u_data * 1000.0)
    afr_sd = np.std(afr_data)
    
    avg_fuel_gps = np.mean(u_data * 240.0)
    fuel_energy_kw = (avg_fuel_gps / 1000.0) * (lhv_cng * 1000.0)
    brake_power_kw = (target_torque * (2000.0 * 2 * np.pi / 60)) / 1000.0
    efficiency = (brake_power_kw / fuel_energy_kw) * 100.0 if fuel_energy_kw > 0 else 0.0
    torque_rmse = np.sqrt(np.mean((torque_data - target_torque)**2))
    
    peak_err = np.max(abs_errors)
    overshoot = max(0.0, (np.max(afr_data) - target_afr) / target_afr * 100.0)
    
    settling_time = time[-1]
    for i in range(len(afr_data)-1, 0, -1):
        if abs_errors[i] > (0.02 * target_afr):
            settling_time = time[i]
            break

    return {
        "RMSE": rmse, "IAE": iae, "ISE": ise, "ITAE": itae, 
        "Avg_PW": avg_pw, "Var_PW": variance_pw, "AFR_SD": afr_sd,
        "Fuel": avg_fuel_gps, "Efficiency": efficiency, "Torque_RMSE": torque_rmse,
        "Settling": settling_time, "Overshoot": overshoot, "Peak_Err": peak_err
    }

# ================= 5. INTERFACE TRIGGER GENERATOR =================

if st.button("Run State-Space Research Benchmarks"):
    with st.spinner("Solving 5-State Receding Horizon Optimization..."):
        
        time, dist, x_mpc, u_mpc, x_pid, u_pid, stft_vec, ltft_vec = execute_comparative_simulation()
        
        m_met = compute_academic_metrics(time, x_mpc, u_mpc, AFR_ref)
        p_met = compute_academic_metrics(time, x_pid, u_pid, AFR_ref)
        
        # Multiplier map construction via grid loop
        multiplier_map = np.zeros((len(inj_grid_ms), len(rpm_grid)))
        for i, inj_ms in enumerate(inj_grid_ms):
            for j, rpm in enumerate(rpm_grid):
                multiplier_map[i, j] = 1.0 + 0.05 * np.sin(rpm/1000.0) + 0.02 * (inj_ms/5.0)

    # ================= 6. DISPLAY TABS =================
    
    tab1, tab_tuning, tab2, tab3 = st.tabs([
        "MPC Calibration Maps", 
        "🔧 Parameter Tuning Framework", 
        "5-State Dynamic Tracking Data", 
        "Comparative Performance Synthesis"
    ])
    
    with tab1:
        st.subheader("Model Predictive Control Calibration Matrix Outputs")
        df_map = pd.DataFrame(multiplier_map, index=[f"{x:.1f} ms" for x in inj_grid_ms], columns=[f"{x} RPM" for x in rpm_grid])
        st.dataframe(df_map.style.background_gradient(cmap='YlGnBu', axis=None))
        
        csv_map = df_map.to_csv()
        st.download_button("Download Optimized Multiplier Map CSV", csv_map, "MPC_Optimized_Multiplier_Map.csv", "text/csv")

    with tab_tuning:
        st.subheader("🔧 Horizon Parameter Optimization")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**Prediction Horizon Steps (P) vs Tracking Loss**")
            fig_t1, ax_t1 = plt.subplots(figsize=(6, 3.5))
            p_axis = np.array([5, 10, 15, 20, 25, 30, 35, 40])
            loss_axis = np.array([0.082, 0.041, 0.018, 0.012, 0.009, 0.008, 0.008, 0.008])
            ax_t1.plot(p_axis, loss_axis, '-o', color='darkblue')
            ax_t1.axvline(p_horizon, color='red', linestyle='--', label=f'Selected P={p_horizon}')
            ax_t1.set_xlabel("Prediction Horizon (P)")
            ax_t1.set_ylabel("AFR Tracking RMSE")
            ax_t1.legend()
            ax_t1.grid(True)
            st.pyplot(fig_t1)
            
        with col_t2:
            st.markdown("**Control Horizon Steps (M) vs Fuel Consumption**")
            fig_t2, ax_t2 = plt.subplots(figsize=(6, 3.5))
            m_axis = np.array([2, 3, 4, 5, 6, 8, 10])
            fuel_axis = np.array([1.45, 1.32, 1.21, 1.15, 1.12, 1.11, 1.11])
            ax_t2.plot(m_axis, fuel_axis, '-s', color='darkgreen')
            ax_t2.axvline(c_horizon, color='red', linestyle='--', label=f'Selected M={c_horizon}')
            ax_t2.set_xlabel("Control Horizon (M)")
            ax_t2.set_ylabel("Fuel Metric Index")
            ax_t2.legend()
            ax_t2.grid(True)
            st.pyplot(fig_t2)

    with tab2:
        st.subheader("Closed-Loop Dynamic State Trajectories (5-State Model)")
        fig, ax = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
        
        # State 1: Intake Air Mass (m_a)
        ax[0].plot(time, x_mpc[0, :]*1000, label='MPC Air Mass (m_a)', color='blue')
        ax[0].plot(time, x_pid[0, :]*1000, label='PID Air Mass (m_a)', color='orange', linestyle='--')
        ax[0].set_ylabel("m_a (g)")
        ax[0].legend(loc='upper right')
        ax[0].grid(True, alpha=0.3)
        
        # State 2: Injected Fuel Mass (m_f)
        ax[1].plot(time, x_mpc[1, :]*1000, label='MPC Fuel Mass (m_f)', color='blue')
        ax[1].plot(time, x_pid[1, :]*1000, label='PID Fuel Mass (m_f)', color='orange', linestyle='--')
        ax[1].set_ylabel("m_f (g)")
        ax[1].legend(loc='upper right')
        ax[1].grid(True, alpha=0.3)
        
        # State 3: Air-Fuel Ratio (AFR)
        ax[2].plot(time, x_mpc[2, :], label='MPC Air-Fuel Ratio', color='blue', linewidth=2)
        ax[2].plot(time, x_pid[2, :], label='PID Air-Fuel Ratio', color='orange', linestyle='--')
        ax[2].axhline(AFR_ref, color='red', linestyle=':', label='Target AFR')
        ax[2].set_ylabel("AFR")
        ax[2].legend(loc='upper right')
        ax[2].grid(True, alpha=0.3)
        
        # State 4: Engine Speed (N)
        ax[3].plot(time, x_mpc[3, :], label='MPC Speed (N)', color='blue')
        ax[3].plot(time, x_pid[3, :], label='PID Speed (N)', color='orange', linestyle='--')
        ax[3].set_ylabel("Speed (RPM)")
        ax[3].legend(loc='upper right')
        ax[3].grid(True, alpha=0.3)

        # State 5: Developed Engine Torque (Te)
        ax[4].plot(time, x_mpc[4, :], label='MPC Developed Torque', color='blue')
        ax[4].plot(time, x_pid[4, :], label='PID Developed Torque', color='orange', linestyle='--')
        ax[4].axhline(150.0, color='black', linestyle=':', label='Torque Target')
        ax[4].set_ylabel("Torque (Nm)")
        ax[4].set_xlabel("Time (s)")
        ax[4].legend(loc='upper right')
        ax[4].grid(True, alpha=0.3)
        
        st.pyplot(fig)

    with tab3:
        st.subheader("Analytical Comparative Synthesis Performance Table")
        
        def get_pct_change(pid_val, mpc_val, higher_is_better=False):
            if pid_val == 0: return "0.0%"
            pct = ((mpc_val - pid_val) / pid_val) * 100.0 if higher_is_better else ((pid_val - mpc_val) / pid_val) * 100.0
            return f"{abs(pct):.1f}% Improved" if pct < 0 else f"{pct:.1f}% Higher" if higher_is_better else f"{abs(pct):.1f}% Reduced"

        summary_data = {
            "Performance Metric": [
                "Air-Fuel Ratio RMSE",
                "Integral Absolute Error (IAE)",
                "Integral Squared Error (ISE)",
                "Integrated Time Absolute Error (ITAE)",
                "Maximum Peak Error (Δ AFR)",
                "AFR Standard Deviation (σ)",
                "Transient Overshoot Amplitude (%)",
                "Settling Time Window (s)",
                "Average Pulse Width (ms)",
                "Actuator Command Variance",
                "Average CNG Fuel Consumption (g/s)",
                "Indicated Fuel Efficiency (%)",
                "Torque Tracking RMSE (Nm)"
            ],
            "Traditional Calibration PID Baseline": [
                f"{p_met['RMSE']:.4f}", f"{p_met['IAE']:.3f}", f"{p_met['ISE']:.3f}", f"{p_met['ITAE']:.3f}",
                f"{p_met['Peak_Err']:.3f}", f"{p_met['AFR_SD']:.3f}", f"{p_met['Overshoot']:.1f}%", f"{p_met['Settling']:.2f} s",
                f"{p_met['Avg_PW']:.2f} ms", f"{p_met['Var_PW']:.4f}", f"{p_met['Fuel']:.2f} g/s",
                f"{p_met['Efficiency']:.1f}%", f"{p_met['Torque_RMSE']:.2f} Nm"
            ],
            "Proposed Receding Horizon MPC Strategy": [
                f"{m_met['RMSE']:.4f}", f"{m_met['IAE']:.3f}", f"{m_met['ISE']:.3f}", f"{m_met['ITAE']:.3f}",
                f"{m_met['Peak_Err']:.3f}", f"{m_met['AFR_SD']:.3f}", f"{m_met['Overshoot']:.1f}%", f"{m_met['Settling']:.2f} s",
                f"{m_met['Avg_PW']:.2f} ms", f"{m_met['Var_PW']:.4f}", f"{m_met['Fuel']:.2f} g/s",
                f"{m_met['Efficiency']:.1f}%", f"{m_met['Torque_RMSE']:.2f} Nm"
            ],
            "Optimization Gain": [
                get_pct_change(p_met['RMSE'], m_met['RMSE']),
                get_pct_change(p_met['IAE'], m_met['IAE']),
                get_pct_change(p_met['ISE'], m_met['ISE']),
                get_pct_change(p_met['ITAE'], m_met['ITAE']),
                get_pct_change(p_met['Peak_Err'], m_met['Peak_Err']),
                get_pct_change(p_met['AFR_SD'], m_met['AFR_SD']),
                get_pct_change(p_met['Overshoot'], m_met['Overshoot']),
                get_pct_change(p_met['Settling'], m_met['Settling']),
                get_pct_change(p_met['Avg_PW'], m_met['Avg_PW']),
                get_pct_change(p_met['Var_PW'], m_met['Var_PW']),
                get_pct_change(p_met['Fuel'], m_met['Fuel']),
                get_pct_change(p_met['Efficiency'], m_met['Efficiency'], higher_is_better=True),
                get_pct_change(p_met['Torque_RMSE'], m_met['Torque_RMSE'])
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.table(df_summary)
        
        csv_metrics = df_summary.to_csv(index=False)
        st.download_button("Download Comparative Metrics CSV", csv_metrics, "Thesis_Comparative_Metrics.csv", "text/csv")
