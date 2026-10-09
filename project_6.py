import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

print("Solving single scenario optimization...")

# --- Network Data ---
n = 5
T = 6  
np.random.seed(42)
W_raw = np.random.rand(n, n)
W_raw[1,1] =.01
W_raw[2,1] =.01
W_raw[3,1] =.01
W_raw[4,1] =.01
W = W_raw / W_raw.sum(axis=1, keepdims=True)

x_initial = np.ones(n) * 2.0  

# Tightened budget parameters to force sparse switching
U_max = 15.0          
step_budget = 3.0     
person_budget = 3.0   
cap = 1.0  

eigenvalues, eigenvectors = np.linalg.eig(W.T)
c = np.real(eigenvectors[:, np.isclose(eigenvalues, 1.0)].flatten())
c = c / np.sum(c) 

# --- Heterogeneous Ad Pricing ---
ad_prices = (c / np.min(c)) ** 1 #

# --- CVXPY Optimization ---
U = cp.Variable((n, T))
X = cp.Variable((n, T + 1))

a = 1
# Linear penalty for Bang-Bang control
cost = cp.multiply(a*ad_prices[:, None]+(1-a), cp.abs(U))

constraints = [
    X[:, 0] == x_initial,
    cp.sum(cost) <= U_max,                 
    cp.sum(cost, axis=0) <= step_budget,   
    cp.sum(cost, axis=1) <= person_budget, 
    cp.abs(U) <= cap                       
]

for k in range(T):
    constraints.append( X[:, k+1] == W @ X[:, k] + U[:, k] )

# Minimize Cumulative Consensus (Time-Sensitive)
cumulative_consensus = sum(c @ X[:, k] for k in range(1, T + 1))
objective = cp.Minimize(cumulative_consensus)

prob = cp.Problem(objective, constraints)
prob.solve()
optimal_U = U.value

print(f"True Infinite-Horizon Consensus: {c @ X[:, T].value:.4f}")

# --- Plotting Part 1 ---
cost_matrix = ad_prices[:, None] * np.abs(optimal_U)

fig1, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, 
    ncols=1, 
    figsize=(12, 15), 
    sharex=False, 
    gridspec_kw={'height_ratios': [1, 1.5, 1.5]}
)
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Create a figure with slightly more height to accommodate both rows cleanly
fig1 = plt.figure(figsize=(14, 10))

# Define a 2x2 grid
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3.5], wspace=0.05, hspace=0.3)

# ==============================================================================
# 1. TOP ROW: Social Power & Influence Volume Heatmap
# ==============================================================================
# Left: Social Power (Top)
ax1_top = fig1.add_subplot(gs[0, 0])
ax1_top.barh(np.arange(n), c, color='#d62728', alpha=0.85)
ax1_top.set_title('1. Social Power', fontsize=14, fontweight='bold')
ax1_top.set_yticks(np.arange(n))
ax1_top.grid(True, linestyle='--', alpha=0.5)
ax1_top.invert_xaxis()
ax1_top.yaxis.tick_right()
ax1_top.set_ylabel('Network Agent ($i$)', fontweight='bold')
ax1_top.yaxis.set_label_position("right")

# Right: Influence Volume Heatmap
ax2_top = fig1.add_subplot(gs[0, 1], sharey=ax1_top) 
im2 = ax2_top.imshow(np.abs(optimal_U), aspect='auto', cmap='Blues', origin='lower')
ax2_top.set_title('2. Effective Influence Heatmap (Volume: $|U|$)', fontsize=14, fontweight='bold')
ax2_top.set_xticks(np.arange(T))
ax2_top.set_yticks(np.arange(n) - 0.5, minor=True)
ax2_top.set_xticks(np.arange(T) - 0.5, minor=True)
ax2_top.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)
ax2_top.tick_params(which="minor", bottom=False, left=False)
plt.setp(ax2_top.get_yticklabels(), visible=False) 

cbar2 = fig1.colorbar(im2, ax=ax2_top, orientation='vertical', pad=0.02, fraction=0.05)
cbar2.set_label('Effective Influence Volume', fontweight='bold')


# ==============================================================================
# 2. BOTTOM ROW: Social Power & Financial Cost Heatmap
# ==============================================================================
# Left: Social Power (Bottom)
ax1_bot = fig1.add_subplot(gs[1, 0])
ax1_bot.barh(np.arange(n), c, color='#d62728', alpha=0.85)
ax1_bot.set_xlabel('Social Power ($c_i$)', fontweight='bold')
ax1_bot.set_yticks(np.arange(n))
ax1_bot.grid(True, linestyle='--', alpha=0.5)
ax1_bot.invert_xaxis()
ax1_bot.yaxis.tick_right()
ax1_bot.set_ylabel('Network Agent ($i$)', fontweight='bold')
ax1_bot.yaxis.set_label_position("right")

# Right: Financial Cost Heatmap
ax2_bot = fig1.add_subplot(gs[1, 1], sharey=ax1_bot)
im3 = ax2_bot.imshow(cost_matrix, aspect='auto', cmap='Reds', origin='lower')
ax2_bot.set_xlabel('Time Step ($k$)', fontweight='bold')
ax2_bot.set_title('3. Financial Cost Heatmap (Price $\\times$ Volume)', fontsize=14, fontweight='bold')
ax2_bot.set_xticks(np.arange(T))
ax2_bot.set_yticks(np.arange(n) - 0.5, minor=True)
ax2_bot.set_xticks(np.arange(T) - 0.5, minor=True)
ax2_bot.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)
ax2_bot.tick_params(which="minor", bottom=False, left=False)
plt.setp(ax2_bot.get_yticklabels(), visible=False) 

cbar3 = fig1.colorbar(im3, ax=ax2_bot, orientation='vertical', pad=0.02, fraction=0.05)
cbar3.set_label('Financial Cost ($p_i \\times |U_{i,k}|$)', fontweight='bold')

fig1.tight_layout()
plt.show()
# ==============================================================================
# PART 2: Pareto Frontier Sweep
# ==============================================================================
#print("Solving budget sweeps for the Pareto Frontier...")

# Lock the specific constraints used for the budget sweep
#step_budget_sweep = 2.0     
#person_budget_sweep = 3.0   
#cap_sweep = 2.0  

#budgets = np.linspace(0, 25, 26)
#optimal_consensus = []

#for u_limit in budgets:
 #   U_sweep = cp.Variable((n, T))
 #   X_sweep = cp.Variable((n, T + 1))

 #   constraints_sweep = [
 #       X_sweep[:, 0] == x_initial,
 #       cp.sum(cp.abs(U_sweep)) <= u_limit,          
 #       cp.sum(cp.abs(U_sweep), axis=0) <= step_budget_sweep,   
 #       cp.sum(cp.abs(U_sweep), axis=1) <= person_budget_sweep, 
 #       cp.abs(U_sweep) <= cap_sweep                       
    

 #   for k in range(T):
 #       constraints_sweep.append( X_sweep[:, k+1] == W @ X_sweep[:, k] + U_sweep[:, k] )

    # Minimize Cumulative Consensus
 #   cum_consensus_sweep = sum(c @ X_sweep[:, k] for k in range(1, T + 1))
 #   objective_sweep = cp.Minimize(cum_consensus_sweep)

#    prob_sweep = cp.Problem(objective_sweep, constraints_sweep)
#    prob_sweep.solve()

 #   optimal_consensus.append(prob_sweep.value)

# --- Plotting Part 2 ---
#fig2, ax = plt.subplots(figsize=(10, 6))

#ax.plot(budgets, optimal_consensus, color='black', linewidth=3, marker='o')

# Highlight the absolute bottleneck (n * person_budget = 15.0)
#ax.axvline(x=15.0, color='red', linestyle='--', linewidth=2, label='Maximum Network Absorption (15.0)')

#ax.set_xlabel('Global Budget Allowed ($U_{\max}$)', fontweight='bold')
#ax.set_ylabel('Optimal Cumulative Consensus Achieved', fontweight='bold')
#ax.set_title('Pareto Frontier: The Cost-Effectiveness Trade-off', fontsize=14, fontweight='bold')
#ax.grid(True, linestyle='--', alpha=0.6)
#ax.legend()

#plt.show()