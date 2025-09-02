import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration: All simulation parameters are here ---
N_SAMPLES = 5000 
N_FEATURES = 5 
NOISE_LEVEL = 0.1
DELAYS_TO_TEST = [0, 5, 10, 20, 50] 
PACK_SIZES_TO_TEST = [1, 5, 10, 20, 50] 

# --- Data Generation ---
def generate_data(n_samples, n_features, noise):
    """Generates a dataset with some noise."""
    print(f"Generating a dataset with {n_samples} samples and {n_features} features...")
    np.random.seed(42) # for reproducibility
    w_true = np.random.randn(n_features)
    X = np.random.rand(n_samples, n_features) * 2 - 1
    y_true = X @ w_true + np.random.randn(n_samples) * noise
    print("Data generation complete.")
    return X, y_true

# --- Algorithm Implementation ---
class AAIRR:

    def __init__(self, n_features, total_samples):
        self.n_features = n_features
        # State variables from the R function
        self.b = np.zeros(n_features)
        self.A = np.zeros((n_features, n_features))
        self.w = np.ones(n_features)
        # Regularization 'a' is defined based on total samples, as in the R code
        self.a = 1.0 / total_samples if total_samples > 0 else 1.0
        self.name = "AAIRR"

    def predict(self, x):
        """Makes a prediction based on the current state."""
        # 1. Create D matrix based on current weights `w`
        D_diag_sqrt = np.sqrt(np.abs(self.w))
        D_outer = np.outer(D_diag_sqrt, D_diag_sqrt) # This is `D` in R

        # 2. Update covariance matrix for prediction calculation (following R code structure)
        temp_A = self.A + np.outer(x, x)
        
        # 3. Calculate inverse and update weights for prediction
        try:
            # CRITICAL CORRECTION: Use * for element-wise multiplication to match R's D * At
            InvA = np.linalg.inv(np.diag([self.a] * self.n_features) + D_outer * temp_A) 
            # CRITICAL CORRECTION: Use * for element-wise multiplication to match R's D * InvA
            AAt = D_outer * InvA
            prediction_weights = AAt.T @ self.b # A.T @ b is equivalent to crossprod(A, b) for vectors/matrices
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            prediction_weights = self.w 
            
        return np.dot(prediction_weights, x)

    def update(self, x, y):
        """Updates the internal state using the true label."""
        # 1. Update covariance matrix `A` and target vector `b`
        self.A += np.outer(x, x)
        self.b += y * x

        # 2. Re-calculate D based on the *previous* step's weights `self.w`
        D_diag_sqrt = np.sqrt(np.abs(self.w))
        D_outer = np.outer(D_diag_sqrt, D_diag_sqrt) # This is `D` in R

        # 3. Re-calculate the final weights for the *next* time step
        try:
            # CRITICAL CORRECTION: Use * for element-wise multiplication to match R's D * self.A
            InvA = np.linalg.inv(np.diag([self.a] * self.n_features) + D_outer * self.A) 
            # CRITICAL CORRECTION: Use * for element-wise multiplication to match R's D * InvA
            AAt = D_outer * InvA
            self.w = AAt.T @ self.b # A.T @ b is equivalent to crossprod(A, b) for vectors/matrices
        except np.linalg.LinAlgError:
            # Suppress warning as it's common for this type of inverse calculation with certain data
            pass # Keep old weights

# --- Simulation Runner for Delayed Feedback ---
def run_simulation_with_delay(algorithm_class, X_data, y_data, delay, **kwargs):
    """
    Runs a full simulation, handling delayed feedback.
    """
    n_samples, n_features = X_data.shape
    algorithm = algorithm_class(n_features=n_features, total_samples=n_samples, **kwargs)
    
    predictions = np.zeros(n_samples)
    feedback_buffer = collections.deque()

    print(f"  Running simulation for {algorithm.name} with delay = {delay}...")

    current_losses = [] # To calculate cumulative loss step-by-step
    for t in range(n_samples):
        x_t, y_t = X_data[t], y_data[t]
        
        # 1. Make a prediction using the current state
        prediction_t = algorithm.predict(x_t)
        predictions[t] = prediction_t

        # Calculate current step's loss
        step_loss = (prediction_t - y_t)**2
        current_losses.append(step_loss)
        
        # 2. Add the data to the buffer to await feedback
        feedback_buffer.append((x_t, y_t))
        
        # 3. If the buffer is full, receive delayed feedback and update the model
        if len(feedback_buffer) > delay:
            x_delayed, y_delayed = feedback_buffer.popleft()
            algorithm.update(x_delayed, y_delayed)
            
    # Calculate final performance metrics
    rmse = np.sqrt(mean_squared_error(y_data, predictions))
    r2 = r2_score(y_data, predictions)
    cumulative_loss = np.cumsum(current_losses)
    # Replace zeros or very small numbers with a tiny positive value for log plotting (if applicable)
    cumulative_loss[cumulative_loss <= 0] = np.finfo(float).eps 
    
    return {
        'predictions': predictions,
        'cumulative_loss': cumulative_loss,
        'performance': {'RMSE': rmse, 'R2': r2}
    }

# --- Simulation Runner for Pack/Batch Updates ---
def run_simulation_with_packs(algorithm_class, X_data, y_data, pack_size, **kwargs):
    """
    Runs a simulation where data arrives in packs (batches).
    The model predicts for the whole pack, then updates with the pack's data.
    """
    n_samples, n_features = X_data.shape
    algorithm = algorithm_class(n_features=n_features, total_samples=n_samples, **kwargs)
    
    predictions = np.zeros(n_samples)
    
    print(f"  Running simulation for {algorithm.name} with pack_size = {pack_size}...")
    
    # Iterate through the data in chunks of size `pack_size`
    for t in range(0, n_samples, pack_size):
        # Define the current pack
        X_pack = X_data[t : t + pack_size]
        y_pack = y_data[t : t + pack_size]
        
        # 1. Make predictions for the entire pack using the *current* model state
        for i in range(len(X_pack)):
            pred_idx = t + i
            predictions[pred_idx] = algorithm.predict(X_pack[i])
            
        # 2. After all predictions for the pack are made, update the model
        #    with the data from the pack
        for i in range(len(X_pack)):
            algorithm.update(X_pack[i], y_pack[i])
            
    # Calculate final performance metrics
    rmse = np.sqrt(mean_squared_error(y_data, predictions))
    r2 = r2_score(y_data, predictions)
    
    cumulative_loss = np.cumsum((predictions - y_data)**2)
    # Replace zeros or very small numbers with a tiny positive value for log plotting (if applicable)
    cumulative_loss[cumulative_loss <= 0] = np.finfo(float).eps 
    
    return {
        'predictions': predictions,
        'cumulative_loss': cumulative_loss,
        'performance': {'RMSE': rmse, 'R2': r2}
    }

# --- Main Execution and Plotting ---
if __name__ == '__main__':
    # Generate consistent data for all runs
    X_data, y_data = generate_data(N_SAMPLES, N_FEATURES, NOISE_LEVEL)
    
    # --- DELAYED FEEDBACK SIMULATION ---
    delay_results = {}
    print("\n--- Starting Continuous Delayed Feedback Simulations ---")
    for delay in DELAYS_TO_TEST:
        delay_results[delay] = run_simulation_with_delay(AAIRR, X_data, y_data, delay=delay)

    # --- PACKS (BATCH) SIMULATION ---
    pack_results = {}
    print("\n--- Starting Pack (Batch) Simulations ---")
    for k in PACK_SIZES_TO_TEST:
        pack_results[k] = run_simulation_with_packs(AAIRR, X_data, y_data, pack_size=k)

    # --- Specific Delay Comparison for the third plot ---
    print("\n--- Starting Specific Delay Comparison Simulations ---")
    # Scenario 1: Continuous small delay (delay=1 for all steps)
    small_continuous_delay_result = run_simulation_with_delay(AAIRR, X_data, y_data, delay=1)
    
    # Scenario 2: One very large delay (delay = N_SAMPLES - 1, meaning update only at the very end)
    # This simulates "one big delay" where almost all feedback is received at the very end.
    large_single_delay_result = run_simulation_with_delay(AAIRR, X_data, y_data, delay=N_SAMPLES - 1)

    # --- Plot the results ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, 7)) # 1 row, 3 columns

    # Define distinct color maps for clarity
    delay_colors = plt.cm.tab10(np.arange(len(DELAYS_TO_TEST))) 
    pack_colors = plt.cm.tab10(np.arange(len(PACK_SIZES_TO_TEST)))
    line_styles = ['-', '--', ':', '-.'] 

    # Plot 1: Effect of Delayed Feedback on Cumulative Square Loss (Left)
    for i, (delay, result_data) in enumerate(delay_results.items()):
        axes[0].plot(result_data['cumulative_loss'], label=f'Delay = {delay} steps', 
                     color=delay_colors[i], linestyle=line_styles[i % len(line_styles)])
        
    axes[0].set_title('Effect of Delayed Feedback on Cumulative Square Loss', fontsize=16)
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Cumulative Square Loss', fontsize=12) 
    axes[0].legend(title='Feedback Delay', fontsize=11)
    axes[0].grid(True, which='both', linestyle='--', alpha=0.7)
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Linear scale with scientific notation

    # Plot 2: Effect of Pack Size on Cumulative Square Loss (Middle)
    for i, (k, result_data) in enumerate(pack_results.items()):
        axes[1].plot(result_data['cumulative_loss'], label=f'Pack Size = {k}', 
                     color=pack_colors[i], linestyle=line_styles[i % len(line_styles)])
        
    axes[1].set_title('Effect of Pack Size on Cumulative Square Loss', fontsize=16)
    axes[1].set_xlabel('Time Steps', fontsize=12)
    axes[1].set_ylabel('Cumulative Square Loss', fontsize=12) 
    axes[1].legend(title='Pack Size (K)', fontsize=11)
    axes[1].grid(True, which='both', linestyle='--', alpha=0.7)
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # scale with scientific notation

    # Plot 3: Comparison: Continuous Small Delay vs. Single Large Delay
    # Using distinct colors for this comparison
    axes[2].plot(small_continuous_delay_result['cumulative_loss'], 
                 label=f'Continuous Delay = 1 step', 
                 color='orange', linestyle='-') # Use 'orange' and 'darkviolet' for clear contrast
    axes[2].plot(large_single_delay_result['cumulative_loss'], 
                 label=f'Single Large Delay = {N_SAMPLES - 1} steps', 
                 color='darkviolet', linestyle='--') 
        
    axes[2].set_title(f'Comparison: Continuous Small Delay vs. Single Large Delay', fontsize=16)
    axes[2].set_xlabel('Time Steps', fontsize=12)
    axes[2].set_ylabel('Cumulative Square Loss (Log Scale)', fontsize=12) # Log scale label
    axes[2].legend(title='Delay Scenario', fontsize=11)
    axes[2].grid(True, which='both', linestyle='--', alpha=0.7)
    axes[2].set_yscale('log') # Keep logarithmic scale

    plt.tight_layout() # Adjusts plot parameters for a tight layout.
    plt.show()

    # --- Print summary tables ---
    print("\n--- Delayed Feedback Simulation Summary ---")
    delay_summary_data = []
    for delay, result_data in delay_results.items():
        delay_summary_data.append({
            'Delay': delay,
            'Final Total Loss': result_data['cumulative_loss'][-1],
            'RMSE': result_data['performance']['RMSE'],
            'R2': result_data['performance']['R2']
        })
    delay_summary_df = pd.DataFrame(delay_summary_data).set_index('Delay')
    print(delay_summary_df.to_string(float_format="%.4e")) # Use scientific notation

    pack_summary_data = []
    for k, result_data in pack_results.items():
        pack_summary_data.append({
            'Pack Size': k,
            'Final Total Loss': result_data['cumulative_loss'][-1],
            'RMSE': result_data['performance']['RMSE'],
            'R2': result_data['performance']['R2']
        })
    pack_summary_df = pd.DataFrame(pack_summary_data).set_index('Pack Size')
    print(pack_summary_df.to_string(float_format="%.4e")) # Use scientific notation
    
    print("\n--- Specific Delay Comparison Summary (for 3rd plot) ---")
    specific_delay_summary_data = [
        {'Scenario': 'Continuous Delay = 1 step',
         'Final Total Loss': small_continuous_delay_result['cumulative_loss'][-1],
         'RMSE': small_continuous_delay_result['performance']['RMSE'],
         'R2': small_continuous_delay_result['performance']['R2']},
        {'Scenario': f'Single Large Delay = {N_SAMPLES - 1} steps',
         'Final Total Loss': large_single_delay_result['cumulative_loss'][-1],
         'RMSE': large_single_delay_result['performance']['RMSE'],
         'R2': large_single_delay_result['performance']['R2']}
    ]
    specific_delay_summary_df = pd.DataFrame(specific_delay_summary_data).set_index('Scenario')
    print(specific_delay_summary_df.to_string(float_format="%.4e")) # Use scientific notation