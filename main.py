import numpy as np
import json

def simulate_short_rate(total_time, initial_rate, volatility, num_simulations, time_step=0.01):
    """
    Simulates the evolution of the short rate r(t).
    """
    num_simulations = int(num_simulations)
    num_steps = int(round(total_time / time_step))
    time_grid = np.linspace(0, total_time, num_steps + 1)
    rate_paths = np.zeros((num_simulations, num_steps + 1))
    rate_paths[:, 0] = initial_rate

    for step in range(1, num_steps + 1):
        brownian_motion = np.random.normal(0, np.sqrt(time_step), num_simulations)
        rate_paths[:, step] = rate_paths[:, step-1] + volatility * brownian_motion
    
    return rate_paths, time_grid

def compute_discount_factors(rate_paths, time_grid):
    """
    Calculates the evolution of the discount factor P(0,t) for each trajectory.
    """
    discount_factors = np.zeros_like(rate_paths)
    discount_factors[:, 0] = 1.0
    for step in range(1, len(time_grid)):
        integral = np.trapz(rate_paths[:, :step+1], time_grid[:step+1], axis=1)
        discount_factors[:, step] = np.exp(-integral)
    return discount_factors

def monte_carlo_bond_price(total_time, initial_rate, volatility, num_simulations, time_step=0.01):
    """
    Performs a Monte Carlo simulation to estimate bond price statistics.
    """
    rate_paths, time_grid = simulate_short_rate(total_time, initial_rate, volatility, num_simulations, time_step)
    discount_factors = compute_discount_factors(rate_paths, time_grid)
    final_discount_factors = discount_factors[:, -1]
    estimated_price = np.mean(final_discount_factors)
    variance = np.var(final_discount_factors)
    
    return round(float(estimated_price), 10), round(float(variance), 10)

def run(input_data, solver_params=None, extra_arguments=None):
    """
    Runs the bond pricing simulation.
    """
    initial_rate = input_data["Initial Interest Rate"]
    volatility = input_data["Volatility"]
    bond_maturity = input_data["Maturity Time"] / 12  # Convert months to years

    # Retrieve number of simulations from solver_params
    num_simulations = solver_params.get("NumberOfSimulations", 10000)

    # Compute bond price statistics
    estimated_price, variance = monte_carlo_bond_price(bond_maturity, initial_rate, volatility, num_simulations)

    return {
        "bond_price": estimated_price,
        "variance": variance
    }