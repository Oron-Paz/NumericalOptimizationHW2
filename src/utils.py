"""
utils.py - Utility functions for plotting and visualization
"""

def plot_contours_with_path(objective_func, histories, method_names, title, x_limits=(-2, 2), y_limits=(-2, 2)):
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not isinstance(x_limits, (tuple, list)) or len(x_limits) != 2:
        x_limits = (-2, 2)
    if not isinstance(y_limits, (tuple, list)) or len(y_limits) != 2:
        y_limits = (-2, 2)
    
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = [X[i, j], Y[i, j]]
            f_val, _, _ = objective_func(point, hessian_needed=False)
            Z[i, j] = f_val
    
    plt.figure(figsize=(10, 8))
    
    contour_levels = np.logspace(-3, 2, 20)  
    plt.contour(X, Y, Z, levels=contour_levels, colors='gray', alpha=0.6)
    plt.contourf(X, Y, Z, levels=contour_levels, alpha=0.3, cmap='viridis')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (history, method_name) in enumerate(zip(histories, method_names)):
        if history:
            x_path = [point["location"][0] for point in history]
            y_path = [point["location"][1] for point in history]
            
            plt.plot(x_path, y_path, color=colors[i % len(colors)], 
                    marker='o', markersize=4, linewidth=2, label=method_name)
            
            plt.plot(x_path[0], y_path[0], color=colors[i % len(colors)], 
                    marker='s', markersize=8, label=f'{method_name} Start')
            plt.plot(x_path[-1], y_path[-1], color=colors[i % len(colors)], 
                    marker='*', markersize=10, label=f'{method_name} End')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    
    plt.tight_layout()
    plt.show()
        

def plot_function_values(histories, method_names, title):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (history, method_name) in enumerate(zip(histories, method_names)):
        if history:
            iterations = [point["iteration"] for point in history]
            function_values = [point["objective_value"] for point in history]
            
            plt.semilogy(iterations, function_values, color=colors[i % len(colors)], 
                        marker='o', linewidth=2, label=method_name)
    
    plt.xlabel('Iteration Number')
    plt.ylabel('Function Value (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
        

def print_algorithm_summary(history, method_name):
    if not history:
        print(f"No history available for {method_name}")
        return
    
    print(f"\n{method_name} Summary:")
    print("=" * 40)
    print(f"Total iterations: {len(history)}")
    print(f"Initial location: {history[0]['location']}")
    print(f"Final location: {history[-1]['location']}")
    print(f"Initial function value: {history[0]['objective_value']:.6e}")
    print(f"Final function value: {history[-1]['objective_value']:.6e}")
    
    if history[0]['objective_value'] != 0:
        reduction = abs(history[0]['objective_value']) / abs(history[-1]['objective_value'])
        print(f"Function value reduction: {reduction:.2e}")
