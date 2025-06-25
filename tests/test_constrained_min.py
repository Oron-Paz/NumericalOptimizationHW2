import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from constrained_min import interior_pt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from examples import (
    qp_objective, qp_ineq_constraints, qp_eq_constraints_mat, qp_eq_constraints_rhs,
    lp_objective, lp_ineq_constraints
)


class TestConstrainedMin(unittest.TestCase):
    
    def test_qp(self):
        print("\n" + "="*50)
        print("QUADRATIC PROGRAMMING TEST")
        print("="*50)
        
        x0 = np.array([0.1, 0.2, 0.7])
        
        x_final, obj_vals, x_path, success = interior_pt(
            qp_objective, 
            qp_ineq_constraints, 
            qp_eq_constraints_mat, 
            qp_eq_constraints_rhs, 
            x0
        )
        
        print(f"Final solution: x = {x_final}")
        print(f"Objective value: {obj_vals[-1]:.6f}")
        print(f"Constraint x + y + z = 1: {np.sum(x_final):.6f}")
        print(f"All variables non-negative: {np.all(x_final >= -1e-6)}")
        print(f"Number of outer iterations: {len(obj_vals)}")
        
        self._plot_qp_results(x_path, obj_vals, x_final)
        
    def test_lp(self):
        print("\n" + "="*50)
        print("LINEAR PROGRAMMING TEST")
        print("="*50)
        
        x0 = np.array([0.5, 0.75])
        
        x_final, obj_vals, x_path, success = interior_pt(
            lp_objective,
            lp_ineq_constraints,
            None,
            None,
            x0
        )
        
        print(f"Final solution: x = {x_final}")
        print(f"Objective value (min -(x+y)): {obj_vals[-1]:.6f}")
        print(f"Maximized value (x+y): {-obj_vals[-1]:.6f}")
        print(f"Number of outer iterations: {len(obj_vals)}")
        
        print("\nConstraint values at final point:")
        constraint_names = ["y + x - 1 ≥ 0", "1 - y ≥ 0", "2 - x ≥ 0", "-y ≤ 0"]
        for i, constraint in enumerate(lp_ineq_constraints):
            g_val, _, _ = constraint(x_final, True)
            print(f"  {constraint_names[i]}: {-g_val:.6f}")
        
        self._plot_lp_results(x_path, obj_vals, x_final)
    
    def _plot_qp_results(self, x_path, obj_vals, x_final):
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        triangle = Poly3DCollection([vertices], alpha=0.3, facecolor='lightblue', edgecolor='blue')
        ax1.add_collection3d(triangle)
        
        path_array = np.array(x_path)
        ax1.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                'ro-', markersize=4, linewidth=2, label='Central Path')
        
        ax1.scatter(x_final[0], x_final[1], x_final[2], 
                   color='red', s=100, marker='*', label='Final Solution')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y') 
        ax1.set_zlabel('z')
        ax1.set_title('QP: Feasible Region and Central Path')
        ax1.legend()
        
        ax2 = fig.add_subplot(132)
        ax2.plot(range(len(obj_vals)), obj_vals, 'b-o', markersize=4)
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('QP: Objective Value vs Iteration')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_lp_results(self, x_path, obj_vals, x_final):
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(121)
        
        x_range = np.linspace(-0.5, 2.5, 100)
        
        y1 = -x_range + 1
        y2 = np.ones_like(x_range)
        y3 = np.zeros_like(x_range)
        
        x_fill = [1, 2, 2, 0, 1]
        y_fill = [0, 0, 1, 1, 0]
        ax1.fill(x_fill, y_fill, alpha=0.3, color='lightblue', label='Feasible Region')
        
        ax1.plot(x_range, y1, 'g--', label='y = -x + 1')
        ax1.axhline(y=1, color='orange', linestyle='--', label='y = 1')
        ax1.axvline(x=2, color='purple', linestyle='--', label='x = 2')
        ax1.axhline(y=0, color='brown', linestyle='--', label='y = 0')
        
        path_array = np.array(x_path)
        ax1.plot(path_array[:, 0], path_array[:, 1], 
                'ro-', markersize=4, linewidth=2, label='Central Path')
        
        for i, point in enumerate(path_array):
            ax1.annotate(f'{i}', (point[0], point[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax1.scatter(x_final[0], x_final[1], 
                   color='red', s=100, marker='*', label='Final Solution')
        
        x_contour = np.linspace(-0.5, 2.5, 50)
        y_contour = np.linspace(-0.5, 1.5, 50)
        X, Y = np.meshgrid(x_contour, y_contour)
        Z = X + Y
        contours = ax1.contour(X, Y, Z, levels=10, alpha=0.6, colors='gray', linestyles=':')
        ax1.clabel(contours, inline=True, fontsize=8)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('LP: Feasible Region and Central Path')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlim(-0.5, 2.5)
        ax1.set_ylim(-0.5, 1.5)
        
        ax2 = fig.add_subplot(122)
        ax2.plot(range(len(obj_vals)), obj_vals, 'b-o', markersize=4)
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Objective Value (min -(x+y))')
        ax2.set_title('LP: Objective Value vs Iteration')
        ax2.grid(True)
        
        # ax2_twin = ax2.twinx()
        # ax2_twin.plot(range(len(obj_vals)), [-val for val in obj_vals], 'r--', alpha=0.7)
        # ax2_twin.set_ylabel('Maximized Value (x+y)', color='r')
        # ax2_twin.tick_params(axis='y', labelcolor='r')
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main(verbosity=2)