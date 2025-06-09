import unittest
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

from src.unconstrained_min import minimizationAlgorithms
from tests.examples import (quadratic_example_1, quadratic_example_2, quadratic_example_3, rosenbrock, my_own_linear_example, last_linear_example)
from src.utils import plot_contours_with_path, plot_function_values


class TestMinimization(unittest.TestCase):
    def setUp(self):
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 100
        self.x0_standard = [1.0, 1.0]  # starting values
        self.x0_rosenbrock = [-1.0, 2.0]  # only for rosebnrock
    
    def test_quadratic_example_1(self):
        print("\n" + "="*60)
        print("Testing Quadratic Example 1 - Both Methods")
        print("="*60)
        
        gd_optimizer = minimizationAlgorithms('Gradient Descent')
        gd_final_x, gd_final_f, gd_success = gd_optimizer.minimize(
            quadratic_example_1, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            self.max_iter
        )
        
        print(f"\nGRADIENT DESCENT FINAL RESULT:")
        print(f"Final location: {gd_final_x}")
        print(f"Final objective value: {gd_final_f}")
        print(f"Iterations: {gd_optimizer.history[-1]["iteration"]}")
        print(f"Success: {gd_success}")
        
        newton_optimizer = minimizationAlgorithms('Newton Search Directions')
        newton_final_x, newton_final_f, newton_success = newton_optimizer.minimize(
            quadratic_example_1, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            self.max_iter
        )
        
        print(f"\nNEWTON'S METHOD FINAL RESULT:")
        print(f"Final location: {newton_final_x}")
        print(f"Final objective value: {newton_final_f}")
        print(f"Iterations: {newton_optimizer.history[-1]["iteration"]}")
        print(f"Success: {newton_success}")
        
        # create comparison plots
        plot_contours_with_path(
            quadratic_example_1, 
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Quadratic Example 1 (Circles) - Method Comparison",
            x_limits=(-2, 2),
            y_limits=(-2, 2)
        )
        
        plot_function_values(
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Quadratic Example 1 (Circles) - Function Values Comparison"
        )
        
        self.assertTrue(gd_success)
        self.assertTrue(newton_success)
    
    def test_quadratic_example_2(self):
        """Test both methods on quadratic example 2 (axis-aligned ellipses)"""
        print("\n" + "="*60)
        print("Testing Quadratic Example 2 - Both Methods")
        print("="*60)
        
        gd_optimizer = minimizationAlgorithms('Gradient Descent')
        gd_final_x, gd_final_f, gd_success = gd_optimizer.minimize(
            quadratic_example_2, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            self.max_iter
        )
        
        print(f"\nGRADIENT DESCENT FINAL RESULT:")
        print(f"Final location: {gd_final_x}")
        print(f"Final objective value: {gd_final_f}")
        print(f"Iterations: {gd_optimizer.history[-1]["iteration"]}")
        print(f"Success: {gd_success}")
        
        newton_optimizer = minimizationAlgorithms('Newton Search Directions')
        newton_final_x, newton_final_f, newton_success = newton_optimizer.minimize(
            quadratic_example_2, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            self.max_iter
        )
        
        print(f"\nNEWTON'S METHOD FINAL RESULT:")
        print(f"Final location: {newton_final_x}")
        print(f"Final objective value: {newton_final_f}")
        print(f"Iterations: {newton_optimizer.history[-1]["iteration"]}")
        print(f"Success: {newton_success}")
        
        # create comparison plots
        plot_contours_with_path(
            quadratic_example_2, 
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Quadratic Example 2 - Method Comparison",
            x_limits=(-2, 2),
            y_limits=(-0.5, 0.5)
        )
        
        plot_function_values(
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Quadratic Example 2 - Function Values Comparison"
        )
        
        self.assertTrue(gd_success)
        self.assertTrue(newton_success)
    
    def test_quadratic_example_3(self):
        """Test both methods on quadratic example 3 (rotated ellipses)"""
        print("\n" + "="*60)
        print("Testing Quadratic Example 3 - Both Methods")
        print("="*60)
        
        gd_optimizer = minimizationAlgorithms('Gradient Descent')
        gd_final_x, gd_final_f, gd_success = gd_optimizer.minimize(
            quadratic_example_3, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            self.max_iter
        )
        
        print(f"\nGRADIENT DESCENT FINAL RESULT:")
        print(f"Final location: {gd_final_x}")
        print(f"Final objective value: {gd_final_f}")
        print(f"Iterations: {gd_optimizer.history[-1]["iteration"]}")
        print(f"Success: {gd_success}")
        
        newton_optimizer = minimizationAlgorithms('Newton Search Directions')
        newton_final_x, newton_final_f, newton_success = newton_optimizer.minimize(
            quadratic_example_3, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            self.max_iter
        )
        
        print(f"\nNEWTON'S METHOD FINAL RESULT:")
        print(f"Final location: {newton_final_x}")
        print(f"Final objective value: {newton_final_f}")
        print(f"Iterations: {newton_optimizer.history[-1]["iteration"]}")
        print(f"Success: {newton_success}")
        
        # create comparison plots
        plot_contours_with_path(
            quadratic_example_3, 
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Quadratic Example 3 - Method Comparison",
            x_limits=(-2, 2),
            y_limits=(-2, 2)
        )
        
        plot_function_values(
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Quadratic Example 3 - Function Values Comparison"
        )
        
        self.assertTrue(gd_success)
        self.assertTrue(newton_success)
    
    def test_rosenbrock(self):
        """Test both methods on Rosenbrock function"""
        print("\n" + "="*60)
        print("Testing Rosenbrock Function - Both Methods")
        print("="*60)
        
        gd_optimizer = minimizationAlgorithms('Gradient Descent')
        gd_final_x, gd_final_f, gd_success = gd_optimizer.minimize(
            rosenbrock, 
            self.x0_rosenbrock.copy(), 
            self.obj_tol, 
            self.param_tol, 
            10000
        )
        
        print(f"\nGRADIENT DESCENT FINAL RESULT:")
        print(f"Final location: {gd_final_x}")
        print(f"Final objective value: {gd_final_f}")
        print(f"Iterations: {gd_optimizer.history[-1]["iteration"]}")
        print(f"Success: {gd_success}")
        
        newton_optimizer = minimizationAlgorithms('Newton Search Directions')
        newton_final_x, newton_final_f, newton_success = newton_optimizer.minimize(
            rosenbrock, 
            self.x0_rosenbrock.copy(), 
            self.obj_tol, 
            self.param_tol, 
            self.max_iter
        )
        
        print(f"\nNEWTON'S METHOD FINAL RESULT:")
        print(f"Final location: {newton_final_x}")
        print(f"Final objective value: {newton_final_f}")
        print(f"Iterations: {newton_optimizer.history[-1]["iteration"]}")
        print(f"Success: {newton_success}")
        
        plot_contours_with_path(
            rosenbrock, 
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Rosenbrock Function - Method Comparison",
            x_limits=(-2, 2),
            y_limits=(-1, 3)
        )
        
        plot_function_values(
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Rosenbrock Function - Function Values Comparison"
        )
    
    def test_linear_example(self):
        """Test both methods on linear function"""
        print("\n" + "="*60)
        print("Testing Linear Function - Both Methods")
        print("="*60)
        
        small_max_iter = 100
        
        gd_optimizer = minimizationAlgorithms('Gradient Descent')
        gd_final_x, gd_final_f, gd_success = gd_optimizer.minimize(
            my_own_linear_example, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            small_max_iter
        )
        
        print(f"\nGRADIENT DESCENT FINAL RESULT:")
        print(f"Final location: {gd_final_x}")
        print(f"Final objective value: {gd_final_f}")
        print(f"Iterations: {gd_optimizer.history[-1]["iteration"]}")
        print(f"Success: {gd_success}")
        
        newton_optimizer = minimizationAlgorithms('Newton Search Directions')
        newton_final_x, newton_final_f, newton_success = newton_optimizer.minimize(
            my_own_linear_example, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            small_max_iter
        )
        
        print(f"\nNEWTON'S METHOD FINAL RESULT:")
        print(f"Final location: {newton_final_x}")
        print(f"Final objective value: {newton_final_f}")
        print(f"Iterations: {newton_optimizer.history[-1]["iteration"]}")
        print(f"Success: {newton_success}")
        
        plot_contours_with_path(
            my_own_linear_example, 
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Linear Function - Method Comparison",
            x_limits=(-5, 5),
            y_limits=(-5, 5)
        )
        
        plot_function_values(
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Linear Function - Function Values Comparison"
        )
    
    def test_exponential_example(self):
        """Test both methods on exponential function"""
        print("\n" + "="*60)
        print("Testing Exponential Function (Smoothed Triangles) - Both Methods")
        print("="*60)
        
        gd_optimizer = minimizationAlgorithms('Gradient Descent')
        gd_final_x, gd_final_f, gd_success = gd_optimizer.minimize(
            last_linear_example, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            self.max_iter
        )
        
        print(f"\nGRADIENT DESCENT FINAL RESULT:")
        print(f"Final location: {gd_final_x}")
        print(f"Final objective value: {gd_final_f}")
        print(f"Iterations: {gd_optimizer.history[-1]["iteration"]}")
        print(f"Success: {gd_success}")
        
        newton_optimizer = minimizationAlgorithms('Newton Search Directions')
        newton_final_x, newton_final_f, newton_success = newton_optimizer.minimize(
            last_linear_example, 
            self.x0_standard.copy(), 
            self.obj_tol, 
            self.param_tol, 
            self.max_iter
        )
        
        print(f"\nNEWTON'S METHOD FINAL RESULT:")
        print(f"Final location: {newton_final_x}")
        print(f"Final objective value: {newton_final_f}")
        print(f"Iterations: {newton_optimizer.history[-1]["iteration"]}")
        print(f"Success: {newton_success}")
        
        plot_contours_with_path(
            last_linear_example, 
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Exponential Function (Smoothed Triangles) - Method Comparison",
            x_limits=(-1, 1),
            y_limits=(-1, 1)
        )
        
        plot_function_values(
            [gd_optimizer.history, newton_optimizer.history], 
            ["Gradient Descent", "Newton Method"], 
            "Exponential Function (Smoothed Triangles) - Function Values Comparison"
        )
        
        self.assertTrue(gd_success)
        self.assertTrue(newton_success)

if __name__ == '__main__':
    unittest.main()