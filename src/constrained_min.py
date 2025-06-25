import numpy as np
from scipy.linalg import solve
from scipy.linalg import LinAlgWarning
import warnings


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    t = 1.0
    mu = 10.0
    epsilon = 1e-6
    max_outer_iter = 50
    max_inner_iter = 100
    
    x = x0.copy()
    obj_vals = []
    x_path = []
    
    is_feasible, constraint_vals, min_slack = _check_strict_feasibility(x, ineq_constraints)
    if not is_feasible:
        raise ValueError(f"Initial point x0 is not strictly feasible. Min slack: {min_slack:.2e}")
    
    if eq_constraints_mat is not None and eq_constraints_rhs is not None:
        eq_violation = np.linalg.norm(eq_constraints_mat @ x - eq_constraints_rhs)
        if eq_violation > 1e-10:
            raise ValueError(f"Initial point x0 does not satisfy equality constraints. Violation: {eq_violation:.2e}")
    
    f_val, _, _ = func(x, True)
    obj_vals.append(f_val)
    x_path.append(x.copy())
    
    for outer_iter in range(max_outer_iter):
        m = len(ineq_constraints)
        if m / t < epsilon:
            break
            
        try:
            x = _solve_barrier_problem(x, t, func, ineq_constraints, 
                                     eq_constraints_mat, eq_constraints_rhs, 
                                     max_inner_iter)
            
            f_val, _, _ = func(x, True)
            obj_vals.append(f_val)
            x_path.append(x.copy())
            
        except Exception as e:
            warnings.warn(f"Inner optimization failed at outer iteration {outer_iter}: {str(e)}")
            break
            
        t *= mu
    
    success = outer_iter < max_outer_iter - 1
    return x, obj_vals, x_path, success


def _check_strict_feasibility(x, ineq_constraints, margin=1e-10):
    constraint_values = []
    min_slack = float('inf')
    
    for constraint in ineq_constraints:
        g_val, _, _ = constraint(x, True)
        constraint_values.append(g_val)
        slack = -g_val
        min_slack = min(min_slack, slack)
    
    is_feasible = min_slack > margin
    return is_feasible, constraint_values, min_slack


def _is_strictly_feasible(x, ineq_constraints):
    is_feasible, _, _ = _check_strict_feasibility(x, ineq_constraints)
    return is_feasible


def _find_max_feasible_step(x, p, ineq_constraints, safety_factor=0.95):
    max_alpha = 1.0
    
    for constraint in ineq_constraints:
        g_val, g_grad, _ = constraint(x, True)
        direction_derivative = np.dot(g_grad, p)
        
        if direction_derivative > 1e-14:
            alpha_to_boundary = -g_val / direction_derivative
            
            if alpha_to_boundary > 0:
                max_alpha = min(max_alpha, safety_factor * alpha_to_boundary)
    
    return max(max_alpha, 1e-12)


def _solve_barrier_problem(x0, t, func, ineq_constraints, eq_constraints_mat, 
                          eq_constraints_rhs, max_iter):
    x = x0.copy()
    tolerance = 1e-8
    
    for iteration in range(max_iter):
        f_val, f_grad, f_hess = func(x, True)
        
        phi_val = 0.0
        phi_grad = np.zeros_like(x)
        phi_hess = np.zeros((len(x), len(x)))
        
        for constraint in ineq_constraints:
            g_val, g_grad, g_hess = constraint(x, True)
            
            if g_val >= 0:
                raise ValueError("Point became infeasible during optimization")
            
            phi_val += -np.log(-g_val)
            phi_grad += g_grad / (-g_val)
            phi_hess += (np.outer(g_grad, g_grad) / (g_val**2)) - (g_hess / g_val)
        
        total_grad = t * f_grad + phi_grad
        total_hess = t * f_hess + phi_hess
        
        if eq_constraints_mat is not None and eq_constraints_rhs is not None:
            A = eq_constraints_mat
            n = len(x)
            p = A.shape[0]
            
            kkt_matrix = np.zeros((n + p, n + p))
            kkt_matrix[:n, :n] = total_hess
            kkt_matrix[:n, n:] = A.T
            kkt_matrix[n:, :n] = A
            
            rhs = np.zeros(n + p)
            rhs[:n] = -total_grad
            
            try:
                cond_num = np.linalg.cond(kkt_matrix)
                if cond_num > 1e12:
                    regularization = max(1e-12, 1e-8 / cond_num)
                    kkt_matrix[:n, :n] += regularization * np.eye(n)
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=LinAlgWarning)
                    solution = solve(kkt_matrix, rhs)
                p_k = solution[:n]
            except np.linalg.LinAlgError:
                regularization = 1e-6
                kkt_matrix[:n, :n] += regularization * np.eye(n)
                solution = solve(kkt_matrix, rhs)
                p_k = solution[:n]
                
        else:
            try:
                p_k = -solve(total_hess, total_grad)
            except np.linalg.LinAlgError:
                regularization = 1e-8
                regularized_hess = total_hess + regularization * np.eye(len(total_hess))
                p_k = -solve(regularized_hess, total_grad)
        
        alpha = _backtracking_line_search(x, p_k, t, func, ineq_constraints, 
                                        eq_constraints_mat, eq_constraints_rhs)
        
        x = x + alpha * p_k
        
        if np.linalg.norm(alpha * p_k) < tolerance:
            break
    
    return x


def _backtracking_line_search(x, p, t, func, ineq_constraints, 
                             eq_constraints_mat, eq_constraints_rhs):
    beta = 0.5
    gamma = 0.01
    
    current_obj = _barrier_objective(x, t, func, ineq_constraints)
    
    f_val, f_grad, f_hess = func(x, True)
    phi_grad = np.zeros_like(x)
    
    for constraint in ineq_constraints:
        g_val, g_grad, g_hess = constraint(x, True)
        phi_grad += g_grad / (-g_val)
    
    total_grad = t * f_grad + phi_grad
    grad_dot_p = np.dot(total_grad, p)
    
    alpha = _find_max_feasible_step(x, p, ineq_constraints, safety_factor=0.9)
    
    max_backtrack = 50
    for iteration in range(max_backtrack):
        x_new = x + alpha * p
        
        is_feasible, constraint_vals, min_slack = _check_strict_feasibility(x_new, ineq_constraints)
        
        if not is_feasible:
            alpha *= beta
            continue
            
        if eq_constraints_mat is not None and eq_constraints_rhs is not None:
            eq_violation = np.linalg.norm(eq_constraints_mat @ x_new - eq_constraints_rhs)
            if eq_violation > 1e-8:
                alpha *= beta
                continue
        
        try:
            new_obj = _barrier_objective(x_new, t, func, ineq_constraints)
            if new_obj <= current_obj + gamma * alpha * grad_dot_p:
                break
        except:
            alpha *= beta
            continue
            
        alpha *= beta
    
    return alpha


def _barrier_objective(x, t, func, ineq_constraints):
    f_val, _, _ = func(x, True)
    
    phi_val = 0.0
    for constraint in ineq_constraints:
        g_val, _, _ = constraint(x, True)
        if g_val >= 0:
            return float('inf')
        phi_val += -np.log(-g_val)
    
    return t * f_val + phi_val