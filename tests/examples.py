import numpy as np


def qp_objective(x, compute_hessian=True):
    x_var, y_var, z_var = x[0], x[1], x[2]
    
    f_val = x_var**2 + y_var**2 + (z_var + 1)**2
    grad = np.array([2*x_var, 2*y_var, 2*(z_var + 1)])
    hess = np.array([[2.0, 0.0, 0.0],
                     [0.0, 2.0, 0.0],
                     [0.0, 0.0, 2.0]])
    
    return f_val, grad, hess


def qp_ineq_constraint_x(x, compute_hessian=True):
    g_val = -x[0]
    grad = np.array([-1.0, 0.0, 0.0])
    hess = np.zeros((3, 3))
    return g_val, grad, hess


def qp_ineq_constraint_y(x, compute_hessian=True):
    g_val = -x[1]
    grad = np.array([0.0, -1.0, 0.0])
    hess = np.zeros((3, 3))
    return g_val, grad, hess


def qp_ineq_constraint_z(x, compute_hessian=True):
    g_val = -x[2]
    grad = np.array([0.0, 0.0, -1.0])
    hess = np.zeros((3, 3))
    return g_val, grad, hess


qp_ineq_constraints = [
    qp_ineq_constraint_x,
    qp_ineq_constraint_y,
    qp_ineq_constraint_z
]

qp_eq_constraints_mat = np.array([[1.0, 1.0, 1.0]])
qp_eq_constraints_rhs = np.array([1.0])


def lp_objective(x, compute_hessian=True):
    x_var, y_var = x[0], x[1]
    
    f_val = -(x_var + y_var)
    grad = np.array([-1.0, -1.0])
    hess = np.zeros((2, 2))
    
    return f_val, grad, hess


def lp_ineq_constraint_1(x, compute_hessian=True):
    x_var, y_var = x[0], x[1]
    g_val = 1.0 - y_var - x_var
    grad = np.array([-1.0, -1.0])
    hess = np.zeros((2, 2))
    return g_val, grad, hess


def lp_ineq_constraint_2(x, compute_hessian=True):
    y_var = x[1]
    g_val = y_var - 1.0
    grad = np.array([0.0, 1.0])
    hess = np.zeros((2, 2))
    return g_val, grad, hess


def lp_ineq_constraint_3(x, compute_hessian=True):
    x_var = x[0]
    g_val = x_var - 2.0
    grad = np.array([1.0, 0.0])
    hess = np.zeros((2, 2))
    return g_val, grad, hess


def lp_ineq_constraint_4(x, compute_hessian=True):
    y_var = x[1]
    g_val = -y_var
    grad = np.array([0.0, -1.0])
    hess = np.zeros((2, 2))
    return g_val, grad, hess


lp_ineq_constraints = [
    lp_ineq_constraint_1,
    lp_ineq_constraint_2,
    lp_ineq_constraint_3,
    lp_ineq_constraint_4
]