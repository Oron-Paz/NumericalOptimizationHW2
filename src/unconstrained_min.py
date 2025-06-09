class minimizationAlgorithms:
    def __init__(self, algorithm):
        self.algorithm = algorithm # either gradient descent OR newton search directions
        self.history = []
        
    def minimize(self, f, x0, obj_tol, param_tol, max_iter):
        self.history = []
        
        if self.algorithm == 'Gradient Descent':
            return self.gradient_descent(f, x0, obj_tol, param_tol, max_iter)
        elif self.algorithm == 'Newton Search Directions':
            return self.newton_search_directions(f, x0, obj_tol, param_tol, max_iter)
        else:
            print("Invalid Selection")
            return None, None, False

    
    def gradient_descent(self, f, x0, obj_tol, param_tol, max_iter):
        x = x0.copy()
        
        #intial value
        f_val, grad, _ = f(x, hessian_needed=False)
        
        self.history.append({
            "iteration": 0,
            "location": x.copy(),
            "objective_value": f_val
        })
        
        print(f"Iteration: 0")
        print(f"Current location: {x}")
        print(f"Current Objective Value: {f_val}")
        print("-" * 40)
        
        for iteration in range(1, max_iter + 1):
            f_val, grad, _ = f(x, hessian_needed=False)
            
            step_size = self.backtracking_line_search(f, x, grad)
            
            x_new = [x[i] - step_size * grad[i] for i in range(len(x))]

            f_val_new, _, _ = f(x_new, hessian_needed=False)
            
            print(f"Iteration: {iteration}")
            print(f"Current location: {x_new}")
            print(f"Current Objective Value: {f_val_new}")
            print("-" * 40)
            
            self.history.append({
                "iteration": iteration,
                "location": x_new.copy(),
                "objective_value": f_val_new
            })
            
            obj_change = abs(f_val_new - f_val)
            if obj_change < obj_tol:
                print(f"Converged: Objective change {obj_change} < {obj_tol}")
                return x_new, f_val_new, True
            
            
            diff = [x_new[i] - x[i] for i in range(len(x))]
            param_change = sum(d**2 for d in diff)**0.5  
            if param_change < param_tol:
                print(f"Converged: Parameter change {param_change} < {param_tol}")
                return x_new, f_val_new, True
            
            x = x_new
            f_val = f_val_new
        
        return x, f_val, False 

    def backtracking_line_search(self, f, x, grad, alpha=1.0, rho=0.5, c1=0.01):
        f_val, _, _ = f(x, hessian_needed=False)
        pk = [-g for g in grad]
        grad_dot_pk = sum(grad[i] * pk[i] for i in range(len(grad)))
        
        while True:
            x_new = [x[i] + alpha * pk[i] for i in range(len(x))]
            f_new, _, _ = f(x_new, hessian_needed=False)
            
            if f_new <= f_val + c1 * alpha * grad_dot_pk:
                return alpha
            
            alpha *= rho
            
            if alpha < 1e-10:
                return alpha

    def newton_search_directions(self, f, x0, obj_tol, param_tol, max_iter):
        x = x0.copy()
        x = x0.copy()
        
        f_val, grad, _ = f(x, hessian_needed=False)
        
        try:
            _, _, initial_hess = f(x, hessian_needed=True)
            B_k = [row[:] for row in initial_hess] 
        except:
            B_k = [[1.0, 0.0], [0.0, 1.0]]
        
        self.history.append({
            "iteration": 0,
            "location": x.copy(),
            "objective_value": f_val
        })
        
        print(f"Iteration: 0")
        print(f"Current location: {x}")
        print(f"Current Objective Value: {f_val}")
        print("-" * 40)
        
        for iteration in range(1, max_iter + 1):
            f_val_old = f_val
            grad_old = grad[:] 
            f_val, grad, _ = f(x, hessian_needed=False)
            
            
            newton_direction = self.solve_2x2_system(B_k, [-g for g in grad])
            
            grad_dot_direction = sum(grad[i] * newton_direction[i] for i in range(len(grad)))
            if grad_dot_direction > 0:
                print(f"Warning: Newton direction is not descent, using steepest descent")
                newton_direction = [-g for g in grad]
            
            step_size = self.backtracking_line_search_newton(f, x, newton_direction, grad)
            
            x_new = [x[i] + step_size * newton_direction[i] for i in range(len(x))]
            
            f_val_new, grad_new, _ = f(x_new, hessian_needed=False)
            
            print(f"Iteration: {iteration}")
            print(f"Current location: {x_new}")
            print(f"Current Objective Value: {f_val_new}")
            print("-" * 40)
            
            self.history.append({
                "iteration": iteration,
                "location": x_new.copy(),
                "objective_value": f_val_new
            })
            
            
            obj_change = abs(f_val_new - f_val)
            if obj_change < obj_tol:
                print(f"Converged: Objective change {obj_change} < {obj_tol}")
                return x_new, f_val_new, True
            
            diff = [x_new[i] - x[i] for i in range(len(x))]
            param_change = sum(d**2 for d in diff)**0.5  # L2 norm
            if param_change < param_tol:
                print(f"Converged: Parameter change {param_change} < {param_tol}")
                return x_new, f_val_new, True
            
            newton_decrement_sq = -sum(grad[i] * newton_direction[i] for i in range(len(grad)))
            if newton_decrement_sq < obj_tol:
                print(f"Converged: Newton decrement {newton_decrement_sq} < {obj_tol}")
                return x_new, f_val_new, True

            if iteration < max_iter:
                try:
                    _, _, exact_hess = f(x_new, hessian_needed=True)
                    B_k = [row[:] for row in exact_hess]  
                except:
                    pass
            
            x = x_new
            f_val = f_val_new
            grad = grad_new[:]
        
        print(f"Failed: Maximum iterations ({max_iter}) reached")
        return x, f_val, False
       
    
    def backtracking_line_search_newton(self, f, x, newton_direction, grad, alpha=1.0, rho=0.5, c1=0.01):
        f_val, _, _ = f(x, hessian_needed=False)
        grad_dot_pk = sum(grad[i] * newton_direction[i] for i in range(len(grad)))
        
        while True:
            x_new = [x[i] + alpha * newton_direction[i] for i in range(len(x))]
            f_new, _, _ = f(x_new, hessian_needed=False)
            
            if f_new <= f_val + c1 * alpha * grad_dot_pk:
                return alpha
            
            alpha *= rho
            
            if alpha < 1e-10:
                return alpha

    def solve_2x2_system(self, A, b):
        a11, a12 = A[0][0], A[0][1]
        a21, a22 = A[1][0], A[1][1]
        b1, b2 = b[0], b[1]

        
        det = a11 * a22 - a12 * a21

        if abs(det) < 1e-10:
            print("Warning: Hessian is singular, using steepest descent direction")
            return [-b1, -b2]  # Return negative gradient
        
        x1 = (b1 * a22 - b2 * a12) / det
        x2 = (a11 * b2 - a21 * b1) / det
        
        return [x1, x2]