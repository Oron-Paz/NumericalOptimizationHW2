import math

def quadratic_example_1(x, hessian_needed=False):
    # [x1,x2] [1,0] [x1]
    #         [0,1] [x2]

    f = x[0]**2 + x[1]**2
    g = [2*x[0], 2*x[1]]

    if hessian_needed:
        h = [[2, 0], [0, 2]]  
    else:
        h = None

    return f, g, h

def quadratic_example_2(x, hessian_needed=False):
    # [x1,x2] [1,  0] [x1]
    #         [0,100] [x2]

    f = x[0]**2 + 100*x[1]**2
    g = [2*x[0], 200*x[1]]

    if hessian_needed:
        h = [[2, 0], [0, 200]]  
    else:
        h = None

    return f, g, h

def quadratic_example_3(x, hessian_needed=False):
    # im not even going to try to write this one out as a comment
    f = (301*x[0]**2 - 198 * math.sqrt(3)*x[0]*x[1] + 103*x[1]**2) / 4
    g = [(301 * x[0] -99 * math.sqrt(3)*x[1]) / 2, (103*x[1]-99*math.sqrt(3)*x[0]) / 2]
    
    if hessian_needed:
        h = [[301/2, -99*math.sqrt(3)/2], [-99*math.sqrt(3)/2, 103/2]]  
    else:
        h = None

    return f, g, h

def rosenbrock(x, hessian_needed=False):
    f = 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    g = [-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]) , 200*(x[1]-x[0]**2)]
    if hessian_needed:
        h = [[2*(600*x[0]**2 - 200*x[1] + 1), -400*x[0]], [-400*x[0], 200]]  
    else:
        h = None

    return f, g, h
def my_own_linear_example(x, hessian_needed=False):
    a = [3, -2]
    f = a[0] * x[0] + a[1] * x[1]
    g = [a[0], a[1]]  
    if hessian_needed:
        h = [[0, 0], [0, 0]]
    else:
        h = None
    
    return f, g, h

def last_linear_example(x, hessian_needed=False):
    x1, x2 = x[0], x[1]
    
    term1 = math.exp(x1 + 3*x2 - 0.1)
    term2 = math.exp(x1 - 3*x2 - 0.1)
    term3 = math.exp(-x1 - 0.1)

    f = term1 + term2 + term3
    df_dx1 = term1 + term2 - term3
    df_dx2 = 3*term1 - 3*term2
    g = [df_dx1, df_dx2]
    
    if hessian_needed:
        h11 = term1 + term2 + term3
        h12 = 3*term1 - 3*term2
        h22 = 9*term1 + 9*term2
        h = [[h11, h12], [h12, h22]]  
    else:
        h = None
    
    return f, g, h