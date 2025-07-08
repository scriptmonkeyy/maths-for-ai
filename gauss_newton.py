import numpy as np

def gauss_newton(F, J, x0, tol=1e-8, max_iter=100):
    """
    Solve F(x) = 0 via Gauss-Newton method by minimizing ||F(x)||^2.

    Parameters:
    - F: function F(x) returning m-dim vector
    - J: function J(x) returning Jacobian matrix (m x n)
    - x0: initial guess (n-dim vector)
    - tol: stopping tolerance on ||delta_x||
    - max_iter: max iterations allowed

    Returns:
    - x: solution vector
    """
    x = x0.copy()

    for i in range(max_iter):
        Fx = F(x)  # residual vector (m,)
        Jx = J(x)  # Jacobian matrix (m, n)

        # Solve normal equations: (J^T J) delta = J^T F
        JTJ = Jx.T @ Jx
        JTF = Jx.T @ Fx

        try:
            delta = np.linalg.solve(JTJ, JTF)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered at iteration", i)
            break

        x_new = x - delta

        if np.linalg.norm(delta) < tol:
            return x_new

        x = x_new

    print("Gauss-Newton did not converge")
    return x
