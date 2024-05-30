import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import Callable


DOMAIN_A: float = 0.
DOMAIN_B: float = 5.


def get_time(method: Callable):
    def wrapper(*args, **kwargs):

        print(f"\n================ START {method.__name__} ================\n")
        begin = time()

        returned_value = method(*args, **kwargs)

        end = time()
        print(f"\n================= END {method.__name__} =================\n")

        print(f"\n{method.__name__} took {end-begin:0.2f} seconds \n")
        return returned_value

    return wrapper


def generate_matrix(*args, **kwargs):
    NotImplementedError("COCN!")
    
def generate_rhs(*args, **kwargs):
    NotImplementedError("COCN!")
    
    
def conjugate_gradient(A, b, tol=1e-6, max_iter=1000):
    x = np.zeros_like(b)
    r = b - A @ x
    p = r
    r_dot_old = r @ r
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = r_dot_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_dot_new = r @ r
        
        if np.sqrt(r_dot_new) < tol:
            break
        
        beta = r_dot_new / r_dot_old
        p = r + beta * p
        r_dot_old = r_dot_new
    
    return x

# Example usage:
# A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
# b = np.array([1, 4, 2])
# x0 = np.zeros_like(b)
# x = conjugate_gradient(A, b, x0)
# print("Solution x:", x)


class Poisson:
    def __init__(self,
                 f: Callable[[np.ndarray], np.ndarray],
                 A: float,
                 B: float,
                 h: float,
                 max_iter: int = 10000):
        self.matrix = A
        self._B = B
        self._max_iter = max_iter
        self._f = f
        self._h = h

        # We'll define Poisson equation on interval (0, 5)
        self._a = DOMAIN_A
        self._b = DOMAIN_B
        
        self._n = int((self._b - self._a) / h)
        # self._poisson_mat = np.zeros((self._n, self._n))
        self._poisson_mat = generate_matrix(self._n - 2)
        # self._mat_rhs = np.zeros((self._n,))
        self._mat_rhs = generate_rhs(self._f, self._h, self.matrix, self._B, self._n - 2)

        self.u = np.zeros(self._n)
        self.u[0] = self.matrix
        self.u[-1] = self._B
        

    def solve_poisson(self, 
                      method:str = "Jakobi",
                      verbose: bool = False):
        '''
        method: "Jakobi", "GS", "numpy".
        verbose = False(def): print ||Ax-b||_max error for every 100 steps.
        '''
        if method == "Jakobi":
            self.u[1:-1] = conjugate_gradient(self.matrix, self._mat_rhs, self._max_iter, verbose)
        elif method == "numpy":
            begin = time()
            self.u[1:-1] = np.linalg.solve(self.matrix, self._mat_rhs)
            end = time()
            print(f"\nnumpy solver took {end-begin:0.2f} seconds \n")
            print(f"error ||Ax - b||_max = {np.linalg.norm(self._poisson_mat @ self.u[1:-1] - self._mat_rhs, np.inf):e}")
        else:
            raise AttributeError(f"Wrong method {method}")
        
        return self.u


