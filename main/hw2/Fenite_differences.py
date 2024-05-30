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


@get_time
def J_solver(A: np.ndarray,
             b: np.ndarray,
             max_iter: int = 10000,
             verbose=False) -> np.ndarray:

    x = np.zeros_like(b)
    D: np.ndarray = np.diag(A) # is vector
    L: np.ndarray = np.tril(A, k=-1)
    U: np.ndarray = np.triu(A, k=1)
    inv_d = 1 / D
    for iter_count in range(1, max_iter + 1):
        x = inv_d * (b - (L + U) @ x)

        if np.linalg.norm(A @ x - b) < 10**(-12):
            break

        if verbose and iter_count % 100 == 0:
            print(f"Iteration {iter_count}: error ||Ax - b||_max = {np.linalg.norm(A @ x - b, np.inf):e}\n")
    
    return x


@get_time
def GS_solver(A: np.ndarray,
              b: np.ndarray,
              max_iter: int = 10000,
              verbose=False) -> np.ndarray:
    x = np.zeros_like(b)
    for iter_count in range(1, max_iter + 1):
        x_new = np.zeros_like(b)
        for i in range(A.shape[0]):
            s1 = A[i, :i] @ x_new[:i]
            s2 = A[i, i+1:] @ x[i+1:]
            x_new[i] = (b[i] - s1 - s2)/ A[i, i]

        if np.linalg.norm(A @ x - b) < 10**(-12):
            break

        x = x_new
        if verbose and iter_count % 100 == 0:
            print(f"Iteration {iter_count}: error ||Ax - b||_max = {np.linalg.norm(A @ x - b, np.inf):e}\n")

    return x
    

def generate_poisson_matrix(n: int) -> np.ndarray:
    return (-np.eye(n, k=1) + 2*np.eye(n) - np.eye(n, k=1).T)


def generate_rhs(f: Callable[[np.ndarray], np.ndarray],
                 h: float, 
                 a: float, 
                 b: float, 
                 n: int) -> np.ndarray:
    rhs = h**2 * f(np.linspace(DOMAIN_A, DOMAIN_B, n)) 
    rhs[0] += a
    rhs[-1] += b
    return rhs


class Poisson:
    def __init__(self,
                 f: Callable[[np.ndarray], np.ndarray],
                 A: float,
                 B: float,
                 h: float,
                 max_iter: int = 1000000):
        self._A = A
        self._B = B
        self._max_iter = max_iter
        self._f = f
        self._h = h

        # We'll define Poisson equation on interval (0, 5)
        self._a = DOMAIN_A
        self._b = DOMAIN_B
        
        self._n = int((self._b - self._a) / h)
        # self._poisson_mat = np.zeros((self._n, self._n))
        self._poisson_mat = generate_poisson_matrix(self._n - 2)
        # self._mat_rhs = np.zeros((self._n,))
        self._mat_rhs = generate_rhs(self._f, self._h, self._A, self._B, self._n - 2)

        self.u = np.zeros(self._n)
        self.u[0] = self._A
        self.u[-1] = self._B
        

    def solve_poisson(self, 
                      method:str = "Jakobi",
                      verbose: bool = False):
        '''
        method: "Jakobi", "GS", "numpy".
        verbose = False(def): print ||Ax-b||_max error for every 100 steps.
        '''
        if method == "Jakobi":
            self.u[1:-1] = J_solver(self._poisson_mat, self._mat_rhs, self._max_iter, verbose)
        elif method == "GS":
            self.u[1:-1] = GS_solver(self._poisson_mat, self._mat_rhs, self._max_iter, verbose)
        elif method == "numpy":
            begin = time()
            self.u[1:-1] = np.linalg.solve(self._poisson_mat, self._mat_rhs)
            end = time()
            print(f"\nnumpy solver took {end-begin:0.2f} seconds \n")
            print(f"error ||Ax - b||_max = {np.linalg.norm(self._poisson_mat @ self.u[1:-1] - self._mat_rhs, np.inf):e}")
        else:
            raise AttributeError(f"Wrong method {method}")
        
        return self.u


def rhs1(x):
    return np.sin(x) - np.cos(2*x)

def u_ex1(x): # a = 0, b = 5, A = 0, B = 1
    return (x - 2*x*np.sin(5) + 10*np.sin(x) - 5*np.cos(x)**2 + x*np.cos(5)**2 + 5) / 10


def rhs2(x):
    return (2*x + 5)*np.exp(x)

def u_ex2(x): # a = 0, b = 5, A = 1, B = 11 * np.exp(5)
    return (2*x + 1)* np.exp(x)


def rhs3(x):
    return np.sin(x)

def u_ex3(x): # a = 0, b = 5, A = 0, B = 1
    return (x - x* np.sin(5) + 5 * np.sin(x))/5


def rhs4(x):
    return np.ones_like(x)

def  u_ex4(x):
    return (27 - 5*x)*x/10


def plot_sol(y, y_ex):
    x = np.linspace(DOMAIN_A, DOMAIN_B, y.shape[0])
    plt.plot(x, y, label="u")
    plt.plot(x, y_ex, label="u_ex")
    plt.legend()
    plt.show()

def plot_error(rhs, u_ex):
    fig, ax = plt.subplots()
    hs = [1, 0.5, 0.1, 0.05, 0.01]
    for method in ["Jakobi", "GS"]:
        errors = []
        for h in hs: 
            p = Poisson(f=rhs, A=0, B=1, h=h, max_iter=100)
            u = p.solve_poisson(method=method)
            x = np.linspace(DOMAIN_A, DOMAIN_B, u.shape[0])
            errors.append(np.linalg.norm(u - u_ex(x), np.inf))
        ax.plot(hs, errors, label=f"{method}")
    ax.set_yscale
    ax.legend()
    plt.show()

    x = np.linspace(DOMAIN_A, DOMAIN_B,)

 
if __name__ == "__main__":
    p = Poisson(f=rhs1, A=0, B=1, h=0.01, max_iter=100000)
    u = p.solve_poisson(verbose=True, method="Jakobi")

    x = np.linspace(DOMAIN_A, DOMAIN_B, u.shape[0])
    u_ex = u_ex1(x)
    print(f"u - u_i error: {np.linalg.norm(u - u_ex, np.inf)}")

    # plot_sol(u, u_ex)
    # plot_error(rhs4, u_ex4)
