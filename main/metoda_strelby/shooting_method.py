from time import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def get_time(method: Callable):
    def wrapper(*args, **kwargs):

        begin = time()

        returned_value = method(*args, **kwargs)

        end = time()

        print(f"{method.__name__} took {end - begin:3.3f} seconds")
        return returned_value

    return wrapper


class Shooting_method:
    def __init__(
        self,
        x_start: float,
        x_end: float,
        f_start: float,
        f_end: float,
        rhs: Callable[[float], float],
        tolerance: float = 10 ** (-14),
        max_iter: int = 100,
    ) -> None:
        """
        Initializes an instance of the Shooting_method class.

        Parameters:
        - x_start (float): Starting value of the independent variable.
        - x_end (float): Ending value of the independent variable.
        - f_start (float): Initial value of the dependent variable.
        - f_end (float): Final value of the dependent variable.
        - rhs (Callable[[float], float]): Function representing the right-hand side of the differential equation.
        """
        self.x_start, self.x_end = x_start, x_end
        self.f_start, self.f_end = f_start, f_end

        self.rhs = rhs

        self._tolerance = tolerance
        self._max_iter = max_iter
        self.u = np.array([])

    def _rhs(self, x: float, v: float) -> tuple[float, float]:

        der_u = v
        der_v = self.rhs(x)
        return (der_u, der_v)

    def _euler_method(self, v: float, h: float, set_u: bool = False) -> float:
        u_i = self.f_start
        v_i = v
        x_i = self.x_start
        u = np.zeros(int((self.x_end - self.x_start) / h + 1))

        u[0] = u_i
        for i in range(int((self.x_end - self.x_start) / h)):
            x_i += h

            k_1 = [h * elem for elem in self._rhs(x_i, v_i)]

            u_i += k_1[0]
            v_i += k_1[1]

            u[i + 1] = u_i

        if set_u:
            self.u = u

        return u[-1] - self.f_end

    def _runge_kutta(self, v: float, h: float, set_u: bool = False) -> float:

        u_i = self.f_start
        v_i = v
        x_i = self.x_start
        u = np.zeros(int((self.x_end - self.x_start) / h + 1))
        u[0] = u_i

        for i in range(int((self.x_end - self.x_start) / h)):
            k_1 = [h * elem for elem in self._rhs(x_i, v_i)]
            k_2 = [h * elem for elem in self._rhs(x_i + h / 2, v_i + k_1[1] / 2)]
            k_3 = [h * elem for elem in self._rhs(x_i + h / 2, v_i + k_2[1] / 2)]
            k_4 = [h * elem for elem in self._rhs(x_i + h, v_i + k_3[1])]

            x_i += h

            u_i += (k_1[0] + 2 * k_2[0] + 2 * k_3[0] + k_4[0]) / 6
            v_i += (k_1[1] + 2 * k_2[1] + 2 * k_3[1] + k_4[1]) / 6

            u[i + 1] = u_i

        if set_u:
            self.u = u

        return u[-1] - self.f_end

    @get_time
    def _bisection_method(
        self, f: Callable[[float, float], float], h: float, a: float, b: float
    ) -> float:

        f_a, f_b, f_c = f(a, h), f(b, h), 0.0

        if f_a * f_b > 0:
            raise Exception(f"bad interval, f_a = {f_a}, f_b = {f_b}")

        if f_a == 0.0 or abs(f_a) < self._tolerance:
            return a

        if f_b == 0.0 or abs(f_b) < self._tolerance:
            return b

        n = 1
        while n < self._max_iter:
            c = (a + b) / 2
            f_c = f(c, h)
            if f_c == 0.0 or abs((b - a) / 2) < self._tolerance:
                return c
            else:
                n += 1
                if f_c * f(a, h) > 0:
                    a = c
                else:
                    b = c
        raise Exception("max iterations exceed with f={0}".format(f_c))

    def _df(self, f: Callable[[float, float], float], h: float, x: float) -> float:
        return (f(x + self._tolerance, h) - f(x, h)) / self._tolerance

    @get_time
    def _newton(
        self, f: Callable[[float, float], float], h: float, a: float, b: float
    ) -> float:

        n = 1
        y = 0

        while n < self._max_iter:
            y = f(a, h)
            dy = self._df(f, h, a)
            c = a - (y / dy)
            if abs(c - a) < self._tolerance:
                return c

            a = c
            n += 1

        raise Exception("max iterations exceed with f={0}".format(y))

    @get_time
    def shoot(self, find_root_method: str, find_u_method: str, h_step: float) -> None:
        """
        Executes the shooting method to solve the boundary value problem.

        Parameters:
        - find_root_method (str): Method to find the root of the boundary value residual equation.
        - find_u_method (str): Method to solve the initial value problem to match boundary conditions.
        """
        if find_root_method == "bisection":
            root_method = self._bisection_method
        elif find_root_method == "newton":
            root_method = self._newton
        else:
            raise Exception("wrong find root method")

        if find_u_method == "euler":
            u_method = self._euler_method
        elif find_u_method == "runge_kutta":
            u_method = self._runge_kutta
        else:
            raise Exception("wrong find solution method")

        a = -10
        b = 10
        v_h = root_method(u_method, h_step, a, b)

        u_method(v_h, h_step, True)


def L_inf_norm_error(u_h: np.ndarray, u_ex: np.ndarray) -> float:
    """
    Calculates the L-infinity norm error between the numerical solution and the exact solution.

    Returns:
    - float: L-infinity norm error.
    """
    return np.max(np.abs(u_h - u_ex))


def L_2_norm_error(u_h: np.ndarray, u_ex: np.ndarray) -> float:
    """
    Calculates the L2 norm error between the numerical solution and the exact solution.

    Returns:
    - float: L2 norm error.
    """
    return np.sum((u_h - u_ex) ** 2, axis=0) ** 0.5


def plot_error(
    u_ex: Callable[[np.ndarray], np.ndarray],
    s: Shooting_method,
    find_root_method: str = "newton",
    l_norm: str = "L_inf",
) -> None:
    """
    Plots the error as a function of step size.

    Parameters:
    - find_root_method (str) =def "newton": Method to find the root of the boundary value residual equation.
    - l_norm (str) =def "L_inf": L-norm error that will be used.
    """
    if l_norm == "L_2":
        norm = L_2_norm_error
    else:
        norm = L_inf_norm_error
    l_norms = []
    h_steps = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    fig, ax = plt.subplots()

    for ode_method in ["euler", "runge_kutta"]:
        l_norms.append([])
        for h in h_steps:
            s.shoot(find_root_method, ode_method, 2**-h)

            mesh = np.linspace(
                s.x_start, s.x_end, int((s.x_end - s.x_start) / 2**-h + 1)
            )

            l_norms[-1].append(norm(s.u, u_ex(mesh)))

        ax.plot([2**-h for h in h_steps], l_norms[-1], "-o", label=f"{ode_method}")
        plt.xticks([2**-h for h in h_steps])

    ax.grid()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{l_norm} norm error")
    ax.set_xlabel("h step size")
    ax.set_ylabel("error")
    ax.legend()
    plt.show()


def plot_u(
    s: Shooting_method,
    u_ex: Callable[[np.ndarray], np.ndarray] | None = None,
    find_u_method: str = "runge_kutta",
    find_root_method: str = "newton",
    h_uh_step: float = 0.01,
    h_uex_step: float = 0.00001,
) -> None:
    """
    Plots the numerical and exact solutions.

    Parameters:
    - find_u_method (str): Method to solve the initial value problem to match boundary conditions.
    - find_root_method (str): Method to find the root of the boundary value residual equation.
    - h_step (float): Step size for numerical integration.
    """
    if u_ex is not None:
        x_ex = np.linspace(s.x_start, s.x_end, int((s.x_end - s.x_start) / h_uex_step))
        y_ex = u_ex(x_ex)
        plt.plot(x_ex, y_ex, label="u_ex")

    x_h = np.linspace(s.x_start, s.x_end, int((s.x_end - s.x_start) / h_uh_step + 1))
    s.shoot(find_root_method, find_u_method, h_uh_step)
    y_h = s.u

    plt.plot(x_h, y_h, label="u_h")
    plt.grid()
    plt.title("u(x)")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.show()


def rhs1(x: float) -> float:
    return np.sin(x) - x * np.cos(2 * x)


def u_ex1(x: np.ndarray) -> np.ndarray:  # a = 0, b = 5, A = 0, B = 1
    return (
        4 * x
        + x * np.sin(10)
        + 4 * x * np.sin(5)
        - 20 * np.sin(x)
        - 5 * np.sin(2 * x)
        + 5 * x * np.cos(2 * x)
        - 5 * x * np.cos(10)
    ) / 20


def rhs2(x: float) -> float:
    return (2 * x + 5) * np.exp(x)


def u_ex2(x: np.ndarray) -> np.ndarray:  # a = 0, b = 5, A = 1, B = 11 * np.exp(5)
    return (2 * x + 1) * np.exp(x)


def rhs3(x: float) -> float:
    return np.sin(x)


def u_ex3(x: np.ndarray) -> np.ndarray:  # a = 0, b = 5, A = 0, B = 1
    return (x + x * np.sin(5) - 5 * np.sin(x)) / 5


if __name__ == "__main__":

    s = Shooting_method(x_start=0.0, x_end=5.0, f_start=0, f_end=1, rhs=rhs1)

    s.shoot("newton", "runge_kutta", 0.00001)
    plot_u(
        s, u_ex=u_ex1, h_uh_step=0.01, find_root_method="newton", find_u_method="euler"
    )
    u = u_ex1(np.linspace(s.x_start, s.x_end, int((s.x_end - s.x_start) / 0.01 + 1)))

    print(L_inf_norm_error(s.u, u))
    plot_error(u_ex1, s, "newton")
