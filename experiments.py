import numpy as np
import matplotlib.pyplot as plt
# -------------------------------
# Helper: Quadratic function generator
# -------------------------------
def generate_quadratic(dim=10, cond_number=10):
    """Generate symmetric positive definite Q with given condition number."""
    V = np.linalg.qr(np.random.randn(dim, dim))[0]  # orthogonal matrix
    eigvals = np.linspace(1, cond_number, dim)      # spread of eigenvalues
    Q = V @ np.diag(eigvals) @ V.T
    return Q

# Quadratic function and gradient
def quadratic_f(x, Q):
    return 0.5 * x.T @ Q @ x

def quadratic_grad(x, Q):
    return Q @ x

# Rosenbrock function and gradient
def rosenbrock_f(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

# -------------------------------
# Optimizers (with loss tracking)
# -------------------------------
def sgd_momentum(f, grad_f, x0, lr=0.01, beta=0.9, tol=1e-6, max_iter=10000):
    x = x0.copy()
    v = np.zeros_like(x)
    losses = []
    for i in range(max_iter):
        loss = f(x)
        losses.append(loss)
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        v = beta * v + (1 - beta) * g
        x -= lr * v
    return x, i + 1, losses

def adam(f, grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-6, max_iter=10000):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    losses = []
    for t in range(1, max_iter + 1):
        loss = f(x)
        losses.append(loss)
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return x, t, losses

def rmsprop(f, grad_f, x0, lr=0.001, beta=0.9, eps=1e-8, tol=1e-6, max_iter=10000):
    x = x0.copy()
    s = np.zeros_like(x)
    losses = []
    for i in range(max_iter):
        loss = f(x)
        losses.append(loss)
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        s = beta * s + (1 - beta) * (g ** 2)
        x -= lr * g / (np.sqrt(s) + eps)
    return x, i + 1, losses
def sara_adam(f, grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999,
              G=1e-3, eps=1e-8, tol=1e-6, max_iter=10000):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)    
    prev_update = np.zeros_like(x)
    losses = []

    for t in range(1, max_iter + 1):
        loss = f(x)
        losses.append(loss)
        g = grad_f(x)

        if np.linalg.norm(g) < tol:
            break

        # First moment
        m = beta1 * m + (1 - beta1) * g

        # Second moment
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # Speed-aware correction
        denom = np.abs(prev_update) + np.abs(g) + eps
        s_t = (np.abs(prev_update) * np.abs(g)) / denom

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        x -= update
        prev_update = update.copy()

    return x, t, losses
##### <<< END SARA-Adam

# -------------------------------
# Experiment Runner + Plotter
# -------------------------------
def run_and_plot(f, grad_f, x0, optimizers, tol, f_name, logy=True):
    plt.figure(figsize=(7, 5))
    for name, opt in optimizers.items():
        x_opt, n_iter, losses = opt(f, grad_f, x0)
        print(f"{f_name} | {name} converged in {n_iter} iterations.")
        plt.plot(losses, label=name)
    if logy:
        plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Loss value")
    plt.title(f"Loss Decay: {f_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# 1️1 Well-conditioned Quadratic
# -------------------------------
dim = 10
Q_well = generate_quadratic(dim, cond_number=10)
x0 = np.random.randn(dim)
optimizers = {
    "SGD+Momentum": lambda f, g, x0=x0: sgd_momentum(f, g, x0, lr=0.01),
    "Adam": lambda f, g, x0=x0: adam(f, g, x0, lr=0.01),
    "RMSProp": lambda f, g, x0=x0: rmsprop(f, g, x0, lr=0.01),
    "ADASARA": lambda f, g, x0=x0: sara_adam(f, g, x0, lr=0.01, G=1e-4),
}
print("\n--- Well-Conditioned Quadratic ---")
run_and_plot(lambda x: quadratic_f(x, Q_well),
             lambda x: quadratic_grad(x, Q_well),
             x0, optimizers, tol=1e-6, f_name="Well-Conditioned Quadratic")

# -------------------------------
# 2️⃣ Ill-conditioned Quadratic
# -------------------------------
Q_ill = generate_quadratic(dim, cond_number=100)
x0 = np.random.randn(dim)
optimizers = {
    "SGD+Momentum": lambda f, g, x0=x0: sgd_momentum(f, g, x0, lr=0.001),
    "Adam": lambda f, g, x0=x0: adam(f, g, x0, lr=0.001),
    "RMSProp": lambda f, g, x0=x0: rmsprop(f, g, x0, lr=0.001),
    "ADASARA": lambda f, g, x0=x0: sara_adam(f, g, x0, lr=0.01, G=5e-3),
}
print("\n--- Ill-Conditioned Quadratic ---")
run_and_plot(lambda x: quadratic_f(x, Q_ill),
             lambda x: quadratic_grad(x, Q_ill),
             x0, optimizers, tol=1e-6, f_name="Ill-Conditioned Quadratic")

# -------------------------------
# 3️⃣ Rosenbrock Function
# -------------------------------
x0 = np.array([-1.2, 1.0])
optimizers = {
    "SGD+Momentum": lambda f, g, x0=x0: sgd_momentum(f, g, x0, lr=0.001, tol=1e-4),
    "Adam": lambda f, g, x0=x0: adam(f, g, x0, lr=0.001, tol=1e-4),
    "RMSProp": lambda f, g, x0=x0: rmsprop(f, g, x0, lr=0.001, tol=1e-4),
    "AdaSARA": lambda f, g, x0=x0: sara_adam(f, g, x0, lr=0.01, G=1e-2),
}
print("\n--- Rosenbrock Function ---")
run_and_plot(rosenbrock_f, rosenbrock_grad, x0, optimizers, tol=1e-4, f_name="Rosenbrock Function")
