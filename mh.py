import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

np.random.seed(21)

R = 100.0
sigma = 1.0

T = 16000
eps = 8.0

def E(x):
    return (np.sqrt((x**2).sum(axis=-1)) - R)**2 / (2*sigma**2)


def H(x, r):
    return E(x) + 0.5 * (r**2).sum(axis=-1)

g1d = np.arange(-2*R, 2*R, 0.1)
xg, yg = np.meshgrid(g1d, g1d)
grid = np.c_[xg.flatten(), yg.flatten()]
q = np.exp(-E(grid))
plt.matshow(q.reshape((g1d.shape[0], g1d.shape[0])))
plt.show()
print(q[q > 0.1].sum() / q.sum())
high_prob_points = grid[np.exp(-E(grid)) > 0.1]

#initialize
x = np.zeros((T, 2))
r = np.zeros((T, 2))

#x[-1] = np.random.random(size=2) * 2 * R - R
x[-1] = [R,0]

for _ in range(1):
    x[0] = x[-1]
    acp_count = 0
    for t in range(T-1):
        e_begin = E(x[0])
        x_tmp = x[t] + np.random.normal(size=(2,)) * eps
        e_end = E(x_tmp)
        acpr = np.exp(min(0, e_begin - e_end))
        acp = np.random.random() < acpr
        if acp:
            x[t+1] = x_tmp
            acp_count += 1
        else:
            x[t+1] = x[t]
    print("acceptance ratio", 1.0*acp_count/T)

    plt.plot(high_prob_points[:,0], high_prob_points[:,1], ".", color="yellow")
    plt.plot(x[:,0], x[:,1], "o-")
    plt.plot(x[0,0], x[0,1], "o", color="blue")
    plt.plot(x[-1,0], x[-1,1], "o", color="red")
    plt.xlim(-2*R, 2*R)
    plt.ylim(-2*R, 2*R)
    plt.show()


