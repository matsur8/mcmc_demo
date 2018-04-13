import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

np.random.seed(21)

R = 3.0
sigma = 0.1

T = 100
eps = 0.05

def E(x):
    return (np.sqrt((x**2).sum(axis=-1)) - R)**2 / (2*sigma**2)

def H(x, r):
    return E(x) + 0.5 * (r**2).sum(axis=-1)

g1d = np.arange(-2*R, 2*R, 0.1)
xg, yg = np.meshgrid(g1d, g1d)
grid = np.stack([xg,yg], -1)
q = np.exp(-E(grid))
plt.matshow(q)
plt.show()
print(q[q > 0.1].sum() / q.sum())
high_prob_points = grid[q > 0.1]

#initialize
x = np.zeros((T, 2))
r = np.zeros((T, 2))

#x[-1] = np.random.random(size=2) * 2 * R - R
x[-1] = [R,0]

for _ in range(10):
    x[0] = x[-1]
    r[0] = np.random.normal(size=2)
    h_begin = H(x[0], r[0])
    direction = 2*np.random.randint(2)-1
    for t in range(T-1):  
        r_tmp = r[t] - direction * 0.5 * eps * x[t] * (np.sqrt((x[t]**2).sum()) - R) / (sigma**2 * np.sqrt((x[t]**2).sum()))
        x[t+1] = x[t] + direction * eps * r_tmp
        r[t+1] = r_tmp - direction * 0.5 * eps * x[t+1] * (np.sqrt((x[t+1]**2).sum()) - R) / (sigma**2 * np.sqrt((x[t+1]**2).sum()))
        #print((0.5*x[t+1]**2/sigma**2 + 0.5 * r[t+1]**2).sum())
    h_end = H(x[-1], r[-1])
    acpr = np.exp(min(0, h_begin - h_end))
    acp = np.random.random() < acpr

    plt.plot(high_prob_points[:,0], high_prob_points[:,1], ".", color="yellow")
    plt.plot(x[:,0], x[:,1], ".-", color="gray")
    plt.plot(x[0,0], x[0,1], "o", color="blue")
    print("Hamiltonian: {} -> {}".format(h_begin, h_end))
    print("Acceptance ratio:", acpr) 
    print("Accepted." if acp else "Rejected.")
    if acp:
        plt.plot(x[-1,0], x[-1,1], "o", color="red")
    else:
        plt.plot(x[-1,0], x[-1,1], "o", color="pink")
    plt.xlim(-2*R, 2*R)
    plt.ylim(-2*R, 2*R)
    plt.show()
        
    if not acp:
        x[-1] = x[0]
         
