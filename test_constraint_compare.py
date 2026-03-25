"""Compare pymgcv constraint (drop+center) vs QR constraint absorption (R mgcv style)."""
import numpy as np
import sys; sys.path.insert(0, '.')
import pandas as pd
import pymgcv.smooth.thin_plate as tp
from scipy import linalg, spatial

data = pd.read_csv('campaign_data.csv')
x = data['age'].values
X = x.reshape(-1, 1)
n = len(X); d = 1; M = 2; k = 10
knots = X.copy(); nk = n

# Reproduce basis construction
dists_kk = spatial.distance.cdist(knots, knots, 'euclidean')
E = tp._tps_kernel(dists_kk, d, 2)
E = 0.5 * (E + E.T)

T_mat = np.column_stack([np.ones(nk), knots])
Q_full = linalg.qr(T_mat, mode='full')[0]
Z_qr = Q_full[:, M:]

ZtEZ = Z_qr.T @ E @ Z_qr
ZtEZ = 0.5 * (ZtEZ + ZtEZ.T)
eigvals, eigvecs = linalg.eigh(ZtEZ)
order = np.argsort(-np.abs(eigvals))
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

n_keep = k - M
D_k = eigvals[:n_keep]
U_k = eigvecs[:, :n_keep]

F_pred = np.linalg.lstsq(E, Z_qr @ U_k, rcond=None)[0]
B_raw = np.column_stack([T_mat, E @ F_pred])
S_raw = np.zeros((k, k))
S_raw[M:, M:] = np.diag(np.abs(D_k))

print("=== Raw (pre-constraint) ===")
print("B_raw shape:", B_raw.shape)
print("S_raw eigs:", np.sort(np.linalg.eigvalsh(S_raw))[::-1][:5])

# --- QR constraint absorption (R mgcv style) ---
C = B_raw.mean(axis=0)  # constraint vector (k,)
Q_c, _ = np.linalg.qr(C.reshape(-1, 1), mode='complete')
Z_con = Q_c[:, 1:]  # (k, k-1) orthogonal complement

B_qr = B_raw @ Z_con
S_qr = Z_con.T @ S_raw @ Z_con

print("\n=== QR constraint absorption (R mgcv style) ===")
print("B_qr shape:", B_qr.shape)
print("S_qr eigs:", np.sort(np.linalg.eigvalsh(S_qr))[::-1][:5])
print("S_qr rank:", np.linalg.matrix_rank(S_qr, tol=1e-10))
print("colMeans max:", np.max(np.abs(B_qr.mean(axis=0))))

# --- pymgcv style (drop column 0 + center) ---
B_pymgcv = B_raw[:, 1:]
S_pymgcv = S_raw[1:, 1:]
cm = B_pymgcv.mean(axis=0)
B_pymgcv -= cm

print("\n=== pymgcv constraint (drop+center) ===")
print("B_pymgcv shape:", B_pymgcv.shape)
print("S_pymgcv eigs:", np.sort(np.linalg.eigvalsh(S_pymgcv))[::-1][:5])
print("S_pymgcv rank:", np.linalg.matrix_rank(S_pymgcv, tol=1e-10))
print("colMeans max:", np.max(np.abs(B_pymgcv.mean(axis=0))))

# --- Check if eigenvalues are the same ---
eigs_qr = np.sort(np.linalg.eigvalsh(S_qr))[::-1]
eigs_pymgcv = np.sort(np.linalg.eigvalsh(S_pymgcv))[::-1]
print("\n=== Eigenvalue comparison ===")
for i in range(9):
    print(f"  eig[{i}]: QR={eigs_qr[i]:12.4f}  pymgcv={eigs_pymgcv[i]:12.4f}  ratio={eigs_qr[i]/(eigs_pymgcv[i]+1e-30):.6f}")
