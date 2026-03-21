library(mgcv)

data <- read.csv('test_poisson_data.csv')
m <- gam(y ~ s(x), data=data, method='REML', family=poisson())
S <- m$smooth[[1]][['S']][[1]]
cat('S dim:', dim(S), '\n')
cat('Is S diagonal:', all(abs(S - diag(diag(S))) < 1e-10), '\n')
cat('S diagonal:', round(diag(S), 6), '\n')
cat('S full matrix:\n')
print(round(S, 4))
cat('\nmgcv sp =', m$sp, '\n')
cat('mgcv edf =', sum(m$edf), '\n')

# Also print design matrix columns for smooth
X_sm <- model.matrix(m)
cat('\nSmooth design matrix columns 2-4 (first 5 rows):\n')
print(round(X_sm[1:5, 2:4], 6))

# Check if S is the penalty in reparameterized or original basis
cat('\nm$smooth[[1]]$F (reparameterization matrix) exists:', 
    !is.null(m$smooth[[1]]$F), '\n')
# cat('Reparameterization:\n')
# if (!is.null(m$smooth[[1]]$F)) print(round(m$smooth[[1]]$F[1:5,1:5], 4))

# Get the null space dimension
cat('m$smooth[[1]]$null.space.dim =', m$smooth[[1]]$null.space.dim, '\n')
cat('m$smooth[[1]]$bs.dim =', m$smooth[[1]]$bs.dim, '\n')

# The FULL 200x200 kernel matrix construction
x <- data$x
n <- length(x)
k <- 10
r_mat <- as.matrix(dist(x))
E <- r_mat^3  # TPS kernel for d=1, m=2: phi(r) = r^3
T_mat <- cbind(1, x)
QR <- qr(T_mat)
Q_full <- qr.Q(QR, complete=TRUE)
Z <- Q_full[,(ncol(T_mat)+1):n]
ZtEZ <- t(Z) %*% E %*% Z
eigs <- sort(eigen(ZtEZ)$values, decreasing=TRUE)
cat('\nPython ZtEZ (all-data, no scale) top 10 eigs:\n')
cat(round(eigs[1:10], 4), '\n')

# Now with mgcv's kernel scaling (might include a constant)
# mgcv uses: E_ij = |x_i - x_j|^3 * constant(d,m)
# For d=1, m=2, the constant depends on the specific formulation
# Try multiplying by various constants to match mgcv:
ratio <- max(sort(eigen(S)$values, decreasing=TRUE)[1]) / eigs[1]
cat('Scale ratio (mgcv/python):', ratio, '\n')
