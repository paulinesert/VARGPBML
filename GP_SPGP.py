import numpy as np
from numpy.linalg import cholesky, det
import matplotlib.pyplot as plt

from scipy.linalg import cho_solve
from scipy.optimize import minimize

# Adapted from Sparse Gaussian Processes using Pseudo-inputs, 
# https://papers.nips.cc/paper/2005/file/4491777b1aa8b5b32c2e8666dbe1a495-Paper.pdf

class GP_SPGP_regression():
    
    def __init__(self,eps=1e-8): 
        self.eps = eps
    
    def kernel(self, X1, X2, theta=[1.0, 1.0]):
        """
        Compute the correlation MATRIX between X1 and X2 based on the
        chosen kernel.
        
        Implemented : Radial Basis Function (RBF) kernel 
        Adapted from: https://towardsdatascience.com/what-on-earth-is-a-gaussian-process-992a6fa2946b
        
        Input
        ------
        X1: Array of shape (n_samples_1, n_features).
        X2: Array of shape (n_samples_2, n_features).
        theta: Array st 
            theta[0]: length parameter (not squared)
            theta[1]: scale parameter (not squared)
        
        Output
        ------
        K: Array of (n_samples_1, n_samples_2) shape.
        
        """
        
        length = theta[0]
        scale = theta[1]
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return scale**2 * np.exp(-0.5 / length**2 * sqdist)

 
        
    def predict(self, X_test, X_train, X_induced, y_train, theta=[1.0, 1.0], noise=1e-8):
        """
        Compute the posterior mean and covariance. 
        
        Input
        -----
        X: Test samples, array of shape (n_samples_test, n_features)
        X_train: Training samples,  Array of shape (n_samples_train, n_features)
        X_induced: Inducing samples, Array of shape(n_inducing_points, n_features)
        y_train: Training samples, Array of shape (n_samples_train, 1)
        
        theta: Kernel parameters, Array st 
            theta[0]: length parameter (not squared)
            theta[1]: scale parameter (not squared)
            Default: [1.0, 1.0]
            
        noise: Noise on y, scalar 
            Default: 1e-8
        
        Output
        ------
        f: the posterior means, Array of shape (n_samples_test,)
        var: the posterior covariance matrix, Array of shape (n_samples_test, n_samples_test)
        """    
        
        n, dim = X_train.shape 
        m = len(X_induced)//dim
        X_induced = X_induced.reshape((m, dim)) # reconstructs the array of induced points in 2D

        Ks = self.kernel(X_train, X_test, theta) 
        K_n = self.kernel(X_train, X_train, theta) 
        
        K_m = self.kernel(X_induced, X_induced, theta) 
        inv_K_m = np.linalg.inv(K_m + self.eps * np.eye(m))
        K_nm = self.kernel(X_train, X_induced, theta)
        K_sm = self.kernel(X_induced, X_test, theta)
        
        Nystrom = K_nm @ (inv_K_m @ K_nm.T)
        Lambda = np.diag(np.diag(K_n - Nystrom))
        Lambda_sigma_inv = np.linalg.inv(Lambda + noise**2 * np.eye(n))
        
        Q_m = K_m + K_nm.T @ Lambda_sigma_inv @ K_nm
        
        #The Cholesky decomposition is useful for solving linear systems with symmetric, positive definite coefficient matrix
        L_Q_m = np.linalg.cholesky(Q_m)
        
        # Compute the posterior mean 
        f = K_sm.T.dot(cho_solve((L_Q_m, True), K_nm.T @ Lambda_sigma_inv @ y_train)) 
        
        
        # Compute the posterior variance
        Kss = self.kernel(X_test, X_test, theta) 
        inv_Q_m = np.linalg.inv(Q_m + self.eps * np.eye(m))

        var = Kss - K_sm.T @ (inv_K_m - inv_Q_m) @ K_sm + noise**2 * np.eye(len(X_test))
                
        return f, var

        
    def ll(self, theta, X_induced, noise=0):         
        """
        Compute the log marginal likelihood. 
        
        Input
        -----
        theta: Kernel parameters, Array st 
            theta[0]: length parameter (not squared)
            theta[1]: scale parameter (not squared)
            
        X_induced: Inducing samples, Array of shape(n_inducing_points, n_features)
        
        noise: Noise on y, scalar 
            Default: 0 
        
        Output
        ------
        log_l: the log marginal likelihood
        """
        
        X_induced = X_induced.reshape((self.m, self.dim)) # reconstructs the array of induced points in 2D
        
        K_n = self.kernel(self.X_train, self.X_train, theta)
        K_m = self.kernel(X_induced, X_induced, theta)
        K_nm = self.kernel(self.X_train, X_induced, theta)
        inv_K_m = np.linalg.inv(K_m + self.eps * np.eye(self.m))
        
        Nystrom = K_nm @ ( inv_K_m @ K_nm.T)
        Q_nn = Nystrom + np.diag(np.diag(K_n - Nystrom))
        
        
        Ky = Q_nn + noise**2 * np.eye(self.n_samples)
        L_Ky = np.linalg.cholesky(Ky)
        
        #efficient dterminant computation: cf Rassmussen A.18
        log_det_Ky = np.sum(np.log(np.diagonal(L_Ky)))
        
        log_l = -(
             log_det_Ky + \
             0.5* self.y_train.T.dot(cho_solve((L_Ky,True),self.y_train)) + \
             0.5* self.n_samples * np.log(2*np.pi)
                 )
        return log_l.flatten()  
    
    def loss_function(self, params):
        """
        Compute the negative of the log marginal likelihood. 
        
        Input
        -----
        params: the parameters w.r.t. to which the optimization will be performed.
        
        Output
        ------
        nll: the negative log marginal likelihood
        """
        
        if self.noise: 
            nll = - self.ll(params[0:2], params[2:self.m*self.dim+2], params[-1])
        else:
            nll = - self.ll(params[0:2], params[2:])
            
        return nll
    
        
    def fit(self, X_train, y_train, params0, m, noise_estimation=True):
        """
        Fit the GP model.
        
        Input
        -----
        X_train: the training set, array of shape (n_samples_train, n_features)
        y_train: the observations at the X_train's locations, array of shape (n_samples_train,1)
        params0: the parameters w.r.t. to which the optimization will be performed for the initialization of the 
                optimization algorithm.
        m: number of inducing points, int. 
        noise_estimation: boolean to set if we want to estimate the noise on y or not.
            Default: True
        """
        
        self.n_samples, self.dim = X_train.shape
        self.X_train = X_train
        self.y_train = y_train
        self.m = m 
        
        if not noise_estimation:
            self.noise = None
        else:
            self.noise = params0[-1]
                  
        # Choice: we decided not to put any bounds on the parameters.
        result = minimize(self.loss_function,
                         params0,
                         method='BFGS',
                         options={'disp': True})
        
        # Set the model final parameters & final negative likelihood value
        if noise_estimation:
            self.theta = result.x[0:2]
            self.X_induced_flat = result.x[2:self.m*self.dim+2]
            self.noise = result.x[-1]
            self.neg_ll_val =  - self.ll(self.theta, self.X_induced_flat, self.noise)
        
        else: 
            self.theta = result.x[0:2]
            self.X_induced_flat = result.x[2:]
            self.neg_ll_val =  - self.ll(self.theta, self.X_induced_flat)

        print("Final negative log marginal likelihood:",self.neg_ll_val)


# Functions to visualize

def plot_gp(mu, covmat, X, X_induced0_flat=None, X_induced_flat=None, X_train=None, Y_train=None, samples=[], width=2.):   
    plt.figure(figsize=(14,11))
    axis = plt.gca()
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = width * np.sqrt(np.diag(covmat))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.2)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'ro', label='Training function points')
    y_min,y_max = axis.get_ylim() 
    
    plt.scatter(X_induced0_flat, 0.95*y_max*np.ones(len(X_induced0_flat)), marker='x', label='Induced points initialized')
    plt.scatter(X_induced_flat, 0.95*y_min*np.ones(len(X_induced_flat)), marker='x', label='Induced points fitted')
    plt.plot(X, mu + uncertainty, ls='--', label="+ 2*sigma")
    plt.plot(X, mu - uncertainty, ls='--', label="- 2*sigma")
    plt.legend(bbox_to_anchor=(1.04,0.5),loc='center left')