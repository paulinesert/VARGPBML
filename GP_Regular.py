import numpy as np
from numpy.linalg import cholesky, det
import matplotlib.pyplot as plt

from scipy.linalg import cho_solve
from scipy.optimize import minimize

class GP_regression():
    
    def __init__(self): 
        pass  
        
    def kernel(self,X1, X2, theta=[1.0, 1.0]):
        """
        Compute the correlation matrix between X1 and X2 based on the
        chosen kernel.
        
        Implemented : Radial Basis Function (RBF) kernel 
        
        Input
        ------
        X1: Training samples, Array of shape (n_samples_1, n_features).
        X2: Training samples, Array of shape (n_samples_2, n_features).
        theta: Kernel parameters, Array st 
            theta[0]: length parameter (not squared)
            theta[1]: scale parameter (not squared)
            Default: [1.0, 1.0]
        
        Output
        ------
        K: Array of (n_samples_1, n_samples_2) shape.
        
        """
        
        length = theta[0]
        scale = theta[1]
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return scale**2 * np.exp(-0.5 / length**2 * sqdist)

 
        
    def predict(self, X, X_train, y_train, theta=[1.0, 1.0], noise=1e-8):
        """
        Compute the posterior mean and covariance. 
        
        Input
        -----
        X: Test samples, array of shape (n_samples_test, n_features)
        X_train: Training samples,  Array of shape (n_samples_train, n_features)
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
        
        n = X_train.shape[0] #number of training samples
        Ks = self.kernel(X_train, X, theta) 
        Ky = self.kernel(X_train, X_train, theta) + noise**2 * np.eye(n)
        
        
        #The Cholesky decomposition is useful for solving linear systems with symmetric, positive definite coefficient matrix
        L_Ky = np.linalg.cholesky(Ky)
        
        # Compute the posterior mean 
        f = Ks.T.dot(cho_solve((L_Ky, True),y_train)) 
        
        # Compute the posterior variance
        Kss = self.kernel(X, X, theta)  
        var = Kss - Ks.T.dot(cho_solve((L_Ky, True),Ks)) + noise**2 * np.eye(len(X))
        
        return f, var

        
    def ll(self, theta, noise=0):  
        """
        Compute the log marginal likelihood. 
        The notations used are the ones from eq. 5.8 in 
        "Gaussian Processes for Machine Learning", C.Rasmussen et. al.
        
        Input
        -----
        theta: Kernel parameters, Array st 
            theta[0]: length parameter (not squared)
            theta[1]: scale parameter (not squared)
            
            
        noise: Noise on y, scalar 
            Default: 0 
        
        Output
        ------
        log_l: the log marginal likelihood
    
        """
        
        Ky = self.kernel(self.X_train, self.X_train, theta) + noise**2 * np.eye(self.n_samples)
        L_Ky = np.linalg.cholesky(Ky)
        
        #efficient determinant computation: cf Rassmussen A.18
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
            nll = - self.ll(params[0:-1], params[-1])
        else:
            nll = - self.ll(params)
            
        return nll
    
        
    def fit(self, X_train, y_train, params0, noise_estimation=True):
        """
        Fit the GP model.
        
        Input
        -----
        X_train: the training set, array of shape (n_samples_train, n_features)
        y_train: the observations at the X_train's locations, array of shape (n_samples_train,1)
        params0: the parameters w.r.t. to which the optimization will be performed for the initialization of the 
                optimization algorithm.
        noise_estimation: boolean to set if we want to estimate the noise on y or not.
            Default: True
        """
        
        self.n_samples = X_train.shape[0]
        self.X_train = X_train
        self.y_train = y_train
          
        if not noise_estimation:
            self.noise = None
        else:
            self.noise = params0[-1]
            
        # Choice: we decided not to put any bounds on the parameters.
        result = minimize(self.loss_function,
                         params0,
                         method='BFGS',
                         options={'disp': True})
        
        # Set the models' parameters & final negative likelihood value
        if noise_estimation:
            self.theta = result.x[0:-1]
            self.noise = result.x[-1]
            self.neg_ll_val =  - self.ll(self.theta, self.noise)

        else: 
            self.theta = result.x
            self.neg_ll_val =  - self.ll(self.theta)
            
        print("Final negative log marginal likelihood:",self.neg_ll_val)



# Functions to visualize

def plot_gp(mu, covmat, X, X_train=None, Y_train=None, samples=[], width=2.):
    plt.figure(figsize=(14,11))
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = width * np.sqrt(np.diag(covmat))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.2)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'ro', label='Training function points')
    plt.plot(X, mu + uncertainty, ls='--', label="+ 2*sigma")
    plt.plot(X, mu - uncertainty, ls='--', label="- 2*sigma")
    plt.legend(bbox_to_anchor=(1.04,0.5),loc='center left')
   