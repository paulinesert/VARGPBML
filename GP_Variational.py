import numpy as np
from numpy.linalg import cholesky, det
import matplotlib.pyplot as plt

from scipy.linalg import cho_solve
from scipy.optimize import minimize

class GP_Var_regression():
    
    def __init__(self,eps=1e-8): 
        self.eps = eps
        #pass 
    
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
    
    def E_step(self):
        """
        Compute the variational lower bound on the true log marginal likelihood 
        using as induced points the points already in the inducing set
        plus a point that is in the working set, for all point of the working set. 
        
        The working set is a random subset of size len_working_set of training points
        that are not yet in the remaining set. 
        
        Selects the point in the working set that maximize the lower bound. 
        
        Updates the inducing set. 
        """
        
        Fvs = []
        self.m += 1 # we now one more inducing point
        
        working_set = list(set(np.random.choice(self.remaining_set, size=(self.len_working_set))))
        
        for j in working_set : 
            induced_pts_j = self.induced_points + [j]
            X_induced_j = self.X_train[induced_pts_j,:]
            Fv = self.lower_bound(self.theta, X_induced_j, self.noise)
            Fvs.append(float(Fv))
     
        # Select the jth point that maximizes the lower bound
        opt_j = working_set[np.argmax(Fvs)]
        self.max_lower_bound = np.max(Fvs)
    
        
        # Update the inducing & remaining sets
        self.induced_points = self.induced_points + [opt_j]
        
        self.remaining_set = [j for j in self.remaining_set if j != opt_j]
        
        
    def M_step(self):
        
        """
        Optimization step of the variational lower bound on the true log marginal likelihood
        w.r.t the parameters of the kernel and the noise sigma_y. 
        
        Updates the optimal kernel parameters and noise.

        """
        
        params0 = np.concatenate([self.theta0, np.array([self.noise0])])
        result = minimize(self.loss_function,
                         params0,
                         method='BFGS',
                         options={'disp': True})
        
        # Update the optimal kernel parameters and the noise
        self.theta = result.x[0:-1]
        self.noise = result.x[-1]
    
    def lower_bound(self, theta, X_induced, noise):
        
        """
        Compute the variational lower bound on the true log marginal likelihood. 
        
        The notations used are the ones from eq. TODO in 
        TODO ADD REF
        
        Input
        -----
        theta: parameters of the kernel used
            Array st 
                theta[0]: length parameter
                theta[1]: scale parameter

        X_induced: matrix of inducing inputs, array of shape (m, n_features)
            
        noise (Optional): value of the noise sigma_y.
        
        Output
        ------
        lb: the variational lower bound 
    
        """
      
        K_n = self.kernel(self.X_train, self.X_train, theta)
        K_m = self.kernel(X_induced, X_induced, theta) + self.eps * np.eye(self.m)
 
        K_nm = self.kernel(self.X_train, X_induced, theta)
        L_Km = np.linalg.cholesky(K_m + self.eps * np.eye(self.m))
        
        Nystrom = K_nm @ ( cho_solve((L_Km,True), K_nm.T))
        Q_nn = Nystrom 
        
        
        Ky = Q_nn + noise**2 * np.eye(self.n_samples)
        L_Ky = np.linalg.cholesky(Ky)
        
        # efficient determinant computation: Rassmussen A.18
        log_det_Ky = np.sum(np.log(np.diagonal(L_Ky)))
        
        # computation of the regularization term 
        X_remaining = self.X_train[self.remaining_set,:]
        K_r = self.kernel(X_remaining, X_remaining, theta)
        K_rm = self.kernel(X_remaining, X_induced, theta)
        reg = 1/(noise**2) * np.trace(K_r - K_rm @( cho_solve((L_Km,True), K_rm.T)))
        
        lb = -(log_det_Ky + \
             0.5* self.y_train.T.dot(cho_solve((L_Ky,True),self.y_train)) + \
             0.5* self.n_samples * np.log(2*np.pi) + \
             0.5 * reg
                 )
        return lb.flatten()  
        

 
        
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

        K_m = self.kernel(X_induced, X_induced, theta) 
        inv_K_m = np.linalg.inv(K_m + self.eps * np.eye(m))
        
        K_nm = self.kernel(X_train, X_induced, theta)
        K_sm = self.kernel(X_test, X_induced, theta)

        # Compute the mean and covariance matrix of the optimal distribution phi
        sigma_not_inv = K_m + 1/(noise**2) * K_nm.T @ K_nm
        sigma = np.linalg.inv(sigma_not_inv + self.eps * np.eye(m))
        
        mu = 1/(noise**2) * K_m @ sigma @ K_nm.T @ y_train
        
        A = K_m @ sigma @ K_m
                
        # Compute the posterior mean 
        f = K_sm @ inv_K_m @ mu 
        
        # Compute the posterior variance
        
        Kss = self.kernel(X_test, X_test, theta) 
        var = Kss - K_sm @ inv_K_m @ K_sm.T + K_sm @ inv_K_m @ A @ inv_K_m @ K_sm.T + noise**2 * np.eye(len(X_test))
                
        return f, var
    
    
    def loss_function(self, params):
        """
        Compute the negative variational lowerbound. 
        
        Input
        -----
        params: the parameters w.r.t. to which the optimization will be performed.
        
        Output
        ------
        nlb : the negative variational lower bound
        """
        
        # Get the inducing points
        X_induced = self.X_train[self.induced_points,:]
        
        
        # Compute the negative variational lower bound (to maximize it via minimization)
        nlb = - self.lower_bound(params[0:2], X_induced, params[-1])
        
        return nlb
    
        
    def fit(self, X_train, y_train, params0, expected_m, len_working_set,  noise_estimation=True):
        """
        Fit the GP model.
        
        Input
        -----
        X_train: the training set, array of shape (n_samples_train, n_features)
        y_train: the observations at the X_train's locations, array of shape (n_samples_train,1)
        params0: the parameters w.r.t. to which the optimization will be performed for the initialization of the 
                optimization algorithm.
        expected_m: number of inducing points wanted, int. 
        len_working_set: size of the working set J, int
        noise_estimation: boolean to set if we want to estimate the noise on y or not.
            Default: True
        """
        
        self.n_samples, self.dim = X_train.shape
        self.X_train = X_train
        self.y_train = y_train
        self.len_working_set = len_working_set
        self.theta0 = params0[0:-1]
        self.noise0 = params0[-1]
        self.theta = params0[0:-1]
        self.noise = params0[-1]
        
        self.induced_points = []    # initialization of the inducing points    
        self.m = 0 # no inducing points at initialization
        self.remaining_set = np.arange(self.n_samples) # initialization of the remaining_set

        
        for i in range(expected_m):
            self.E_step()
            self.M_step()
            
        self.X_induced = self.X_train[self.induced_points,:]
        self.max_lower_bound = self.lower_bound(self.theta, self.X_induced, self.noise)

        print("Fit complete!")

# Functions to visualize

def plot_gp(mu, covmat, X, X_induced_flat=None, X_train=None, Y_train=None, samples=[], width=2.):   
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
    
    plt.scatter(X_induced_flat, 0.95*y_min*np.ones(len(X_induced_flat)), marker='x', label='Induced points fitted')
    plt.plot(X, mu + uncertainty, ls='--', label="+ 2*sigma")
    plt.plot(X, mu - uncertainty, ls='--', label="- 2*sigma")
    plt.legend(bbox_to_anchor=(1.04,0.5),loc='center left')