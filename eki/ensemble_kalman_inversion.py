import numpy as np

class EKI:
    def __init__(self, y, d, k, Gamma, J, initializer, forward_model, lensing_model=None):
        self.y = y
        
        self.d = d
        self.k = k
        self.J = J
        self.n = 0

        self.Gamma = Gamma

        self.H = np.block([np.zeros((self.k, self.d)), np.eye(self.k)])
        self.H_star = self.H.T
        self.H_perp = np.block([np.eye(self.d), np.zeros((self.d, self.k))])

        self.initializer = initializer
        self.forward_model = forward_model
        self.lensing_model = lensing_model
        
        self.z = self._initialize_ensemble()
        self.compute_mean_parameters()
        
        self.z_hat = None
        self.C = None
        self.K = None
        
        self.history = [{"z": self.z,
                         "u": self.u,
                         "z_hat": self.z_hat,
                         "C": self.C,
                         "K": self.K}]
        
    def _initialize_ensemble(self):
        ensemble = []
        for j in range(self.J):
            psi = self.initializer()
            G_psi = self.forward_model(psi)
            z = np.concatenate([psi, G_psi])
            ensemble.append(z)
        return ensemble

    def _Xi(self, z_n):
        u = z_n[:self.d]
        G_u = self.forward_model(u)
        return np.concatenate([u, G_u])

    def prediction_step(self):
        # hat{z_{n+1}^(j)} = Xi(z_n^(j)) [Iglesias (9)]
        self.z_hat = [self._Xi(z_j) for z_j in self.z]
        # bar{z_{n+1}} = (1 / J) Sum_{j=1}^J hat{z_{n+1}^(j)} [Iglesias (10)]
        z_bar = np.mean(self.z_hat, axis=0)
        # C_{n+1} = (1 / J) Sum_{j=1}^J hat{z_{n+1}^(j)} hat{z_{n+1}^(j)}^T - bar{z_{n+1}^(j)} bar{z_{n+1}^(j)}^T [Iglesias (11)]
        self.C = np.mean([np.outer(z_j, z_j) for z_j in self.z_hat], axis=0) - np.outer(z_bar, z_bar)

    def _particle_analysis(self, z_hat_j):
        # y_{n+1}^(j) = y + eta_{n+1}^(j) [Iglesias (15)]
        y_j = self.y + np.random.multivariate_normal(np.zeros(self.k), self.Gamma)
        # z_{n+1}^(j) = (I - K_{n+1} H) hat{z_{n+1}}^(j) + K_{n+1} y_{n+1}^(j) [Iglesias (14)]
        z_n_1 = (np.eye(self.d + self.k) - self.K @ self.H) @ z_hat_j + self.K @ y_j
        return z_n_1

    def analysis_step(self):
        # K_{n+1} = C_{n+1} H* (H C_{n+1} H* + Gamma)^{-1}
        self.K = self.C @ self.H_star @ np.linalg.inv(self.H @ self.C @ self.H_star + self.Gamma)
        self.z = [self._particle_analysis(z_hat_j) for z_hat_j in self.z_hat]
    
    def compute_mean_parameters(self):
        # u_{n+1} = (1/J) Sum_{j=1}^J H_perp * z_{n+1}^(j) [Iglesias (16)]
        self.u = np.mean([self.H_perp @ z_j for z_j in self.z], axis=0)

    # Full Algorithm
    def invert(self, stopping_algo=None):
        while not stopping_algo(self):
            self.n += 1
            self.prediction_step()
            self.analysis_step()
            self.compute_mean_parameters()
            self.history.append({"z": self.z,
                                 "u": self.u,
                                 "z_hat": self.z_hat,
                                 "C": self.C,
                                 "K": self.K})
        return self.u

    
def naive_convergence_stopping(eki_obj, error=1e-3):
    if eki_obj.n >= 1:
        if np.linalg.norm(eki_obj.history[-1]["u"] - eki_obj.history[-2]["u"]) < error:
            return True
    return False



