import time

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm

def _nan_check(name, arr):
    arr = jnp.asarray(arr)
    n_nan = int(jnp.sum(jnp.isnan(arr)))
    if n_nan > 0:
        print(f"  [NaN] {name}: {n_nan}/{arr.size} NaN values  shape={arr.shape}  "
              f"min={float(jnp.nanmin(arr)):.4g}  max={float(jnp.nanmax(arr)):.4g}")
    else:
        print(f"  [ok]  {name}: shape={arr.shape}  "
              f"min={float(arr.min()):.4g}  max={float(arr.max()):.4g}")


class EKI:
    def __init__(self, y, d, k, Gamma, J, initializer, forward_model, lensing_model=None, seed=0, verbose=False, param_bounds=None):
        self.verbose = verbose
        self.y = jnp.asarray(y)
        # param_bounds: array of shape (d, 2) with [low, high] per parameter, or None
        self.param_bounds = jnp.asarray(param_bounds) if param_bounds is not None else None

        self.d = d
        self.k = k
        self.J = J
        self.n = 0

        self.Gamma = jnp.asarray(Gamma)

        self.H = jnp.block([jnp.zeros((self.k, self.d)), jnp.eye(self.k)])
        self.H_star = self.H.T
        self.H_perp = jnp.block([jnp.eye(self.d), jnp.zeros((self.d, self.k))])

        self.key = jax.random.PRNGKey(seed)

        self.initializer = initializer
        self.forward_model = forward_model
        self.lensing_model = lensing_model

        self.z = self._initialize_ensemble()   # shape (J, d+k)
        if self.verbose:
            print("[init] ensemble z:")
            _nan_check("z", self.z)
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
        for _ in range(self.J):
            psi = self.initializer()
            G_psi = self.forward_model(psi)
            z = jnp.concatenate([jnp.asarray(psi), jnp.asarray(G_psi)])
            ensemble.append(z)
        return self._clamp_params(jnp.stack(ensemble))  # (J, d+k)

    def _clamp_params(self, z):
        if self.param_bounds is None:
            return z
        lo, hi = self.param_bounds[:, 0], self.param_bounds[:, 1]
        clamped = jnp.clip(z[:, :self.d], lo, hi)
        return z.at[:, :self.d].set(clamped)

    def _Xi(self, z_n):
        u = z_n[:self.d]
        G_u = self.forward_model(u)
        return jnp.concatenate([u, jnp.asarray(G_u)])

    def prediction_step(self):
        # hat{z_{n+1}^(j)} = Xi(z_n^(j)) [Iglesias (9)]
        self.z_hat = jnp.stack([self._Xi(self.z[j]) for j in range(self.J)])  # (J, d+k)
        # Drop NaN particles before computing statistics
        valid = ~jnp.any(jnp.isnan(self.z_hat), axis=1)  # (J,)
        z_valid = self.z_hat[valid]                        # (J_valid, d+k)
        J_valid = z_valid.shape[0]
        if self.verbose:
            print(f"[n={self.n}] prediction_step: {J_valid}/{self.J} valid particles")
        # bar{z_{n+1}} = (1 / J) Sum_{j=1}^J hat{z_{n+1}^(j)} [Iglesias (10)]
        z_bar = jnp.nanmean(z_valid, axis=0)
        # C_{n+1} = (1 / J) Sum_{j=1}^J hat{z^(j)} hat{z^(j)}^T - bar{z} bar{z}^T [Iglesias (11)]
        self.C = (z_valid.T @ z_valid) / J_valid - jnp.outer(z_bar, z_bar)
        if self.verbose:
            _nan_check("z_bar", z_bar)
            _nan_check("C", self.C)

    def analysis_step(self):
        # K_{n+1} = C_{n+1} H* (H C_{n+1} H* + Gamma)^{-1}  [use solve for stability]
        A = self.C @ self.H_star                              # (d+k, k)
        B = self.H @ self.C @ self.H_star + self.Gamma       # (k, k)
        self.K = jnp.linalg.solve(B.T, A.T).T               # (d+k, k)

        # y_{n+1}^(j) = y + eta_{n+1}^(j) [Iglesias (15)] — vectorized over J particles
        self.key, subkey = jax.random.split(self.key)
        eta = jax.random.multivariate_normal(subkey, jnp.zeros(self.k), self.Gamma, shape=(self.J,))  # (J, k)
        y_j = self.y + eta  # (J, k)

        # z_{n+1}^(j) = (I - K H) hat{z^(j)} + K y^(j) [Iglesias (14)] — vectorized
        IKH = jnp.eye(self.d + self.k) - self.K @ self.H    # (d+k, d+k)
        self.z = self._clamp_params((IKH @ self.z_hat.T).T + (self.K @ y_j.T).T)  # (J, d+k)
        if self.verbose:
            print(f"[n={self.n}] analysis_step:")
            _nan_check("A (C H*)", self.C @ self.H_star)
            _nan_check("B (H C H* + Gamma)", self.H @ self.C @ self.H_star + self.Gamma)
            _nan_check("K", self.K)
            _nan_check("eta", eta)
            _nan_check("z (updated)", self.z)

    def compute_mean_parameters(self):
        # u_{n+1} = (1/J) Sum_{j=1}^J H_perp * z_{n+1}^(j) [Iglesias (16)]
        self.u = self.H_perp @ jnp.nanmean(self.z, axis=0)
        if self.verbose:
            _nan_check("u", self.u)

    def save(self, path):
        """Save full EKI state to a .npz file.

        Callables (initializer, forward_model) are not serialisable and must
        be re-supplied when calling EKI.load().
        """
        arrays = {
            "n":           np.array(self.n),
            "d":           np.array(self.d),
            "k":           np.array(self.k),
            "J":           np.array(self.J),
            "verbose":     np.array(self.verbose),
            "y":           np.array(self.y),
            "Gamma":       np.array(self.Gamma),
            "z":           np.array(self.z),
            "u":           np.array(self.u),
            "key":         np.array(self.key),
            "history_len": np.array(len(self.history)),
        }
        for i, entry in enumerate(self.history):
            arrays[f"h{i}_z"] = np.array(entry["z"])
            arrays[f"h{i}_u"] = np.array(entry["u"])
            for field in ("z_hat", "C", "K"):
                val = entry[field]
                arrays[f"h{i}_{field}_none"] = np.array(val is None)
                if val is not None:
                    arrays[f"h{i}_{field}"] = np.array(val)
        np.savez(path, **arrays)
        print(f"EKI saved to {path}.npz  ({len(self.history)} history entries)")

    @classmethod
    def load(cls, path, initializer, forward_model, lensing_model=None):
        """Reconstitute an EKI from a file saved with EKI.save().

        initializer and forward_model must be re-provided since callables
        cannot be serialised.
        """
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        obj = object.__new__(cls)

        obj.n           = int(data["n"])
        obj.d           = int(data["d"])
        obj.k           = int(data["k"])
        obj.J           = int(data["J"])
        obj.verbose     = bool(data["verbose"])
        obj.y           = jnp.asarray(data["y"])
        obj.Gamma       = jnp.asarray(data["Gamma"])
        obj.z           = jnp.asarray(data["z"])
        obj.u           = jnp.asarray(data["u"])
        obj.key         = jnp.asarray(data["key"])

        obj.H      = jnp.block([jnp.zeros((obj.k, obj.d)), jnp.eye(obj.k)])
        obj.H_star = obj.H.T
        obj.H_perp = jnp.block([jnp.eye(obj.d), jnp.zeros((obj.d, obj.k))])

        obj.initializer   = initializer
        obj.forward_model = forward_model
        obj.lensing_model = lensing_model

        history_len = int(data["history_len"])
        obj.history = []
        for i in range(history_len):
            entry = {
                "z": jnp.asarray(data[f"h{i}_z"]),
                "u": jnp.asarray(data[f"h{i}_u"]),
            }
            for field in ("z_hat", "C", "K"):
                if bool(data[f"h{i}_{field}_none"]):
                    entry[field] = None
                else:
                    entry[field] = jnp.asarray(data[f"h{i}_{field}"])
            obj.history.append(entry)

        last = obj.history[-1] if obj.history else {}
        obj.z_hat = last.get("z_hat")
        obj.C     = last.get("C")
        obj.K     = last.get("K")

        print(f"EKI loaded from {path}  (n={obj.n}, {history_len} history entries)")
        return obj

    def animate_ensemble(self, param_names=None, true_params=None, save_path=None,
                         y_obs_ps=None, param_xlims=None, col_order=None, frames_dir=None):
        if param_names is None:
            param_names = [f"θ_{i}" for i in range(self.d)]

        # ── filter history to frames with finite parameter values ──────────────
        def _params_finite(entry):
            return bool(jnp.all(jnp.isfinite(jnp.asarray(entry["z"])[:, :self.d])))

        finite_history = [e for e in self.history if _params_finite(e)]
        n_iters = len(finite_history)
        if n_iters == 0:
            raise ValueError("No finite history entries to animate.")
        param_history = [np.array(jnp.asarray(e["z"])[:, :self.d]) for e in finite_history]

        # reorder columns for display if requested
        if col_order is not None:
            param_history = [p[:, col_order] for p in param_history]

        # ensemble power spectra: prefer z_hat (actual G(u)), fall back to z[:,d:]
        def _get_ps(entry):
            zh = entry.get("z_hat")
            if zh is not None:
                zh = jnp.asarray(zh)
                if jnp.all(jnp.isfinite(zh)):
                    return np.array(zh[:, self.d:])
            return np.array(jnp.asarray(entry["z"])[:, self.d:])

        ps_history = [_get_ps(e) for e in finite_history]   # list of (J, k)

        # ── x-limits: use provided limits or fall back to data range ───────────
        if param_xlims is not None:
            xlims = [tuple(param_xlims[i]) for i in range(self.d)]
        else:
            all_params = np.concatenate(param_history, axis=0)
            xlims = [(float(all_params[:, i].min()), float(all_params[:, i].max()))
                     for i in range(self.d)]

        # ── figure layout: d×d triangle plot + PS in top-right corner ──────────
        d = self.d
        fig, axes = plt.subplots(d, d, figsize=(3.5 * d, 3.5 * d),
                                 gridspec_kw={"hspace": 0.05, "wspace": 0.05})
        if d == 1:
            axes = np.array([[axes]])

        # top-right corner [0, d-1] → PS panel; rest of upper triangle → hidden
        ax_ps = axes[0, d - 1] if (y_obs_ps is not None and d >= 2) else None
        for row in range(d):
            for col in range(d):
                if col > row and not (row == 0 and col == d - 1 and ax_ps is not None):
                    axes[row, col].set_visible(False)

        # ── PS setup ───────────────────────────────────────────────────────────
        if ax_ps is not None:
            y_obs_ps_arr = np.array(y_obs_ps)
            if y_obs_ps_arr.ndim == 2:
                n_ps     = y_obs_ps_arr.shape[1]
                obs_mean = y_obs_ps_arr.mean(axis=0)
                obs_std  = y_obs_ps_arr.std(axis=0)
            else:
                n_ps     = len(y_obs_ps_arr)
                obs_mean = y_obs_ps_arr
                obs_std  = None
            ps_bins  = np.arange(n_ps)
            all_ps   = np.concatenate([p[:, :n_ps] for p in ps_history])
            pos_vals = all_ps[all_ps > 0]
            ps_lo    = float(pos_vals.min()) * 0.3 if len(pos_vals) else 1e-1
            ps_hi    = max(float(all_ps.max()) * 3, float(obs_mean.max()) * 3)
        else:
            n_ps = obs_mean = obs_std = ps_bins = ps_lo = ps_hi = None

        def update(frame):
            params = param_history[frame]   # (J, d) numpy array

            for row in range(d):
                for col in range(d):
                    ax = axes[row, col]
                    if col > row:
                        continue  # hidden or PS (handled below)
                    ax.cla()

                    if row == col:
                        # 1D histogram on diagonal
                        ax.hist(params[:, row], bins=np.linspace(*xlims[row], 21),
                                density=True, alpha=0.7, color="steelblue")
                        ax.axvline(float(params[:, row].mean()), color="red",
                                   linestyle="--", label="mean")
                        if true_params is not None:
                            ax.axvline(true_params[row], color="green",
                                       linestyle="-", label="truth")
                        ax.set_xlim(xlims[row])
                        ax.set_yticks([])
                        if row == 0:
                            ax.legend(fontsize=7)
                    else:
                        # 2D histogram below diagonal
                        ax.hist2d(params[:, col], params[:, row],
                                  bins=[np.linspace(*xlims[col], 21),
                                        np.linspace(*xlims[row], 21)],
                                  cmap="Blues")
                        if true_params is not None:
                            ax.plot(true_params[col], true_params[row], "r*",
                                    markersize=10, zorder=5)
                        ax.set_xlim(xlims[col])
                        ax.set_ylim(xlims[row])

                    # edge labels only
                    ax.set_ylabel(param_names[row] if (col == 0 and row > 0) else "")
                    ax.set_xlabel(param_names[col] if row == d - 1 else "")
                    if col > 0:
                        ax.set_yticklabels([])
                    if row < d - 1:
                        ax.set_xticklabels([])

            # power spectrum panel
            if ax_ps is not None:
                ax_ps.cla()
                ps = ps_history[frame][:, :n_ps]   # (J, n_ps)
                ax_ps.fill_between(ps_bins, ps.min(axis=0), ps.max(axis=0),
                                   alpha=0.3, color="steelblue", label="ensemble range")
                ax_ps.plot(ps_bins, ps.mean(axis=0), color="steelblue", lw=1.5,
                           label="ensemble mean")
                if obs_mean is not None:
                    ax_ps.plot(ps_bins, obs_mean, color="tomato", lw=1.5,
                               label="observed mean")
                    if obs_std is not None:
                        ax_ps.plot(ps_bins, obs_mean + 2 * obs_std, color="tomato",
                                   lw=1.0, linestyle="--", label="observed ±2σ")
                        ax_ps.plot(ps_bins, obs_mean - 2 * obs_std, color="tomato",
                                   lw=1.0, linestyle="--")
                ax_ps.set_yscale("log")
                ax_ps.set_ylim(10 ** 2.5, 10 ** 6.5)
                ax_ps.set_xlabel("bin index")
                ax_ps.set_ylabel("power")
                ax_ps.set_title("Power spectrum")
                ax_ps.legend(fontsize=7)

            fig.suptitle(f"EKI Ensemble — Iteration {frame}/{n_iters - 1}", fontsize=13)

        ani = animation.FuncAnimation(fig, update, frames=n_iters,
                                      interval=500, blit=False)
        if save_path is not None:
            ani.save(save_path, writer="pillow")
        else:
            plt.show()

        if frames_dir is not None:
            import os
            os.makedirs(frames_dir, exist_ok=True)
            for frame in range(n_iters):
                update(frame)
                fig.savefig(os.path.join(frames_dir, f"frame_{frame:02d}.png"),
                            dpi=150, bbox_inches="tight")

        return ani

    def discrepancy_stopping(self, tau=1.0, theory_model=None):
        if self.n == 0:
            return False
        u_mean = np.array(self.u)
        eval_model = theory_model if theory_model is not None else self.forward_model
        g_mean = jnp.asarray(eval_model(u_mean))
        residual = g_mean - self.y
        discrepancy = float(jnp.sqrt(residual @ jnp.linalg.solve(self.Gamma, residual)))
        print(f"[n={self.n}] discrepancy = {discrepancy:.4g}  (τ = {tau})")
        return discrepancy <= tau

    def compute_tau(self, fiducial_params, stochastic_n=1, theory_model=None):
        eval_model = theory_model if theory_model is not None else self.forward_model
        discrepancies = []
        for _ in range(stochastic_n):
            g_mean = jnp.asarray(eval_model(fiducial_params))
            residual = g_mean - self.y
            discrepancy = float(jnp.sqrt(residual @ jnp.linalg.solve(self.Gamma, residual)))
            discrepancies.append(discrepancy)
        mean_tau = np.mean(np.array(discrepancies))
        std_tau = np.std(np.array(discrepancies))
        return mean_tau, std_tau

    def invert(self, max_iter=10, timed=False, stopping_algo=None):
        pbar = tqdm(total=max_iter)
        i = 0
        iter_times = []
        while not stopping_algo(self) and not (jnp.isnan(self.u).any()) and i < max_iter:
            pbar.update()
            self.n += 1
            t0 = time.perf_counter() if timed else None
            self.prediction_step()
            self.analysis_step()
            self.compute_mean_parameters()
            if timed:
                iter_times.append(time.perf_counter() - t0)
            self.history.append({"z": self.z,
                                 "u": self.u,
                                 "z_hat": self.z_hat,
                                 "C": self.C,
                                 "K": self.K})
            i += 1
        pbar.close()
        if timed and iter_times:
            total = sum(iter_times)
            avg = total / len(iter_times)
            print(f"Total time: {total:.3f}s  |  Time per iteration: {avg:.3f}s  ({len(iter_times)} iterations)")
        return self.u
