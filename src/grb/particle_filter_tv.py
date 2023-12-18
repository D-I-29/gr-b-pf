# load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from numba import jit
from scipy.special import logsumexp
from tqdm import tqdm
from grb.truncated_normal import TruncatedNormal
import torch


# jit compile core function
def gen_truncnorm(left_bound, right_bound, mu, sig, size):
    truncated_normal = TruncatedNormal(loc=torch.tensor(mu, dtype=torch.float64), scale=torch.tensor(sig, dtype=torch.float64), a=left_bound, b=right_bound)
    r = torch.flatten(truncated_normal.rsample(sample_shape=(1,))).detach().numpy().copy()
    return r

def gen_truncnorm_uniform(left_bound, right_bound, mu, sig, size, eps_rate):
    r = gen_truncnorm(left_bound, right_bound, mu, sig, size)
    n_uniform = int(size * eps_rate)
    ind = np.random.choice(np.arange(size), size=n_uniform, replace=False)
    r[ind] = np.random.uniform(low=left_bound, high=right_bound, size=n_uniform)
    return r

@jit(nopython=True)
def normalize(loglikelihood):
    max_ind = np.argmax(loglikelihood)
    tmp = np.sum(np.exp(loglikelihood - loglikelihood[max_ind])) - np.exp(0)
    return loglikelihood - loglikelihood[max_ind] - np.log1p(tmp)


@jit(nopython=True)
def sys_resampling(normalized_loglikelihood):
    N = len(normalized_loglikelihood)
    p = np.exp(normalized_loglikelihood)
    ind_to_sort = np.argsort(p)
    sorted_p = p[ind_to_sort]
    cump = np.cumsum(sorted_p)
    r = np.random.rand(N)
    ind = np.searchsorted(cump, r)
    return ind_to_sort[ind]

@jit(nopython=True)
def calc_loglikelihood_tGR(m, beta, m_lower, m_upper):
    return np.log(beta) - beta * (m - m_lower) - np.log(1 - np.exp(-beta * (m_upper - m_lower)))


# time varying gr b est particle filler
class TimeVariantGRParticleFilter:
    def __init__(self, num_particle, sig_log_sig_log_beta, log_sig_log_beta_lw, log_sig_log_beta_up, m_lower, eps_rate, type='GR', m_upper=None):
        self.rng = np.random.default_rng(seed=123)
        self.num_particle = num_particle
        self.type = type
        self.m_lower = m_lower
        self.m_upper = m_upper
        self.dim = 2
        self.loglikelihood = np.zeros(self.num_particle)
        self.id_log_beta = 0
        self.id_log_sig_log_beta = 1
        self.log_sig_log_beta_lw = log_sig_log_beta_lw
        self.log_sig_log_beta_up = log_sig_log_beta_up
        self.sig_log_sig_log_beta = sig_log_sig_log_beta
        self.eps_rate = eps_rate
        self.x = np.zeros((self.dim, self.num_particle))
        self.xn = np.zeros((self.dim, self.num_particle))
        if self.type == 'tGR' and m_upper is None:
            raise ValueError('incorrect argument in m_upper')

        # init state
        self.x[self.id_log_beta, :] = self.rng.normal(loc=0, scale=1 * np.log(10), size=self.num_particle)
        self.x[self.id_log_sig_log_beta, :] = self.rng.uniform(low=log_sig_log_beta_lw, high=log_sig_log_beta_up, size=self.num_particle)

    def updateState(self):
        self.xn[self.id_log_sig_log_beta, :] = gen_truncnorm_uniform(
            left_bound=self.log_sig_log_beta_lw,
            right_bound=self.log_sig_log_beta_up,
            mu=self.x[self.id_log_sig_log_beta, :],
            sig=self.sig_log_sig_log_beta,
            size=self.num_particle,
            eps_rate=self.eps_rate
        )
        self.xn[self.id_log_beta, :] = self.rng.normal(loc=self.x[self.id_log_beta, :], scale=np.exp(self.xn[self.id_log_sig_log_beta, :]), size=self.num_particle)

    def resampling(self, m):
        beta = np.exp(self.xn[self.id_log_beta, :])
        if self.type == 'GR':
            self.loglikelihood = stats.expon.logpdf(m - self.m_lower, scale=1 / beta)
        elif self.type == 'tGR':
            self.loglikelihood = calc_loglikelihood_tGR(m=m, beta=beta, m_lower=self.m_lower, m_upper=self.m_upper)
        normalized_loglikelihood = normalize(self.loglikelihood)
        ind = sys_resampling(normalized_loglikelihood)
        self.x = self.xn[:, ind]

    def predict(self, l=0.025, u=0.975, l2=0.16, u2=0.84, n_block=None):
        xn_log_beta = self.rng.normal(loc=self.x[self.id_log_beta, :], scale=np.exp(self.x[self.id_log_sig_log_beta, :]), size=self.num_particle)
        beta = np.exp(xn_log_beta)
        if self.type == 'GR':
            m = np.random.exponential(scale=1/beta, size=self.num_particle) + self.m_lower
        elif self.type == 'tGR':
            m_lower = self.m_lower
            m_upper = self.m_upper
            u_rnd = np.random.uniform(size=self.num_particle)
            m = m_lower - (1/beta) * np.log(1 - (1 - np.exp(-beta * (m_upper -m_lower))) * u_rnd)
        ql, med, qu, ql2, qu2 = np.quantile(m, q=[l, 0.5, u, l2, u2])
        m_blocks = np.random.choice(m, size=int(self.num_particle/10) * n_block, replace=True).reshape(-1, n_block)
        max_m = np.max(m_blocks, axis=1)
        max_m_ql, max_m_med, max_m_qu, max_m_ql2, max_m_qu2 = np.quantile(max_m, q=[l, 0.5, u, l2, u2])
        return ql, med, qu, max_m_ql, max_m_med, max_m_qu, ql2, qu2, max_m_ql2, max_m_qu2


    def calc_average_loglikelihood(self):
        return logsumexp(self.loglikelihood) - np.log(self.num_particle)

    def batch(self, m, date_time, ql=0.025, qu=0.975, ql2=0.16, qu2=0.84, predict=True, n_eq_next_pred=None):
        N = len(m)
        dat = []
        beta = []
        b = []
        m_pred = []
        max_m_pred = []
        log_sig_log_beta = []
        average_loglikelihood = []
        for i in tqdm(range(2, N)):

            self.updateState()
            self.resampling(m=m[i])
            beta_ql, beta_med, beta_qu, beta_ql2, beta_qu2 = np.quantile(np.exp(self.x[self.id_log_beta, :]), q=[0.25, 0.5, 0.75, 0.025, 0.975])
            log_sig_log_beta_ql, log_sig_log_beta_med, log_sig_log_beta_qu = np.quantile(self.x[self.id_log_sig_log_beta, :], q=[0.25, 0.5, 0.75])
            if predict:
                if n_eq_next_pred is None:
                    m_ql, m_med, m_qu, max_m_ql, max_m_med, max_m_qu, m_ql2, m_qu2, max_m_ql2, max_m_qu2 = self.predict(l=ql, l2=ql2, u=qu, u2=qu2, n_block=100)
                else:
                    m_ql, m_med, m_qu, max_m_ql, max_m_med, max_m_qu, m_ql2, m_qu2, max_m_ql2, max_m_qu2 = self.predict(l=ql, l2=ql2, u=qu, u2=qu2, n_block=n_eq_next_pred[i])
            else:
                m_ql, m_med, m_qu, max_m_ql, max_m_med, max_m_qu, m_ql2, m_qu2, max_m_ql2, max_m_qu2 = [np.nan] * 10
            dat.append(
                {
                    'm': m[i],
                    'date_time': date_time[i]
                }
            )
            beta.append(
                {
                    'beta_ql': beta_ql,
                    'beta_med': beta_med,
                    'beta_qu': beta_qu,
                    'beta_ql2': beta_ql2,
                    'beta_qu2': beta_qu2,

                }
            )
            log_sig_log_beta.append(
                {
                    'log_sig_log_beta_ql': log_sig_log_beta_ql,
                    'log_sig_log_beta_med': log_sig_log_beta_med,
                    'log_sig_log_beta_qu': log_sig_log_beta_qu,
                }
            )
            b.append(
                {
                    'b_ql': beta_ql / np.log(10),
                    'b_med': beta_med / np.log(10),
                    'b_qu': beta_qu / np.log(10),
                    'b_ql2': beta_ql2 / np.log(10),
                    'b_qu2': beta_qu2 / np.log(10)
                }
            )
            m_pred.append(
                {
                    'm_ql': m_ql,
                    'm_med': m_med,
                    'm_qu': m_qu,
                    'm_ql2': m_ql2,
                    'm_qu2': m_qu2
                }
            )
            max_m_pred.append(
                {
                    'max_m_ql': max_m_ql,
                    'max_m_med': max_m_med,
                    'max_m_qu': max_m_qu,
                    'max_m_ql2': max_m_ql2,
                    'max_m_qu2': max_m_qu2
                }
            )

            average_loglikelihood.append(self.calc_average_loglikelihood())
        self.average_loglikelihood = average_loglikelihood
        res_df = pd.concat(
            [
                pd.DataFrame(dat),
                pd.DataFrame(beta),
                pd.DataFrame(log_sig_log_beta),
                pd.DataFrame(b),
                pd.DataFrame(m_pred),
                pd.DataFrame(max_m_pred),
                pd.Series(average_loglikelihood)
            ],
            axis=1
        )
        res_df = res_df.rename(columns={res_df.columns[-1]: 'ave_ll'})
        self.res_df = res_df
        return res_df

    def tuning_hyper_parameter(self):
        pass