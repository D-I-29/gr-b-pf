# %%
#@title load package
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import grb.particle_filter
import grb.particle_filter_tv
import argparse
from tqdm import tqdm

importlib.reload(grb.particle_filter)
importlib.reload(grb.particle_filter_tv)
from grb.particle_filter import GRParticleFilter
from grb.particle_filter_tv import TimeVariantGRParticleFilter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='input data path')
    parser.add_argument('--num_particle', type=int, default=10_000, help='number of particle for particle filter')
    parser.add_argument('--m_lower', type=float, default=3.0, help='lower bound of magnitude')
    args = parser.parse_args()

    if args.data is None:
        raise ValueError('input data path')

    dat = pd.read_csv(args.data)
    dat['date_time'] = pd.to_datetime(dat['date_time'])
    dat = dat.loc[dat['magnitude'] >= args.m_lower]
    print(f"event size: {dat.shape[0]}")
    print(f"minimum magnitude: {dat['magnitude'].min()}")
    print(f"maximum magnitude: {dat['magnitude'].max()}")

    np.random.seed(123)
    num_particle = args.num_particle
    initial_cut_len = 1000
    log_sig_log_beta_grid = np.linspace(-6, -1, num=10)

    model = GRParticleFilter(
        num_particle=num_particle,
        log_sig_log_beta=np.nan,
        m_lower=args.m_lower,
    )
    res_tuning = model.tuning_hyper_parameter(
        m=dat['magnitude'].values, 
        date_time=dat['date_time'].values,
        initial_cut_len=initial_cut_len,
        method='optimizer', 
        log_sig_log_beta_grid=log_sig_log_beta_grid, 
    )
    res_df_gr = model.batch(m=dat['magnitude'].values, date_time=dat['date_time'].values, ql=0.6, qu=0.7, ql2=0.55, qu2=0.65)

    os.makedirs('./result', exist_ok=True)
    res_df_gr.to_csv('./result/result.csv', index=False)
