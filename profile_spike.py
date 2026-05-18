#!/usr/bin/env python3
import os
import time
import psutil
import argparse

def print_rss(tag):
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / (1024.0**2)
    print(f"{time.strftime('%H:%M:%S')} {tag}: {rss:.1f} MB", flush=True)

def main(nk, dim, method):
    print_rss('start')

    # import local package
    import importlib
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), 'code'))

    print_rss('before imports')
    from scripts.lattice import LATTICE
    import scripts.obs as obs
    print_rss('after imports')

    latt = LATTICE(t=1.0, tp=0.0, dim=dim)
    print_rss('after LATTICE init')

    print_rss(f'before get_e_k nk={nk}')
    latt.get_e_k(nk)
    print_rss('after get_e_k')

    # phase_k and fine grid may allocate
    try:
        print_rss(f'before get_phase_k nk={nk}')
        latt.get_phase_k(nk)
        print_rss('after get_phase_k')
    except Exception as e:
        print('get_phase_k failed:', e)

    # call the expensive invchi function
    try:
        print_rss('before get_invchi0_min')
        # use safe defaults: mu=0, beta=10
        q_min, invchi0_min, invchi0_grid = obs.get_invchi0_min(latt, mu=0.0, beta=10.0, nk=nk, method=method, niw=1, S_iwk=None, refine=False, niw_extr=False)
        print('q_min, invchi0_min:', q_min, invchi0_min)
        print_rss('after get_invchi0_min')
    except Exception as e:
        print('get_invchi0_min failed:', repr(e))
        print_rss('after get_invchi0_min (failed)')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nk', type=int, default=150, help='grid size per dim')
    p.add_argument('--dim', type=int, default=3)
    p.add_argument('--method', type=str, default='matsubara', choices=['matsubara','lindhard'])
    args = p.parse_args()
    main(args.nk, args.dim, args.method)
