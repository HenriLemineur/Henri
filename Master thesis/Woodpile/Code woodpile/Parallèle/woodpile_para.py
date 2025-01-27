

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.expansion import Expansion
from khepri.layer import Layer
from khepri.misc import coords
from khepri.alternative import redheffer_product
from khepri.factory import make_woodpile
from khepri.tools import block2dense
from khepri.eigentricks import scattering_splitlr
from khepri.misc import str2linspace_args
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from functools import partial

from multiprocessing import Pool
from itertools import product


"""
    Parameters
    From 'A three-dimentional photonic crystal operating at infrared wavelengths
    SY. Lin and JG. Fleming. Letters to Nature 1998
"""
N = 512
NUM_THREADS = 8
polarization = (1, 0)  # norm
theta = 0
phi = 0

rods_height = 1.414 / 4
rods_shift = 0.5
rods_eps = 3.6**2
rods_w = 0.28
a = 1.414

def worker_twisted(config, return_cl=False, fields=False):
    twist_angle, frequency, index_contrast, pwx, kpx, kpy = config
    pw = (pwx, 1)
    canvas_size = (N, 1)
    
    background_eps = 1
    median_eps = (background_eps + rods_eps) / 2
    diff_eps = (rods_eps - background_eps) / 2
    background_eps2 = median_eps + index_contrast * (-diff_eps)
    rods_eps2 = median_eps + index_contrast * (diff_eps)

    pattern = Drawing(canvas_size, background_eps2)
    pattern.rectangle((0, 0), (rods_w, 1), rods_eps2)
    
    pattern2 = Drawing(canvas_size, background_eps)
    pattern2.rectangle((rods_shift, 0), (rods_w, 1), rods_eps) # These rods do not change
    pattern2.rectangle((-rods_shift, 0), (rods_w, 1), rods_eps)
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    e2.rotate(twist_angle)

    """
        Define the crystal layers. (Be careful that layers with different twist angles are different objects.)
    """
    twcl = Crystal.from_expansion(e1 + e2)
    etw = twcl.expansion
    twcl.add_layer("Sref",   Layer.half_infinite(etw, "reflexion", 1), False)
    twcl.add_layer("1",      Layer.pixmap(e1, pattern.canvas(),  rods_height), True)
    twcl.add_layer("2",      Layer.pixmap(e2, pattern.canvas(),  rods_height), True)
    twcl.add_layer("3",      Layer.pixmap(e1, pattern2.canvas(), rods_height), True)
    twcl.add_layer("4",      Layer.pixmap(e2, pattern2.canvas(), rods_height), True)
    twcl.add_layer("Strans", Layer.half_infinite(etw, "transmission", 1), False)

    """
        Define the device and solve.
    """
    device = ["1", "2", "3", "4", "1", "2", "3", "4", "1", "2", "3", "4"]
    twcl.set_device(device, [fields] * len(device))
    twcl.set_source(a / frequency, polarization[0], polarization[1], theta, phi, kp=(kpx, kpy))
    twcl.solve()

    if not return_cl:
        return twcl.poynting_flux_end()
    else:
        return twcl


def worker_notwist(config, return_cl=False, fields=False, return_det=False):
    frequency, kp = config
    wl = 1 / frequency
    cl = make_woodpile(rods_w, rods_eps, rods_shift, rods_height, (pwx, pwx))
    cl.set_source(wl, polarization[0], polarization[1], 0, 0, kp=kp)
    cl.solve()

    if return_cl:
        return cl
    elif return_det:
        Sl, Sr = scattering_splitlr(cl.Stot)
        Sl, Sr = block2dense(Sl), block2dense(Sr)
        return *cl.poynting_flux_end(), np.linalg.det(Sl-Sr)
    else:
        return cl.poynting_flux_end()

if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please provide subprogram. ex: ct"
    if sys.argv[1] == "ct":
        assert len(sys.argv) == 7, "call ct <freqs> <angles> <contrasts> <pws> <filename>"
        filename = str(sys.argv[6])
        assert not os.path.isfile(filename), "File exists."
        
        kpoints = ([0], [0])  # List of k-vectors to evaluate for band diagram for example.
        angles = np.deg2rad(np.linspace(*str2linspace_args(sys.argv[2])))  # -90, 90
        frequencies = np.linspace(*str2linspace_args(sys.argv[3]))  # 0.9/a, 1.2/a # 0.3, 0.7
        contrasts = np.linspace(*str2linspace_args(sys.argv[4]))
        pwxs = np.linspace(*str2linspace_args(sys.argv[5])).astype(int)

        sweep_variables = angles, frequencies, contrasts, pwxs, *kpoints
        configs = list(product(*sweep_variables))
        
        # Using ThreadPool instead of Pool
        with ThreadPool(NUM_THREADS) as p:
            # Run the parallel computation and track progress with tqdm
            RT = list(tqdm(p.imap(worker_twisted, configs), total=len(configs)))
        
        # Save the results
        tqdm.write(f"Saving results to {filename}...")
        np.savez_compressed(filename, RT=np.array(RT).reshape(tuple((len(x) for x in sweep_variables))+(2,)), configs=configs, args=sys.argv[1:])
        tqdm.write("Save completed.")

    elif sys.argv[1] == "ctbd":
        assert len(sys.argv) > 1
        a = 1.414
        twist_angle  = float(sys.argv[2])
        frequencies = np.linspace(0.3/a, 0.7/a, NF)
        M = 91
        kpath = [ (kx, 0) for kx in np.linspace(0, 0.98*np.pi, M) ]
        worker_twisted_angle = partial(worker_twisted, angle_deg=twist_angle)
        with Pool(NUM_THREADS) as p:
            configs = list(product(frequencies, kpath))
            RT = list(tqdm(p.imap(worker_twisted_angle, configs), total=len(configs)))
        np.savez_compressed(f"data/twpbd_2_{pwx}_{float(twist_angle):0.2f}.npz", RT=RT, F=frequencies, A=kpath, twist_angle=twist_angle, pwx=pwx)

    elif sys.argv[1] == "c":
        a = 1.414
        freqs = np.linspace(0.35/a, 0.65/a, NF)
        kp = (0, 0)
        RT = list()
        for i, f in enumerate(tqdm(freqs)):
            RT.append(worker_notwist((f, kp)))
        np.savez_compressed("woodpile2.npz", RT=RT, F=a*freqs)

    elif sys.argv[1] == "cbd":
        a = 1.414
        NF, M = 201, 91
        frequencies = np.linspace(0.01/a, 0.65/a, NF)
        from itertools import product
        kpath = [ (kx, 0) for kx in np.linspace(0, 0.99*np.pi, M) ]
    
        worker_instance = partial(worker_notwist, return_det=True)
        with Pool(NUM_THREADS) as p:
            configs = list(product(frequencies, kpath))
            scalar = list(tqdm(p.imap(worker_instance, configs), total=len(configs)))
        np.savez_compressed("woodpile_untwisted_bd.npz", RT=scalar, F=frequencies, A=kpath)

    elif sys.argv[1] == "pt":
        data = np.load(sys.argv[2])
        axisx = int(sys.argv[3]) # x variable ?
        axisy = int(sys.argv[4]) # y variable ?
        axisd = int(sys.argv[5]) # data variable ? (R or T)
        swap = False
        if axisx < axisy:
            axisx, axisy = axisy, axisx
            swap = True
        RT = data["RT"]
        for i in reversed(range(6)):
            if i != axisx and i != axisy:
                RT = RT.take(axis=i, indices=0)
        if swap:
            RT = np.transpose(RT, (1,0,2))
        
        T = RT[:,:,axisd] # only keep either R or T
        extent = data["configs"][axisx][[0,-1]], data["configs"][axisy][[0,-1]]

        plt.imshow(np.abs(T), origin="lower", extent=extent, aspect="auto")
        plt.show()
