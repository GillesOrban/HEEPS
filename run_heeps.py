#!/home/gorban/anaconda3/bin/python3.7

# ===================== #
# End-to-end simulation #
# ===================== #

import heeps
import sys

def run_heeps(savename='ADI_contrast_curve.png'):

    # 1. Create a config dictionary with your simulation parameters in read_config
    conf = heeps.config.read_config(verbose=False)

    # 2. Update config parameters. The following parameters 
    # will be updated to match the selected spectral band and HCI mode:
    #   lam, pscale, flux_star, flux_bckg, ls_dRspi, ls_dRint, npupil, ndet,
    #   ravc_t, ravc_r
    conf = heeps.config.update_config(saveconf=True, verbose=True, **conf) 

    # 3. Load entrance pupil, and create 'wavefront' object
    wf = heeps.pupil.pupil(savefits=True, verbose=True, **conf)

    # 4. Load wavefront errors
    phase_screens, amp_screens, tiptilts, misaligns = heeps.wavefront.load_errors(verbose=True, **conf)

    # 5. Propagate one frame of offaxis psf (i.e. planet)
    psf = heeps.wavefront.propagate_one(wf, onaxis=False, savefits=True, verbose=True, **conf)

    # 6. Propagate cube of onaxis psfs (i.e. star)
    psfs = heeps.wavefront.propagate_cube(wf, phase_screens=phase_screens, \
        amp_screens=amp_screens, tiptilts=tiptilts, misaligns=misaligns, onaxis=True, \
        savefits=True, verbose=True, **conf)

    # 7. Produce 5-sigma sensitivity (contrast) curves for each set of PSFs (modes, bands)
    sep, sen = heeps.contrast.adi_one(savepsf=True, savefits=True, verbose=True, **conf)

    # 8. Create a figure 
    if savename != '':
        import matplotlib.pyplot as plt
        from matplotlib.pylab import ScalarFormatter
        plt.figure(figsize=(12,4))
        plt.grid(True), plt.grid(which='minor', linestyle=':')
        plt.loglog(), plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.xlabel('Angular separation $[arcsec]$')
        plt.ylabel('5-$\sigma$ sensitivity (contrast)')
        label = '%s-band %s'%(conf['band'], conf['mode'])
        plt.plot(sep, sen, label=label, marker='d', markevery=0.12, markersize=4)
        plt.legend()
        plt.xticks([0.02, 0.1, 0.5, 1])
        plt.xlim(0.02,1)
        plt.ylim(1e-6,1e-2)
        plt.savefig('%s/%s'%(conf['dir_output'], savename), dpi=300, transparent=True)

if __name__ == "__main__":
    '''
    Terminal command line example
    > python run_heeps.py test.png
    '''
    if len(sys.argv) > 1:
        run_heeps(sys.argv[1])
    else:
        run_heeps()