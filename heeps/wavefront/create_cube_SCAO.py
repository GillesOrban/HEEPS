from heeps.util.img_processing import crop_img, resize_img
from heeps.util.multiCPU import multiCPU
from heeps.util.notify import notify
import os
import numpy as np
from astropy.io import fits

# email when finished
conf = {}
conf['send_to'] = 'gorban@uliege.be'
conf['send_message'] = 'cube calculation finished OK.'

# useful inputs
<<<<<<< HEAD
tag = 'Cbasic_v3_20210223'
prefix = 'Residual_phase_screen_'#'tarPhase_1hr_100ms_'
suffix = 'ms'
duration = 600 #3600
samp = 100 #300
=======
os.chdir(os.path.normpath(os.path.expandvars('$HOME/heeps_metis/input_files')))
tag = 'Cfull_20211007'#'Cbasic_20201130'
prefix = 'Residual_phase_screen_'#'tarPhase_1hr_100ms_'
suffix = 'ms'
<<<<<<< HEAD
duration = 600#3600
samp = 100#300
>>>>>>> 59a3165 (NCPA chromatic scaling)
start = 2101#0#1001
=======
duration = 3600#600#
samp = 300#100#
start = 2101
>>>>>>> d75245c (Cfull upgrade)
nimg = 720
band = 'L'
npupil = 285#720
pad_frame = False
<<<<<<< HEAD
savename = 'cube_%s_%ss_%sms_scao_only_%s.fits'%(tag, duration, samp, npupil)

#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-10-16T10:25:14/residualPhaseScreens'
#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-11-05T12:40:27/residualPhaseScreens'
input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_v3_2021-02-23T10:29:47/residualPhaseScreens'
output_folder = '/mnt/disk4tb/METIS/METIS_CBASIC_CUBES'
=======
savename = 'cube_%s_%ss_%sms_0piston_meters_scao_only_%s_%s.fits'%(tag, duration, samp, band, npupil)
#savename = 'cube_%s_%ss_%sms_0piston_meters_scao_only_%s_WVseeing.fits'%(tag, duration, samp, npupil)

#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-10-16T10:25:14/residualPhaseScreens'
#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-11-05T12:40:27/residualPhaseScreens'
#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-11-30T20:52:24/residualPhaseScreens'
#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_uncorrected_2021-06-01T12:02:36/residualPhaseScreens'
<<<<<<< HEAD
input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cfull_noWtt_2021-10-04T12:28:42/residualPhaseScreens'
output_folder = 'wavefront/cbasic'
>>>>>>> 59a3165 (NCPA chromatic scaling)
=======
input_folder = '/mnt/disk12tb/Users/gorban/METIS/METIS_COMPASS/gorban_metis_baseline_Cfull_noWtt_2021-10-07T09:00:32/residualPhaseScreens'
output_folder = 'wavefront/cfull'
>>>>>>> d75245c (Cfull upgrade)
cpu_count = None

# mask
mask = fits.getdata(os.path.join(input_folder, 'Telescope_Pupil.fits'))
# mask = crop_img(mask, nimg)
mask_pupil = np.rot90(resize_img(mask, npupil))
fits.writeto(os.path.join(output_folder, 'mask_%s_%s_%s.fits'%(tag, band, npupil)), np.float32(mask_pupil), overwrite=True)

# filenames
nframes = len([name for name in os.listdir(input_folder) if name.startswith(prefix)])
nframes = 12000
print('%s frames'%nframes)
frames = [str(frame).zfill(6) if pad_frame is True else str(frame) \
    for frame in range(start, start + nframes*samp, samp)]
filenames = np.array([os.path.join(input_folder, '%s%s%s.fits'%(prefix, frame, suffix)) \
    for frame in frames])

def remove_piston(filename):
    data = np.float32(fits.getdata(filename))
#     data = crop_img(data, nimg)
    data -= np.mean(data[mask!=0])      # remove piston
    data[mask==0] = 0
    data = resize_img(data, npupil)
    data = np.rot90(data) * 1e-6        # rotate, convert to meters
    return data

# create cube
cube = multiCPU(remove_piston, posvars=[filenames], case='create cube SCAO', cpu_count=cpu_count)
print(cube.shape)
<<<<<<< HEAD
hdr = fits.Header()
hdr.set('SCAO_DIR', input_folder)
hdr.set('UNIT', 'meters')
hdr.set('PISTON', 'removed')
hdr.set('DURATION [s]', duration)
hdr.set('SAMPLING [ms]', samp)
# ** add here other important variables used to create the cube ***
fits.writeto(os.path.join(output_folder, savename), np.float32(cube), hdr, overwrite=True)

# send email when simulation finished
print(time.strftime("%Y-%m-%d %H:%M:%S: " + "%s\n"%conf['send_message'], time.localtime()))
if conf['send_to'] is not None:
    os.system('echo "%s" | mail -s "%s" %s'%(conf['send_message'], \
            conf['send_subject'], conf['send_to']))
=======
fits.writeto(os.path.join(output_folder, savename), np.float32(cube), overwrite=True)
notify(conf['send_message'], conf['send_to'])
>>>>>>> ff9bdf6 (major update with new utility: multiCPU)
