from heeps.util.img_processing import crop_img, pad_img #, resize_img
import os
import numpy as np
from astropy.io import fits
import multiprocessing as mpro
from functools import partial
import time


#--------------------------------------------------#
import numpy
from scipy.interpolate import interp2d,RectBivariateSpline, griddata

def zoomWithMissingData(data, newSize,
                        method='linear',
                        non_valid_value=numpy.nan):
    '''
    Zoom 2-dimensional or 3D arrays using griddata interpolation.
    This allows interpolation over unstructured data, e.g. interpolating values
    inside a pupil but excluding everything outside.
    See also DM.CustomShapes.

    Note that it can be time consuming, particularly on 3D data

    Parameters
    ----------
    data : ndArray
        2d or 3d array. If 3d array, interpolate by slices of the first dim.
    newSize : tuple
        2 value for the new array (or new slices) size.
    method: str
        'linear', 'cubic', 'nearest'
    non_valid_value: float
        typically, NaN or 0. value in the array that are not valid for the
        interpolation.

    Returns
    -------
    arr : ndarray
        of dimension (newSize[0], newSize[1]) or
        (data.shape[0], newSize[0], newSize[1])
    '''
    if len(data.shape) == 3:
        arr = data[0, :, :]
    else:
        assert len(data.shape) == 2
        arr = data

    Nx = arr.shape[0]
    Ny = arr.shape[1]
    coordX = (numpy.arange(Nx) - Nx / 2. + 0.5) / (Nx / 2.)
    coordY = (numpy.arange(Ny) - Ny / 2. + 0.5) / (Ny / 2.)
    Nx = newSize[0]
    Ny = newSize[1]
    ncoordX = (numpy.arange(Nx) - Nx / 2. + 0.5) / (Nx / 2.)
    ncoordY = (numpy.arange(Ny) - Ny / 2. + 0.5) / (Ny / 2.)

    x, y = numpy.meshgrid(coordX, coordY)
    xnew, ynew = numpy.meshgrid(ncoordX, ncoordY)

    if len(data.shape) == 2:
        idx = ~(arr == non_valid_value)
        znew = griddata((x[idx], y[idx]), arr[idx], (xnew, ynew),
                        method=method)
        return znew
    elif len(data.shape) == 3:
        narr = numpy.zeros((data.shape[0], newSize[0], newSize[1]))
        for i in range(data.shape[0]):
            arr = data[i, :, :]
            idx = ~(arr == non_valid_value)
            znew = griddata((x[idx], y[idx]), arr[idx], (xnew, ynew),
                            method=method)
            narr[i, :, :] = znew
        return narr
#--------------------------------------------------#
    
    
# email when finished
conf = {}
conf['send_to'] = 'gorban@uliege.be'
conf['send_message'] = 'cube calculation finished OK.'
conf['send_subject'] = 'fenrir noreply'

# useful inputs
tag = 'Cbasic_20210219'
prefix = 'Residual_phase_screen_'#'tarPhase_1hr_100ms_'
suffix = 'ms'
duration = 600 #3600
samp = 100 #300
start = 2101#0#1001
nimg = 720
npupil = 285
pad_frame = False
savename = 'cube_%s_%ss_%sms_0piston_meters_scao_only_%s_GOXinterp.fits'%(tag, duration, samp, npupil)

#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-10-16T10:25:14/residualPhaseScreens'
#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-11-05T12:40:27/residualPhaseScreens'
input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_v2_2021-02-18T13:00:36/residualPhaseScreens'
output_folder = '/mnt/disk4tb/METIS/METIS_CBASIC_CUBES'
cpu_count = None

# mask
mask = fits.getdata(os.path.join(input_folder, 'Telescope_Pupil.fits'))
#>> GOX
# mask = crop_img(mask, nimg)
# mask_pupil = np.rot90(resize_img(mask, npupil))
mask_nan = np.copy(mask)
mask_nan[mask_nan == 0] = np.nan
mask_pupil = zoomWithMissingData(mask_nan, [npupil,npupil], method='nearest', non_valid_value=np.nan)
mask_pupil = np.rot90(mask_pupil)
# << GOX

fits.writeto(os.path.join(output_folder, 'mask_%s_%s.fits'%(tag, npupil)), np.float32(mask_pupil), overwrite=True)

# filenames
nframes = len([name for name in os.listdir(input_folder) if name.startswith(prefix)])
frames = [str(frame).zfill(6) if pad_frame is True else str(frame) \
    for frame in range(start, start + nframes*samp, samp)]
filenames = np.array([os.path.join(input_folder, '%s%s%s.fits'%(prefix, frame, suffix)) \
    for frame in frames])

def remove_piston(filename):
    data = np.float32(fits.getdata(filename))
#     data = crop_img(data, nimg)
    data -= np.mean(data[mask!=0])      # remove piston
    #>> GOX
#     data[mask==0] = 0
#     data = resize_img(data, npupil)
    data[mask == 0 ] = np.nan
    data = zoomWithMissingData(data, [npupil,npupil], method='nearest', non_valid_value=np.nan)
    #<< GOX
    data = np.rot90(data) * 1e-6        # rotate, convert to meters
    return data

if cpu_count == None:
    cpu_count = mpro.cpu_count() - 1
p = mpro.Pool(cpu_count)
cube = np.array(p.starmap(partial(remove_piston), zip(filenames)))
p.close()
p.join()

# save cube
print(cube.shape)
fits.writeto(os.path.join(output_folder, savename), np.float32(cube), overwrite=True)

# send email when simulation finished
print(time.strftime("%Y-%m-%d %H:%M:%S: " + "%s\n"%conf['send_message'], time.localtime()))
if conf['send_to'] is not None:
    os.system('echo "%s" | mail -s "%s" %s'%(conf['send_message'], \
            conf['send_subject'], conf['send_to']))
