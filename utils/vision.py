from .numpy import to_np
import numpy as np


def to_camera(X, calibration, referencial, ret_homogeneous=False):
    """ X must be a matriz 3xN with x, y and z coordinates on each row """
    if X.shape[0] != 3:
        return np.array([])

    camera = calibration.id
    ext = list(filter(lambda x: getattr(x, 'from') == referencial and getattr(x, 'to') == camera, calibration.extrinsic))
    if len(ext) == 0:
        return np.array([])
    
    X_ = np.vstack([X, np.ones(X.shape[1])])
    RT = to_np(ext[0].tf)
    x = np.asarray(np.matmul(RT, X_))
    x[0:2,:] = x[0:2,:]/x[2,:]
    
    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
    d = to_np(calibration.distortion)
    x[0,:] = x[0,:]*(1 + d[0]*r + d[1]*r*r + d[4]*r*r*r) + 2*d[2]*x[0,:]*x[1,:] + d[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + d[0]*r + d[1]*r*r + d[4]*r*r*r) + 2*d[3]*x[0,:]*x[1,:] + d[2]*(r + 2*x[1,:]*x[1,:])

    K = to_np(calibration.intrinsic)
    x[2,:] = 1.0
    x = np.matmul(K, x[0:3,:])
    if ret_homogeneous:
        return x[0:3,:]
    else:
        return x[0:2,:]


def resolution_validation(joints, calibration):
    w, h = calibration.resolution.width, calibration.resolution.height
    if joints.shape[0] != 2:
        return None
    return np.logical_not(np.logical_or(joints[0,:] > w, joints[1,:] > h))