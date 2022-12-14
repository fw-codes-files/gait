from typing import Dict, List, Tuple
import numpy as np
import os
import cv2 as cv
from tqdm import tqdm
from itertools import chain


def d_print(*args):
    if os.getlogin() == 'flyinghu':
        print(*args)

def cross_ten(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def expm(X):  # % expm
    D, V = np.linalg.eig(X)
    return V @ np.diag(np.exp(D)) @ np.linalg.inv(V)  #% V*diag(exp(diag(D)))/V


def lie_log(R):
    D, V = np.linalg.eig(R)  # % [V,D] = eig(R),
    theta = np.max(np.angle(D))  # % [theta,i] = max(angle(diag(D))),
    k = np.argmin(np.abs(D-1))  # % [   ~ ,k] = min(abs(diag(D)-1)),
    v1 = theta * V[:, k]  # % theta*V(:,k)
    v2 = -theta * V[:, k]  # % -theta*V(:,k)
    R1 = expm(cross_ten(v1))  # % expm(cross_ten( v1 )),
    R2 = expm(cross_ten(v2))  # % expm(cross_ten( v2 )),
    # % if max(angle(eig(R * R2'))) > max(angle(eig(R * R1')))
    if np.max(np.angle(np.linalg.eig(R @ np.transpose(R2))[0])) > np.max(np.angle(np.linalg.eig(R @ np.transpose(R1))[0])):
        return v1
    else:
        return v2

def rotation_matrixs_mean(rotations):
    epsilon = 1e-10  # tolerance
    number_rotations = len(rotations)
    R = rotations[0]
    for nb_it in range(20):
        list_r = np.full((3, number_rotations), np.nan + 0j, dtype=complex)
        for i, RI in enumerate(rotations):
            r = lie_log(np.transpose(R) @ RI)
            list_r[:, i] = r
        r = np.mean(list_r, axis=1)
        if np.linalg.norm(r) < epsilon:
            break
        R = R @ expm(cross_ten(r))
    else:
        d_print('the maximum number of iteration where reached')
    return R.real

def compute_relative_rt(image_dir_list: List[Tuple[str, str, str, str]], image_size: Tuple[int, int] = (1280, 720), unit: float = 50, x_num: int = 11, y_num: int = 11, distCoeffs: Dict[str, np.ndarray] = {}, cameraMatrix: Dict[str, np.ndarray] = {}, calibrate_flag: int = cv.CALIB_RATIONAL_MODEL, pnp_flag: int = cv.SOLVEPNP_ITERATIVE, calibrate_iterations: int = 50, calibrate_termination_eps: float = 1e-10):
    """????????????rt

    Args:
        image_dir_list (List[Tuple[str, str, str, str]]): ??????????????????rt????????????????????? 
            ??????: 
                [(image0_dir, image1_dir), ...]????????????image0_dir???image1_dir???????????????rt
                
                [(image0_dir, image1_dir), (image2_dir, image3_dir)...]????????????image0_dir???image1_dir???????????????rt, ?????????image2_dir???image3_dir????????????rt
        
        image_size (Tuple[int, int], optional): ????????????. Defaults to (1280, 720).
        unit (float, optional): ???????????????. Defaults to 50.
        x_num (int, optional): ?????????????????????. Defaults to 11.
        y_num (int, optional): ?????????????????????. Defaults to 11.
        distCoeffs (Dict[str, np.ndarray], optional): ????????????(1x14) (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,??x,??y]]]]) of 4, 5, 8, 12 or 14 elements, Defaults to None.
        cameraMatrix (Dict[str, np.ndarray], optional): ????????????(3x3) Defaults to None.
        
        calibrate_flag (int, optional): ????????????, https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d. Defaults to cv.CALIB_RATIONAL_MODEL.
        
        pnp_flag (int, optional): pnp??????, https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d. Defaults to cv.SOLVEPNP_ITERATIVE.

        calibrate_iterations (int, optional): ????????????????????????. Defaults to 50.
        
        calibrate_termination_eps (float, optional): ??????????????????. Defaults to 1e-10.
    Returns:
        ret_list (List[Tuple[np.ndarray, np.ndarray, float, float, float, float]]): ??????RT(4x4), ??????RT??????, 0->1???3d????????????, 1->0???3d????????????, 0->1???2d????????????, 1->0???2d????????????
    """    
    p2d_list_d = {}
    dist_d = {}
    cam_d = {}
    T_list_d = {}
    cam_p3d_list_d = {}
    p3d = np.concatenate((np.mgrid[0:x_num, 0:y_num].T.reshape(-1, 2), np.zeros((x_num * y_num, 1))), axis=1).astype(np.float32) * unit
    W, H = image_size
    # ???????????????????????????????????????
    unique_image_dir_list = set(chain(*[l[:2] for l in image_dir_list]))
    d_print('?????????????????????????????????')
    # ???????????????????????????
    for image_dir in unique_image_dir_list:
        p2d_list = []
        for image_path in tqdm(sorted(os.listdir(image_dir)), desc=f"?????? {os.path.basename(image_dir)} ????????????", leave=False):
            image_path = os.path.join(image_dir, image_path)
            if not os.path.isfile(image_path):
                continue
            gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            assert gray.shape == (H, W)
            ret, corners = cv.findChessboardCornersSB(gray, (x_num, y_num), None)
            if not ret:
                p2d_list.append(None)
                continue
            p2d_list.append(corners)
        exist_corner_p2d_list = [p2d for p2d in p2d_list if p2d is not None]  # ????????????????????????
        exist_corner_p3d_list = [p3d, ] * len(exist_corner_p2d_list)
        d_print(f'{os.path.basename(image_dir)} ??????')
        dist = distCoeffs.get(image_dir, None)
        mtx = cameraMatrix.get(image_dir, None)
        if mtx is not None and dist is not None:
            d_print(f'{image_dir} ???????????????????????????????????? {mtx} {dist}')
        else:
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                exist_corner_p3d_list, exist_corner_p2d_list, (W, H), None, None, flags=calibrate_flag, criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, calibrate_iterations, calibrate_termination_eps))  # ??????
            d_print(f'{os.path.basename(image_dir)} ?????????')
            exist_corner_undistort_p2d_list = [cv.undistortPoints(p2d, mtx, dist, None, None, mtx) for p2d in exist_corner_p2d_list]  # https://blog.csdn.net/jonathanzh/article/details/104418758
            d_print(f'{os.path.basename(image_dir)} ????????????')
            mtx = cv.initCameraMatrix2D(exist_corner_p3d_list, exist_corner_undistort_p2d_list, (W, H))  # ???????????????????????????
            d_print(f'{image_dir} {dist} {mtx}')
        T_list = []
        cam_p3d_list = []
        for p2d in tqdm(p2d_list, desc=f"?????? {os.path.basename(image_dir)} RT", leave=False):
            if p2d is None:
                T_list.append(None)
                cam_p3d_list.append(None)
                continue
            _, R_v, t = cv.solvePnP(p3d, p2d, mtx, dist, None, None, False, flags=pnp_flag)
            R, _ = cv.Rodrigues(R_v)
            T_list.append(
                np.concatenate((np.hstack((R, t)), np.eye(4)[-1:, :]), axis=0)
            )
            cam_p3d_list.append(
                p3d @ R.T + t.T
            )
        p2d_list_d[image_dir] = p2d_list
        dist_d[image_dir] = dist
        cam_d[image_dir] = mtx
        T_list_d[image_dir] = T_list
        cam_p3d_list_d[image_dir] = cam_p3d_list

    # ??????
    test_p2d_list_d = {}
    test_cam_p3d_list_d = {}
    unique_image_dir_list = set(chain(*[l[2:] if len(l) > 2 else l[:2] for l in image_dir_list]))
    d_print('test ????????????')
    for image_dir in unique_image_dir_list:
        if image_dir in p2d_list_d:
            test_p2d_list_d[image_dir] = p2d_list_d[image_dir]
            test_cam_p3d_list_d[image_dir] = cam_p3d_list_d[image_dir]
            continue

        p2d_list = []
        for image_path in tqdm(sorted(os.listdir(image_dir)), desc=f"?????? {os.path.basename(image_dir)} ????????????", leave=False):
            image_path = os.path.join(image_dir, image_path)
            if not os.path.isfile(image_path):
                continue
            gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            assert gray.shape == (H, W)
            ret, corners = cv.findChessboardCornersSB(gray, (x_num, y_num), None)
            if not ret:
                p2d_list.append(None)
                continue
            p2d_list.append(corners)
        test_p2d_list_d[image_dir] = p2d_list
        test_cam_p3d_list_d[image_dir] = cam_p3d_list

    ret_list = []
    # ????????????RT
    d_print(f'????????????RT')
    for t in image_dir_list:
        if len(t) == 2:
            image0_dir, image1_dir = t
            test_image0_dir, test_image1_dir = t
        else:
            image0_dir, image1_dir, test_image0_dir, test_image1_dir = t
        cam0 = cam_d[image0_dir]
        cam1 = cam_d[image1_dir]
        dist0 = dist_d[image0_dir]
        dist1 = dist_d[image1_dir]
        d_print(f'?????? {os.path.basename(image0_dir)} ??? {os.path.basename(image1_dir)} ?????????RT')
        T_list = np.stack([T1 @ np.linalg.inv(T0) for T0, T1 in zip(T_list_d[image0_dir], T_list_d[image1_dir]) if T0 is not None and T1 is not None], axis=0)
        T = np.eye(4)
        T[:3, -1] = T_list[:, :3, -1].mean(axis=0)
        T[:3, :3] = rotation_matrixs_mean(T_list[:, :3, :3])
        T[-1, -1] = 1
        T_inv = np.linalg.inv(T)
        # ????????????
        error0_1_3d_list = []
        error1_0_3d_list = []
        error0_1_2d_list = []
        error1_0_2d_list = []
        for c03d, c13d, p2d0, p2d1 in tqdm(zip(test_cam_p3d_list_d[test_image0_dir], test_cam_p3d_list_d[test_image1_dir], test_p2d_list_d[test_image0_dir], test_p2d_list_d[test_image1_dir]), desc=f'???????????? {os.path.basename(image0_dir)} {os.path.basename(image1_dir)}', leave=False):
            if c03d is None or c13d is None or p2d0 is None or p2d1 is None:
                continue
            cam03d_cam13d = (T[:3, :3] @ c03d.T + T[:3, -1:]).T  # n x 3 0??????3d->1??????3d
            cam13d_cam03d = (T_inv[:3, :3] @ c13d.T + T_inv[:3, -1:]).T  # n x 3 1??????3d->0??????3d
            cam13d_cam03d_cam02d, _ = cv.projectPoints(cam13d_cam03d, np.eye(3), np.zeros(3), cam0, dist0)  # 1??????3d->0??????3d->0??????2d 
            cam03d_cam13d_cam12d, _ = cv.projectPoints(cam03d_cam13d, np.eye(3), np.zeros(3), cam1, dist1)  # 0??????3d->1??????3d->1??????2d
            error0_1_3d = np.linalg.norm(cam03d_cam13d - c13d, axis=1).mean()
            error1_0_3d = np.linalg.norm(cam13d_cam03d - c03d, axis=1).mean()
            error0_1_2d = np.linalg.norm(cam03d_cam13d_cam12d - p2d1, axis=2).mean()
            error1_0_2d = np.linalg.norm(cam13d_cam03d_cam02d - p2d0, axis=2).mean()
            error0_1_3d_list.append(error0_1_3d)
            error1_0_3d_list.append(error1_0_3d)
            error0_1_2d_list.append(error0_1_2d)
            error1_0_2d_list.append(error1_0_2d)
        error0_1_3d_avg = sum(error0_1_3d_list) / len(error0_1_3d_list)
        error1_0_3d_avg = sum(error1_0_3d_list) / len(error1_0_3d_list)
        error0_1_2d_avg = sum(error0_1_2d_list) / len(error0_1_2d_list)
        error1_0_2d_avg = sum(error1_0_2d_list) / len(error1_0_2d_list)
        ret_list.append((T, T_inv, error0_1_3d_avg, error1_0_3d_avg, error0_1_2d_avg, error1_0_2d_avg))
    return ret_list

if __name__ == '__main__':
    from pprint import pprint
    image_dir_list = [
        ('./data/image0_mini', './data/image2_mini'),
        # ('./data/image 0', './data/image 2')
    ]
    dist = {
        './data/image0_mini': np.array([[6.30676712e-02,2.55872619e-01,-2.29933386e-04,-2.39181315e-05,-2.40607963e-01,-1.24535270e-02,2.46142112e-01,-1.76471377e-01,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00]]),
        './data/image2_mini': np.array([[-4.44426998e+00,4.35360665e+01,3.91375460e-03,-7.23408915e-04,9.49715316e+01,-4.51740863e+00,4.40729230e+01,8.88821371e+01,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00]])
    }
    mtx = {
        './data/image2_mini': np.array([[598.66016445,0.,639.5,],[0.,598.66016445,359.5,],[0.,0.,1.,]]),
        './data/image0_mini': np.array([[610.86995698,0.,639.5,],[0.,610.86995698,359.5,],[0.,0.,1.,]])
    }
    ret = compute_relative_rt(image_dir_list, calibrate_termination_eps=1e-3, distCoeffs=dist, cameraMatrix=mtx)
    pprint(ret)
    print(ret[0][0][:3, :3].tolist())
    print(ret[0][0][:3, -1].tolist())
    print(ret[0][2:])