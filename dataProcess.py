import time
K4ABT_JOINT_NAMES = ["pelvis", "spine - navel", "spine - chest", "neck", "left clavicle", "left shoulder", "left elbow",
                     "left wrist", "left hand", "left handtip", "left thumb", "right clavicle", "right shoulder",
                     "right elbow",
                     "right wrist", "right hand", "right handtip", "right thumb", "left hip", "left knee", "left ankle",
                     "left foot",
                     "right hip", "right knee", "right ankle", "right foot", "head", "nose", "left eye", "left ear",
                     "right eye", "right ear"]
Master = 0
sub1 = 1
sub2 = 2
import numpy as np
import pykinect_azure as pykinect


class utils(object):
    def __init__(self):
        pass

    @classmethod
    def sub2MasterRT(cls, ak_id, joints):
        if ak_id == sub1:
            r = np.array([0.9716459085814197, -0.008863263283334227, -0.23627456676705091, 0.0037945095563309355,
                          0.9997529893227077, -0.02189890493921318, 0.23641030018637987, 0.020381435285084383,
                          0.9714395334046815]).reshape([3, 3])

            t = np.array([429.1234786125495, -37.135246339367896, 182.34116001319433]).reshape(1, 3)
        elif ak_id == sub2:
            r = np.array(
                [0.960294354896268, 0.025595421146279804, 0.27781221422159447, -0.01871431897730644, 0.9994495506342523,
                 -0.027392882327477473, -0.27836042502406144, 0.021106163870679703, 0.9602447623533737]).reshape([3, 3])
            t = np.array([-555.8313577789826, -32.22200112504963, 122.8330270541916]).reshape(1, 3)
        joints = np.dot(joints, r.T) + t
        return joints

    @classmethod
    def midFilter(cls, p0, p1, p2):# 中值观察
        x = np.array([p0[:, 0], p1[:, 0], p2[:, 0]])
        y = np.array([p0[:, 1], p1[:, 1], p2[:, 1]])
        z = np.array([p0[:, 2], p1[:, 2], p2[:, 2]])
        x = np.median(x, axis=0).reshape([32, 1])
        y = np.median(y, axis=0).reshape([32, 1])
        z = np.median(z, axis=0).reshape([32, 1])
        return np.hstack((x, y, z)).reshape([-1, 3])

    @classmethod
    def bonesLengtg(cls, joints_ak: np):# 均值骨长
        # 每32行是一帧，先不考虑五官
        bl = np.zeros([26, 1])
        frame_num = joints_ak.shape[0] / 32
        for fn in range(int(frame_num)):
            joints = joints_ak[fn * 32:(fn + 1) * 32]
            bl[0, 0] += np.linalg.norm(joints[0] - joints[1])
            bl[1, 0] += np.linalg.norm(joints[0] - joints[18])
            bl[2, 0] += np.linalg.norm(joints[0] - joints[22])
            bl[3, 0] += np.linalg.norm(joints[1] - joints[2])
            bl[4, 0] += np.linalg.norm(joints[2] - joints[3])
            bl[5, 0] += np.linalg.norm(joints[2] - joints[4])
            bl[6, 0] += np.linalg.norm(joints[2] - joints[11])
            bl[7, 0] += np.linalg.norm(joints[3] - joints[26])
            # bl[8,0] = np.linalg.norm(joints[26] - joints[27])
            # bl[9,0] = np.linalg.norm(joints[26] - joints[28])
            # bl[10,0] = np.linalg.norm(joints[26] - joints[29])
            # bl[11,0] = np.linalg.norm(joints[26] - joints[30])
            # bl[12,0] = np.linalg.norm(joints[26] - joints[31])
            bl[8, 0] += np.linalg.norm(joints[18] - joints[19])
            bl[9, 0] += np.linalg.norm(joints[19] - joints[20])
            bl[10, 0] += np.linalg.norm(joints[20] - joints[21])
            bl[11, 0] += np.linalg.norm(joints[22] - joints[23])
            bl[12, 0] += np.linalg.norm(joints[23] - joints[24])
            bl[13, 0] += np.linalg.norm(joints[24] - joints[25])
            bl[14, 0] += np.linalg.norm(joints[4] - joints[5])
            bl[15, 0] += np.linalg.norm(joints[5] - joints[6])
            bl[16, 0] += np.linalg.norm(joints[6] - joints[7])
            bl[17, 0] += np.linalg.norm(joints[7] - joints[8])
            bl[18, 0] += np.linalg.norm(joints[7] - joints[10])
            bl[19, 0] += np.linalg.norm(joints[8] - joints[9])
            bl[20, 0] += np.linalg.norm(joints[11] - joints[12])
            bl[21, 0] += np.linalg.norm(joints[12] - joints[13])
            bl[22, 0] += np.linalg.norm(joints[13] - joints[14])
            bl[23, 0] += np.linalg.norm(joints[14] - joints[15])
            bl[24, 0] += np.linalg.norm(joints[14] - joints[17])
            bl[25, 0] += np.linalg.norm(joints[15] - joints[16])
        return bl, frame_num

    @classmethod
    def boneLengthDict(cls, bl: np):
        bld = dict()
        bld['0-1'] = bl[0, 0]
        bld['0-18'] = bl[1, 0]
        bld['0-22'] = bl[2, 0]
        bld['1-2'] = bl[3, 0]
        bld['2-3'] = bl[4, 0]
        bld['2-4'] = bl[5, 0]
        bld['2-11'] = bl[6, 0]
        bld['3-26'] = bl[7, 0]
        bld['18-19'] = bl[8, 0]
        bld['19-20'] = bl[9, 0]
        bld['20-21'] = bl[10, 0]
        bld['22-23'] = bl[11, 0]
        bld['23-24'] = bl[12, 0]
        bld['24-25'] = bl[13, 0]
        bld['4-5'] = bl[14, 0]
        bld['5-6'] = bl[15, 0]
        bld['6-7'] = bl[16, 0]
        bld['7-8'] = bl[17, 0]
        bld['7-10'] = bl[18, 0]
        bld['8-9'] = bl[19, 0]
        bld['11-12'] = bl[20, 0]
        bld['12-13'] = bl[21, 0]
        bld['13-14'] = bl[22, 0]
        bld['14-15'] = bl[23, 0]
        bld['14-17'] = bl[24, 0]
        bld['15-16'] = bl[25, 0]
        return bld

    @classmethod
    def distriDistance(cls, bu0: np, bu1: np, bld: float):# 骨长样本
        # bu都是(36,3),每个点对于相邻36个散点都有距离
        """
        :param bu0:
        :param bu1:
        :return:(36,36)
            \ joint1
      joint0 \_______
              | distance ……
              | distance ……
              | ……       ……
        """
        bu0 = bu0[:,None,:]
        bu1 = bu1[None,:,:]
        b_distance_score = np.linalg.norm(bu0-bu1,axis=-1)
        # b_distance_score -= bld
        # if bld == 280.10292586429114:
        # return 1 - abs(b_distance_score) / sum(sum(abs(b_distance_score)))
        return 1 - abs(b_distance_score-bld)/abs(b_distance_score - bld)**2/(36**2-1)

    @classmethod
    def threeAKs(cls, joint_ak_0: np, joint_ak_1: np, joint_ak_2: np, joint_set_bar: np):# 三点观察
        joint_ak_0 = joint_ak_0[None,:,:]
        joint_ak_1 = joint_ak_1[None,:,:]
        joint_ak_2 = joint_ak_2[None,:,:]
        js = np.linalg.norm((joint_set_bar-joint_ak_0),axis=-1)+np.linalg.norm((joint_set_bar-joint_ak_1),axis=-1)+np.linalg.norm((joint_set_bar-joint_ak_2),axis=-1)
        # js是所有样本
        # 样本数据标准化
        js_bar = np.mean(js)
        std = np.var(js,ddof=1)
        # return 1 - js / sum(sum(js))
        return 1 - abs(js-js_bar)/std
    @classmethod
    def dissimilation(cls, d:int, c:int,j1:np, j2:np):
        x_times = np.random.random(32)
        y_times = np.random.random(32)
        z_times = np.random.random(32)
        if d ==1:
            if c ==1:
                j1[:,0] *= x_times
            elif c==2:
                j1[:, 0] *= x_times
                j1[:, 1] *= y_times
            else:
                j1[:, 0] *= x_times
                j1[:, 1] *= y_times
                j1[:, 2] *= z_times
        else:
            if c ==1:
                j1[:,0] *= x_times
                j2[:,0] *= x_times
            elif c==2:
                j1[:, 0] *= x_times
                j1[:, 1] *= y_times
                j2[:, 0] *= x_times
                j2[:, 1] *= y_times
            else:
                j1[:, 0] *= x_times
                j1[:, 1] *= y_times
                j1[:, 2] *= z_times
                j2[:, 0] *= x_times
                j2[:, 1] *= y_times
                j2[:, 2] *= z_times
        return j1,j2
    @classmethod
    def bonesLengthMid(cls,joints_ak: np):# 中值骨长
        # 每32行是一帧，先不考虑五官
        frame_num = int(joints_ak.shape[0] / 32)
        bl = np.zeros([frame_num, 20, 1])
        for fn in range(frame_num):
            joints = joints_ak[fn * 32:(fn + 1) * 32]
            bl_unit = np.zeros([20, 1])
            bl_unit[0, 0] = np.linalg.norm(joints[0] - joints[1])
            bl_unit[1, 0] = np.linalg.norm(joints[0] - joints[18])
            bl_unit[2, 0] = np.linalg.norm(joints[0] - joints[22])
            bl_unit[3, 0] = np.linalg.norm(joints[1] - joints[2])
            bl_unit[4, 0] = np.linalg.norm(joints[2] - joints[3])
            bl_unit[5, 0] = np.linalg.norm(joints[2] - joints[4])
            bl_unit[6, 0] = np.linalg.norm(joints[2] - joints[11])
            bl_unit[7, 0] = np.linalg.norm(joints[3] - joints[26])
            bl_unit[8, 0] = np.linalg.norm(joints[18] - joints[19])
            bl_unit[9, 0] = np.linalg.norm(joints[19] - joints[20])
            bl_unit[10, 0] = np.linalg.norm(joints[20] - joints[21])
            bl_unit[11, 0] = np.linalg.norm(joints[22] - joints[23])
            bl_unit[12, 0] = np.linalg.norm(joints[23] - joints[24])
            bl_unit[13, 0] = np.linalg.norm(joints[24] - joints[25])
            bl_unit[14, 0] = np.linalg.norm(joints[4] - joints[5])
            bl_unit[15, 0] = np.linalg.norm(joints[5] - joints[6])
            bl_unit[16, 0] = np.linalg.norm(joints[6] - joints[7])
            # bl_unit[17, 0] = np.linalg.norm(joints[7] - joints[8])
            # bl_unit[18, 0] = np.linalg.norm(joints[7] - joints[10])
            # bl_unit[19, 0] = np.linalg.norm(joints[8] - joints[9])
            bl_unit[17, 0] = np.linalg.norm(joints[11] - joints[12])
            bl_unit[18, 0] = np.linalg.norm(joints[12] - joints[13])
            bl_unit[19, 0] = np.linalg.norm(joints[13] - joints[14])
            # bl_unit[23, 0] = np.linalg.norm(joints[14] - joints[15])
            # bl_unit[24, 0] = np.linalg.norm(joints[14] - joints[17])
            # bl_unit[25, 0] = np.linalg.norm(joints[15] - joints[16])
            bl[fn] = bl_unit
        return bl
    @classmethod
    def dictAtMidValue(cls,bl0,bl1,bl2):
        bld = dict()
        rs = []
        for bj in range(bl0.shape[1]):
            rs.append(np.median([bl0[:,bj,:],bl1[:,bj,:],bl2[:,bj,:]]))
        bld['0-1'] = rs[0]
        bld['0-18'] = rs[1]
        bld['0-22'] = rs[2]
        bld['1-2'] = rs[3]
        bld['2-3'] = rs[4]
        bld['2-4'] = rs[5]
        bld['2-11'] = rs[6]
        bld['3-26'] = rs[7]
        bld['18-19'] = rs[8]
        bld['19-20'] = rs[9]
        bld['20-21'] = rs[10]
        bld['22-23'] = rs[11]
        bld['23-24'] = rs[12]
        bld['24-25'] = rs[13]
        bld['4-5'] = rs[14]
        bld['5-6'] = rs[15]
        bld['6-7'] = rs[16]
        # bld['7-8'] = rs[17]
        # bld['7-10'] = rs[18]
        # bld['8-9'] = rs[19]
        bld['11-12'] = rs[17]
        bld['12-13'] = rs[18]
        bld['13-14'] = rs[19]
        # bld['14-15'] = rs[23]
        # bld['14-17'] = rs[24]
        # bld['15-16'] = rs[25]
        return bld
class DataProcess(object):
    def __init__(self, coor_path: str, img_pth: str, ak_id: int):
        self.joints = K4ABT_JOINT_NAMES
        self.joints_coor = np.loadtxt(coor_path)
        self.img_pth = img_pth
        self.ak_id = ak_id

    @classmethod
    def getAKintrisics(cls, ak_id: int, intrisics_pth: str):
        pykinect.initialize_libraries(track_body=False)
        # 开启ak
        ak_configuration = pykinect.Configuration()
        ak_configuration.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
        ak_configuration.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        ak_configuration.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        ak_configuration.synchronized_images_only = True
        ak_configuration.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32

        Device = pykinect.start_device(device_index=ak_id, config=ak_configuration)
        # 得到内参
        cal_handle = Device.get_calibration(ak_configuration.depth_mode,
                                            ak_configuration.color_resolution)
        rgb_para_handle = cal_handle._handle.color_camera_calibration.intrinsics.parameters.param
        dep_para_handle = cal_handle._handle.depth_camera_calibration.intrinsics.parameters.param
        np.savetxt(f'{intrisics_pth}/{ak_id}_rgb_in.txt', np.array(
            [[rgb_para_handle.fx, 0, rgb_para_handle.cx], [0, rgb_para_handle.fy, rgb_para_handle.cy], [0, 0, 1]]))
        np.savetxt(f'{intrisics_pth}/{ak_id}_dep_in.txt', np.array(
            [[dep_para_handle.fx, 0, dep_para_handle.cx], [0, dep_para_handle.fy, dep_para_handle.cy], [0, 0, 1]]))
        # 得到两个镜头的rt
        r = np.array(cal_handle._handle.color_camera_calibration.extrinsics.rotation).reshape([3, 3])
        t = np.array(cal_handle._handle.color_camera_calibration.extrinsics.translation).reshape([3, 1])
        rgbT = np.hstack((r, t))
        np.savetxt(f'{intrisics_pth}/{ak_id}_rgb_ex.txt', rgbT)
        r = np.array(cal_handle._handle.depth_camera_calibration.extrinsics.rotation).reshape([3, 3])
        t = np.array(cal_handle._handle.depth_camera_calibration.extrinsics.translation).reshape([3, 1])
        depT = np.hstack((r, t))
        np.savetxt(f'{intrisics_pth}/{ak_id}_dep_ex.txt', depT)
        Device.close()

    @classmethod
    def verfyJointsData(cls, ak_id: int, pic_idx: int):
        import cv2
        # 先得到dep到rgb的rt
        T = np.loadtxt(f'param/{ak_id}_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        data0_frame0 = np.loadtxt(f'./data/fw/data{ak_id}.txt')[(pic_idx) * 32:(pic_idx + 1) * 32]
        data0_frame0 = np.dot(data0_frame0, r.T) + t  # 这个是当前ak的rgb下的joints的3d坐标
        if ak_id != Master:  # 从相机需要再将本相机坐标系转到主相机坐标系
            data0_frame0 = utils.sub2MasterRT(ak_id, data0_frame0)
        rgbIn = np.loadtxt(f'param/{ak_id}_rgb_in.txt')
        uv1 = np.dot(rgbIn, data0_frame0.T) / data0_frame0.T[2]
        p_img0 = cv2.imread(f'./data/fw/image0/image{pic_idx}.bmp')  # 都要画在主相机上！！
        for i in range(uv1.shape[1]):
            u = int(uv1[0, i])
            v = int(uv1[1, i])
            cv2.circle(p_img0, (u, v), 2, (0, 255, 0))
        cv2.imshow('p', p_img0)
        cv2.waitKey(0)

    @classmethod
    def jointsIn3D_interpolation(cls):  # 3D中画一下3个ak的关键点，然后用λ和β进行空间离散化
        import open3d as o3d
        # 得到三个ak的joints在主相机rgb空间的坐标 【3D的】
        T = np.loadtxt(f'param/0_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        joints_ak_0 = np.loadtxt('./data/fw/data0.txt')[:32]
        joints_ak_0 = np.dot(joints_ak_0, r.T) + t  # 在ak0的rgb坐标系中了
        cs0 = np.zeros([32, 3])
        cs0[:, 0] = 1  # red

        T = np.loadtxt(f'param/1_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        joints_ak_1 = np.loadtxt('./data/fw/data0.txt')[:32]
        joints_ak_1 = np.dot(joints_ak_1, r.T) + t  # 在ak1的rgb坐标系中了
        joints_ak_1 = utils.sub2MasterRT(1, joints_ak_1)
        cs1 = np.zeros([32, 3])
        cs1[:, 1] = 1  # green

        T = np.loadtxt(f'param/2_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        joints_ak_2 = np.loadtxt('./data/fw/data2.txt')[:32]
        joints_ak_2 = np.dot(joints_ak_2, r.T) + t  # 在ak2的rgb坐标系中了
        joints_ak_2 = utils.sub2MasterRT(2, joints_ak_2)
        cs2 = np.zeros([32, 3])
        cs2[:, 2] = 1  # blue
        # 连线
        line = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 7], [11, 2], [12, 11],
                [13, 12], [14, 13], [15, 14], [16, 15], [17, 14], [18, 0], [19, 18], [20, 19], [21, 20], [22, 0],
                [23, 22], [24, 23], [25, 24], [26, 3], [27, 26], [28, 26], [29, 26], [30, 26], [31, 26]]
        cl0 = [[1, 0, 0] for i in range(len(line))]
        cl1 = [[0, 1, 0] for j in range(len(line))]
        cl2 = [[0, 0, 1] for k in range(len(line))]

        # 先画一下主相机的空间分布 按点云画
        '''画图 start'''
        test0_pcd = o3d.geometry.PointCloud()
        test0_pcd.points = o3d.utility.Vector3dVector(np.vstack([joints_ak_0]))  # 定义点云坐标位置
        test0_pcd.colors = o3d.utility.Vector3dVector(np.vstack([cs0]))  # 定义点云的颜色
        test0_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lines0_pcd = o3d.geometry.LineSet()
        lines0_pcd.lines = o3d.utility.Vector2iVector(line)
        lines0_pcd.colors = o3d.utility.Vector3dVector(cl0)  # 线条颜色
        lines0_pcd.points = o3d.utility.Vector3dVector(joints_ak_0)
        lines0_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        test1_pcd = o3d.geometry.PointCloud()
        test1_pcd.points = o3d.utility.Vector3dVector(np.vstack([joints_ak_1]))  # 定义点云坐标位置
        test1_pcd.colors = o3d.utility.Vector3dVector(np.vstack([cs1]))  # 定义点云的颜色
        test1_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lines1_pcd = o3d.geometry.LineSet()
        lines1_pcd.lines = o3d.utility.Vector2iVector(line)
        lines1_pcd.colors = o3d.utility.Vector3dVector(cl1)  # 线条颜色
        lines1_pcd.points = o3d.utility.Vector3dVector(joints_ak_1)
        lines1_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        test2_pcd = o3d.geometry.PointCloud()
        test2_pcd.points = o3d.utility.Vector3dVector(np.vstack([joints_ak_2]))  # 定义点云坐标位置
        test2_pcd.colors = o3d.utility.Vector3dVector(np.vstack([cs2]))  # 定义点云的颜色
        test2_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lines2_pcd = o3d.geometry.LineSet()
        lines2_pcd.lines = o3d.utility.Vector2iVector(line)
        lines2_pcd.colors = o3d.utility.Vector3dVector(cl2)  # 线条颜色
        lines2_pcd.points = o3d.utility.Vector3dVector(joints_ak_2)
        lines2_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([test0_pcd, lines0_pcd, test1_pcd, lines1_pcd, test2_pcd, lines2_pcd],
                                          window_name="master AK space")
        '''画图 end'''

        '''affine算法'''
        affine_l = [1, 2, 3, 4, 5, 6, 7, 8]
        dist_set = []
        for lamda in affine_l:
            for beta in affine_l[:9 - lamda]:
                joint_bar = lamda * joints_ak_0 / 10 + beta * joints_ak_1 / 10 + (10 - lamda - beta) * joints_ak_2 / 10
                dist_set.append(joint_bar)
        joint_bar_set = np.array(dist_set).reshape(-1, 3)
        cl = np.zeros(joint_bar_set.shape)
        distri_pcd = o3d.geometry.PointCloud()
        distri_pcd.points = o3d.utility.Vector3dVector(joint_bar_set)  # 定义点云坐标位置
        distri_pcd.colors = o3d.utility.Vector3dVector(cl)  # 定义点云的颜色
        distri_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries(
            [distri_pcd, test0_pcd, lines0_pcd, test1_pcd, lines1_pcd, test2_pcd, lines2_pcd],
            window_name="dist joints points")
        '''画散点，散点的颜色是黑色'''

    @classmethod
    def BPapplication(cls):  # 用来测fps
        # 三个观察值，真实估计放在36个离散空间内,创建因子图模型
        from dataStructure import belief_propagation, string2factor_graph, factor
        import open3d as o3d
        import cv2
        # todo 第一帧和非第一帧观察值不一样
        # 每一帧的观察值数量一样
        joints_ak_0 = np.loadtxt('./data/fw/data0.txt')
        joints_ak_1 = np.loadtxt('./data/fw/data1.txt')
        joints_ak_2 = np.loadtxt('./data/fw/data2.txt')
        mrf = string2factor_graph(
            # f'f0({K4ABT_JOINT_NAMES[0]})f1({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[1]})f2({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[18]})f3({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[22]})f4({K4ABT_JOINT_NAMES[18]},{K4ABT_JOINT_NAMES[19]})f5({K4ABT_JOINT_NAMES[20]},{K4ABT_JOINT_NAMES[19]})f6({K4ABT_JOINT_NAMES[21]},{K4ABT_JOINT_NAMES[20]})f7({K4ABT_JOINT_NAMES[22]},{K4ABT_JOINT_NAMES[23]})f8({K4ABT_JOINT_NAMES[23]},{K4ABT_JOINT_NAMES[24]})f9({K4ABT_JOINT_NAMES[24]},{K4ABT_JOINT_NAMES[25]})f10({K4ABT_JOINT_NAMES[1]},{K4ABT_JOINT_NAMES[2]})f11({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[3]})f12({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[4]})f13({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[11]})f14({K4ABT_JOINT_NAMES[3]},{K4ABT_JOINT_NAMES[26]})f15({K4ABT_JOINT_NAMES[4]},{K4ABT_JOINT_NAMES[5]})f16({K4ABT_JOINT_NAMES[5]},{K4ABT_JOINT_NAMES[6]})f17({K4ABT_JOINT_NAMES[6]},{K4ABT_JOINT_NAMES[7]})f18({K4ABT_JOINT_NAMES[7]},{K4ABT_JOINT_NAMES[8]})f19({K4ABT_JOINT_NAMES[7]},{K4ABT_JOINT_NAMES[10]})f20({K4ABT_JOINT_NAMES[8]},{K4ABT_JOINT_NAMES[9]})f21({K4ABT_JOINT_NAMES[11]},{K4ABT_JOINT_NAMES[12]})f22({K4ABT_JOINT_NAMES[12]},{K4ABT_JOINT_NAMES[13]})f23({K4ABT_JOINT_NAMES[13]},{K4ABT_JOINT_NAMES[14]})f24({K4ABT_JOINT_NAMES[14]},{K4ABT_JOINT_NAMES[15]})f25({K4ABT_JOINT_NAMES[14]},{K4ABT_JOINT_NAMES[17]})f26({K4ABT_JOINT_NAMES[15]},{K4ABT_JOINT_NAMES[16]})f27({K4ABT_JOINT_NAMES[1]})f28({K4ABT_JOINT_NAMES[2]})f29({K4ABT_JOINT_NAMES[3]})f30({K4ABT_JOINT_NAMES[4]})f31({K4ABT_JOINT_NAMES[5]})f32({K4ABT_JOINT_NAMES[6]})f33({K4ABT_JOINT_NAMES[7]})f34({K4ABT_JOINT_NAMES[8]})f35({K4ABT_JOINT_NAMES[9]})f36({K4ABT_JOINT_NAMES[10]})f37({K4ABT_JOINT_NAMES[11]})f38({K4ABT_JOINT_NAMES[12]})f39({K4ABT_JOINT_NAMES[13]})f40({K4ABT_JOINT_NAMES[14]})f41({K4ABT_JOINT_NAMES[15]})f42({K4ABT_JOINT_NAMES[16]})f43({K4ABT_JOINT_NAMES[17]})f44({K4ABT_JOINT_NAMES[18]})f45({K4ABT_JOINT_NAMES[19]})f46({K4ABT_JOINT_NAMES[20]})f47({K4ABT_JOINT_NAMES[21]})f48({K4ABT_JOINT_NAMES[22]})f49({K4ABT_JOINT_NAMES[23]})f50({K4ABT_JOINT_NAMES[24]})f51({K4ABT_JOINT_NAMES[25]})f52({K4ABT_JOINT_NAMES[26]})')
            f'f0({K4ABT_JOINT_NAMES[0]})f1({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[1]})f2({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[18]})f3({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[22]})f4({K4ABT_JOINT_NAMES[18]},{K4ABT_JOINT_NAMES[19]})f5({K4ABT_JOINT_NAMES[20]},{K4ABT_JOINT_NAMES[19]})f6({K4ABT_JOINT_NAMES[21]},{K4ABT_JOINT_NAMES[20]})f7({K4ABT_JOINT_NAMES[22]},{K4ABT_JOINT_NAMES[23]})f8({K4ABT_JOINT_NAMES[23]},{K4ABT_JOINT_NAMES[24]})f9({K4ABT_JOINT_NAMES[24]},{K4ABT_JOINT_NAMES[25]})f10({K4ABT_JOINT_NAMES[1]},{K4ABT_JOINT_NAMES[2]})f11({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[3]})f12({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[4]})f13({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[11]})f14({K4ABT_JOINT_NAMES[3]},{K4ABT_JOINT_NAMES[26]})f15({K4ABT_JOINT_NAMES[4]},{K4ABT_JOINT_NAMES[5]})f16({K4ABT_JOINT_NAMES[5]},{K4ABT_JOINT_NAMES[6]})f17({K4ABT_JOINT_NAMES[6]},{K4ABT_JOINT_NAMES[7]})f21({K4ABT_JOINT_NAMES[11]},{K4ABT_JOINT_NAMES[12]})f22({K4ABT_JOINT_NAMES[12]},{K4ABT_JOINT_NAMES[13]})f23({K4ABT_JOINT_NAMES[13]},{K4ABT_JOINT_NAMES[14]})f27({K4ABT_JOINT_NAMES[1]})f28({K4ABT_JOINT_NAMES[2]})f29({K4ABT_JOINT_NAMES[3]})f30({K4ABT_JOINT_NAMES[4]})f31({K4ABT_JOINT_NAMES[5]})f32({K4ABT_JOINT_NAMES[6]})f33({K4ABT_JOINT_NAMES[7]})f37({K4ABT_JOINT_NAMES[11]})f38({K4ABT_JOINT_NAMES[12]})f39({K4ABT_JOINT_NAMES[13]})f40({K4ABT_JOINT_NAMES[14]})f44({K4ABT_JOINT_NAMES[18]})f45({K4ABT_JOINT_NAMES[19]})f46({K4ABT_JOINT_NAMES[20]})f47({K4ABT_JOINT_NAMES[21]})f48({K4ABT_JOINT_NAMES[22]})f49({K4ABT_JOINT_NAMES[23]})f50({K4ABT_JOINT_NAMES[24]})f51({K4ABT_JOINT_NAMES[25]})f52({K4ABT_JOINT_NAMES[26]})')
        # 每个关键点的离散点都有概率，这里利用欧式距离呈现概率关系
        # 第一帧每个离散点,(36,32,3),每个关节是36个离散点
        # 计算中值 loss = 离散点到中值的距离
        # 对齐
        T = np.loadtxt(f'param/0_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        joints_ak_0 = np.dot(joints_ak_0, r.T) + t  # 在ak0的rgb坐标系中了
        T = np.loadtxt(f'param/1_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        joints_ak_1 = np.dot(joints_ak_1, r.T) + t  # 在ak1的rgb坐标系中了
        joints_ak_1 = utils.sub2MasterRT(1, joints_ak_1)
        T = np.loadtxt(f'param/2_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        joints_ak_2 = np.dot(joints_ak_2, r.T) + t  # 在ak2的rgb坐标系中了
        joints_ak_2 = utils.sub2MasterRT(2, joints_ak_2)
        # 离散点到观察值的score————uni
        # joints_ak_mid = utils.midFilter(joints_ak_0, joints_ak_1, joints_ak_2)
        s = time.time()
        affine_l = [1, 2, 3, 4, 5, 6, 7, 8]
        dist_set = []
        for lamda in affine_l:
            for beta in affine_l[:9 - lamda]:
                joint_bar = lamda * joints_ak_0 / 10 + beta * joints_ak_1 / 10 + (10 - lamda - beta) * joints_ak_2 / 10
                dist_set.append(joint_bar)
        joint_bar_set = np.array(dist_set).reshape(-1, 3200, 3)
        # 离散点之间的距离score
        # 离散点之间的骨头长度做标准，----pairwise
        bl0 = utils.bonesLengthMid(joints_ak_0)
        bl1 = utils.bonesLengthMid(joints_ak_1)
        bl2 = utils.bonesLengthMid(joints_ak_2)
        bld = utils.dictAtMidValue(bl0, bl1, bl2)
        # ak估计做观察值
        for ff in range(0,100):
            joints_score = utils.threeAKs(joints_ak_0[ff * 32:(ff + 1) * 32], joints_ak_1[ff * 32:(ff + 1) * 32],
                                          joints_ak_2[ff * 32:(ff + 1) * 32],
                                          joint_bar_set[:, ff * 32:(ff + 1) * 32, :])

            # 27个(36,36)的骨长关系也加上
            # 0-1 0-18 0-22
            f0 = factor([K4ABT_JOINT_NAMES[0]], joints_score[:, 0])  # joint 36个状态
            f27 = factor([K4ABT_JOINT_NAMES[1]], joints_score[:, 1])
            distance01 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 0, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 1, :],
                                              bld['0-1'])  # pairwise (36,36)个状态
            f1 = factor([K4ABT_JOINT_NAMES[0], K4ABT_JOINT_NAMES[1]], distance01)

            f44 = factor([K4ABT_JOINT_NAMES[18]], joints_score[:, 18])
            distance018 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 0, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 18, :], bld['0-18'])
            f2 = factor([K4ABT_JOINT_NAMES[0], K4ABT_JOINT_NAMES[18]], distance018)

            f48 = factor([K4ABT_JOINT_NAMES[22]], joints_score[:, 22])
            distance022 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 0, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 22, :], bld['0-22'])
            f3 = factor([K4ABT_JOINT_NAMES[0], K4ABT_JOINT_NAMES[22]], distance022)

            # 18-19
            f45 = factor([K4ABT_JOINT_NAMES[19]], joints_score[:, 19])
            distance1819 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 18, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 19, :], bld['18-19'])
            f4 = factor([K4ABT_JOINT_NAMES[18], K4ABT_JOINT_NAMES[19]], distance1819)

            # 19-20
            f46 = factor([K4ABT_JOINT_NAMES[20]], joints_score[:, 20])
            distance1920 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 19, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 20, :], bld['19-20'])
            f5 = factor([K4ABT_JOINT_NAMES[19], K4ABT_JOINT_NAMES[20]], distance1920)

            # 20-21
            f47 = factor([K4ABT_JOINT_NAMES[21]], joints_score[:, 21])
            distance2021 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 20, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 21, :], bld['20-21'])
            f6 = factor([K4ABT_JOINT_NAMES[20], K4ABT_JOINT_NAMES[21]], distance2021)

            # 22-23
            f48 = factor([K4ABT_JOINT_NAMES[22]], joints_score[:, 22])
            f49 = factor([K4ABT_JOINT_NAMES[23]], joints_score[:, 23])
            distance2223 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 22, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 23, :], bld['22-23'])
            f7 = factor([K4ABT_JOINT_NAMES[22], K4ABT_JOINT_NAMES[23]], distance2223)

            # 23-24
            f50 = factor([K4ABT_JOINT_NAMES[24]], joints_score[:, 24])
            distance2324 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 23, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 24, :], bld['23-24'])
            f8 = factor([K4ABT_JOINT_NAMES[24], K4ABT_JOINT_NAMES[23]], distance2324)

            # 24-25
            f51 = factor([K4ABT_JOINT_NAMES[25]], joints_score[:, 25])
            distance2425 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 24, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 25, :], bld['24-25'])
            f9 = factor([K4ABT_JOINT_NAMES[24], K4ABT_JOINT_NAMES[25]], distance2425)

            # 1-2
            f28 = factor([K4ABT_JOINT_NAMES[2]], joints_score[:, 2])
            distance12 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 1, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 2, :], bld['1-2'])
            f10 = factor([K4ABT_JOINT_NAMES[2], K4ABT_JOINT_NAMES[1]], distance12)

            # 2-3 2-4 2-11
            f29 = factor([K4ABT_JOINT_NAMES[3]], joints_score[:, 3])
            f30 = factor([K4ABT_JOINT_NAMES[4]], joints_score[:, 4])
            f37 = factor([K4ABT_JOINT_NAMES[11]], joints_score[:, 11])
            distance23 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 2, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 3, :], bld['2-3'])
            f11 = factor([K4ABT_JOINT_NAMES[2], K4ABT_JOINT_NAMES[3]], distance23)

            distance24 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 2, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 4, :], bld['2-4'])
            f12 = factor([K4ABT_JOINT_NAMES[2], K4ABT_JOINT_NAMES[4]], distance24)

            distance211 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 2, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 11, :], bld['2-11'])
            f13 = factor([K4ABT_JOINT_NAMES[2], K4ABT_JOINT_NAMES[11]], distance211)

            # 3-26
            f52 = factor([K4ABT_JOINT_NAMES[26]], joints_score[:, 26])
            distance326 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 3, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 26, :], bld['3-26'])
            f14 = factor([K4ABT_JOINT_NAMES[3], K4ABT_JOINT_NAMES[26]], distance326)

            # 4-5
            f31 = factor([K4ABT_JOINT_NAMES[5]], joints_score[:, 5])
            distance45 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 4, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 5, :], bld['4-5'])
            f15 = factor([K4ABT_JOINT_NAMES[5], K4ABT_JOINT_NAMES[4]], distance45)

            # 5-6
            f32 = factor([K4ABT_JOINT_NAMES[6]], joints_score[:, 6])
            distance56 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 5, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 6, :], bld['5-6'])
            f16 = factor([K4ABT_JOINT_NAMES[5], K4ABT_JOINT_NAMES[6]], distance56)

            # 6-7
            f33 = factor([K4ABT_JOINT_NAMES[7]], joints_score[:, 7])
            distance67 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 6, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 7, :], bld['6-7'])
            f17 = factor([K4ABT_JOINT_NAMES[6], K4ABT_JOINT_NAMES[7]], distance67)

            # 7-8 7-10
            # f34 = factor([K4ABT_JOINT_NAMES[8]], joints_score[:, 8])
            # distance78 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 7, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 8, :], bld['7-8'])
            # f18 = factor([K4ABT_JOINT_NAMES[7], K4ABT_JOINT_NAMES[8]], distance78)

            # f36 = factor([K4ABT_JOINT_NAMES[10]], joints_score[:, 10])
            # distance710 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 7, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 10, :], bld['7-10'])
            # f19 = factor([K4ABT_JOINT_NAMES[10], K4ABT_JOINT_NAMES[7]], distance710)

            # 8-9
            # f35 = factor([K4ABT_JOINT_NAMES[9]], joints_score[:, 9])
            # distance89 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 8, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 9, :], bld['8-9'])
            # f20 = factor([K4ABT_JOINT_NAMES[8], K4ABT_JOINT_NAMES[9]], distance89)

            # 11-12
            f38 = factor([K4ABT_JOINT_NAMES[12]], joints_score[:, 12])
            distance1112 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 11, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 12, :], bld['11-12'])
            f21 = factor([K4ABT_JOINT_NAMES[11], K4ABT_JOINT_NAMES[12]], distance1112)

            # 12-13
            f39 = factor([K4ABT_JOINT_NAMES[13]], joints_score[:, 13])
            distance1213 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 12, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 13, :], bld['12-13'])
            f22 = factor([K4ABT_JOINT_NAMES[12], K4ABT_JOINT_NAMES[13]], distance1213)

            # 13-14
            f40 = factor([K4ABT_JOINT_NAMES[14]], joints_score[:, 14])
            distance1314 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 13, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 14, :], bld['13-14'])
            f23 = factor([K4ABT_JOINT_NAMES[13], K4ABT_JOINT_NAMES[14]], distance1314)

            # 14-15 14-17
            # f41 = factor([K4ABT_JOINT_NAMES[15]], joints_score[:, 15])
            # distance1415 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 14, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 15, :], bld['14-15'])
            # f24 = factor([K4ABT_JOINT_NAMES[14], K4ABT_JOINT_NAMES[15]], distance1415)

            # f43 = factor([K4ABT_JOINT_NAMES[17]], joints_score[:, 17])
            # distance1417 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 14, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 17, :], bld['14-17'])
            # f25 = factor([K4ABT_JOINT_NAMES[14], K4ABT_JOINT_NAMES[17]], distance1417)

            # 15-16
            # f42 = factor([K4ABT_JOINT_NAMES[16]], joints_score[:, 16])
            # distance1516 = utils.distriDistance(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 15, :], joint_bar_set[:, ff * 32:(ff + 1) * 32, :][:, 16, :], bld['15-16'])
            # f26 = factor([K4ABT_JOINT_NAMES[16], K4ABT_JOINT_NAMES[15]], distance1516)
            names = locals()
            for n in range(53):  # 动态变量名
                if n in [18,19,20,34,35,36,24,25,26,41,42,43]:
                    continue
                mrf.change_factor_distribution(f'f{n}', names['f%s' % n])
            bp = belief_propagation(mrf)
            answer = []
            for jidex, j in enumerate(K4ABT_JOINT_NAMES):
                if j in ["nose", "left eye", "left ear",
                         "right eye", "right ear", "left hand", "left handtip", "left thumb", "right hand",
                         "right handtip", "right thumb"]:
                    continue
                score_array = bp.belief(j).get_distribution()
                idx = np.argmax(score_array)
                answer.append(joint_bar_set[:, ff * 32:(ff + 1) * 32, :][idx, jidex, :])
            bpStep1 = np.array(answer)
        print(time.time()-s)
            # '''画图 start'''
            # line = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 7], [11, 2], [12, 11],
            #         [13, 12], [14, 13], [15, 14], [16, 15], [17, 14], [18, 0], [19, 18], [20, 19], [21, 20], [22, 0],
            #         [23, 22], [24, 23], [25, 24], [26, 3], [27, 26], [28, 26], [29, 26], [30, 26], [31, 26]]
            # cl0 = [[1, 0, 0] for i in range(len(line))]
            # cl1 = [[0, 1, 0] for j in range(len(line))]
            # cl2 = [[0, 0, 1] for k in range(len(line))]
            # cs0 = np.zeros([32, 3])
            # cs0[:, 0] = 1  # red
            # cs1 = np.zeros([32, 3])
            # cs1[:, 1] = 1  # green
            # cs2 = np.zeros([32, 3])
            # cs2[:, 2] = 1  # blue
            # line_bp = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 4], [6, 5], [7, 6], [8, 2], [8, 9],
            #                       [9, 10], [10, 11], [12, 0], [12, 13], [13, 14], [14, 15], [16, 0],
            #                       [16, 17], [17, 18], [18, 19], [20, 3]]
            # cl_bp = [[0, 0, 0] for l in range(len(line_bp))]
            # test0_pcd = o3d.geometry.PointCloud()
            # test0_pcd.points = o3d.utility.Vector3dVector(joints_ak_0[ff * 32:(ff + 1) * 32])  # 定义点云坐标位置
            # test0_pcd.colors = o3d.utility.Vector3dVector(cs0)  # 定义点云的颜色
            # test0_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # lines0_pcd = o3d.geometry.LineSet()
            # lines0_pcd.lines = o3d.utility.Vector2iVector(line)
            # lines0_pcd.colors = o3d.utility.Vector3dVector(cl0)  # 线条颜色
            # lines0_pcd.points = o3d.utility.Vector3dVector(joints_ak_0[ff * 32:(ff + 1) * 32])
            # lines0_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            #
            # test1_pcd = o3d.geometry.PointCloud()
            # test1_pcd.points = o3d.utility.Vector3dVector(joints_ak_1[ff * 32:(ff + 1) * 32])  # 定义点云坐标位置
            # test1_pcd.colors = o3d.utility.Vector3dVector(cs1)  # 定义点云的颜色
            # test1_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # lines1_pcd = o3d.geometry.LineSet()
            # lines1_pcd.lines = o3d.utility.Vector2iVector(line)
            # lines1_pcd.colors = o3d.utility.Vector3dVector(cl1)  # 线条颜色
            # lines1_pcd.points = o3d.utility.Vector3dVector(joints_ak_1[ff * 32:(ff + 1) * 32])
            # lines1_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            #
            # test2_pcd = o3d.geometry.PointCloud()
            # test2_pcd.points = o3d.utility.Vector3dVector(joints_ak_2[ff * 32:(ff + 1) * 32])  # 定义点云坐标位置
            # test2_pcd.colors = o3d.utility.Vector3dVector(cs2)  # 定义点云的颜色
            # test2_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # lines2_pcd = o3d.geometry.LineSet()
            # lines2_pcd.lines = o3d.utility.Vector2iVector(line)
            # lines2_pcd.colors = o3d.utility.Vector3dVector(cl2)  # 线条颜色
            # lines2_pcd.points = o3d.utility.Vector3dVector(joints_ak_2[ff * 32:(ff + 1) * 32])
            # lines2_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            #
            # bp_pcd = o3d.geometry.LineSet()
            # bp_pcd.lines = o3d.utility.Vector2iVector(line_bp)
            # bp_pcd.colors = o3d.utility.Vector3dVector(cl_bp)  # 线条颜色
            # bp_pcd.points = o3d.utility.Vector3dVector(bpStep1)
            # bpp_pcd = o3d.geometry.PointCloud()
            # bpp_pcd.points = o3d.utility.Vector3dVector(bpStep1)
            # bpp_pcd.colors = o3d.utility.Vector3dVector(cl_bp)
            # bp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # bpp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # o3d.visualization.draw_geometries(
            #     [bp_pcd, bpp_pcd, test0_pcd, lines0_pcd,test1_pcd, lines1_pcd,test2_pcd, lines2_pcd],
            #     window_name=f"BP joints points {ff}")
            # '''画图end 头没有画 '''
            #
            # bg = cv2.imread(f'./data/fw/image0/image{ff}.bmp')
            # in_param = np.loadtxt('./param/0_rgb_in.txt')
            # uv1 = np.dot(in_param, bpStep1.T) / bpStep1.T[2]
            # for i in range(uv1.shape[1]):
            #     u = int(uv1[0, i])
            #     v = int(uv1[1, i])
            #     cv2.circle(bg, (u, v), 2, (0, 255, 0))
            # cv2.imshow(f'{ff}', bg)
            # cv2.waitKey(0)
    @classmethod
    def verifyBP(cls, frame_num: int):
        from dataStructure import belief_propagation, string2factor_graph, factor
        import open3d as o3d
        import cv2
        import random
        # 一帧
        joints_ak_0 = np.loadtxt('./data/fw/data0.txt')[frame_num * 32:(frame_num + 1) * 32, :]
        joints_ak_1 = np.loadtxt('./data/fw/data1.txt')[frame_num * 32:(frame_num + 1) * 32, :]
        joints_ak_2 = np.loadtxt('./data/fw/data2.txt')[frame_num * 32:(frame_num + 1) * 32, :]
        # 异化：ak1和2随机{1,2}个 ，关节点坐标随机{x,y,z}→0.5-2倍
        # dissimilation = random.random()
        # if dissimilation<=0.5:
        #     dissimilation=1
        # else:
        #     dissimilation>0.5
        #     dissimilation = 2
        # coor = random.random()
        # if coor<=0.333:
        #     coor = 1
        # elif 0.333<coor<=0.667:
        #     coor=2
        # else:
        #     coor=3
        # joints_ak_1, joints_ak_2 = utils.dissimilation(dissimilation,coor,joints_ak_1,joints_ak_2)
        # 初始化
        mrf = string2factor_graph(
            f'f0({K4ABT_JOINT_NAMES[0]})f1({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[1]})f2({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[18]})f3({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[22]})f4({K4ABT_JOINT_NAMES[18]},{K4ABT_JOINT_NAMES[19]})f5({K4ABT_JOINT_NAMES[20]},{K4ABT_JOINT_NAMES[19]})f6({K4ABT_JOINT_NAMES[21]},{K4ABT_JOINT_NAMES[20]})f7({K4ABT_JOINT_NAMES[22]},{K4ABT_JOINT_NAMES[23]})f8({K4ABT_JOINT_NAMES[23]},{K4ABT_JOINT_NAMES[24]})f9({K4ABT_JOINT_NAMES[24]},{K4ABT_JOINT_NAMES[25]})f10({K4ABT_JOINT_NAMES[1]},{K4ABT_JOINT_NAMES[2]})f11({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[3]})f12({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[4]})f13({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[11]})f14({K4ABT_JOINT_NAMES[3]},{K4ABT_JOINT_NAMES[26]})f15({K4ABT_JOINT_NAMES[4]},{K4ABT_JOINT_NAMES[5]})f16({K4ABT_JOINT_NAMES[5]},{K4ABT_JOINT_NAMES[6]})f17({K4ABT_JOINT_NAMES[6]},{K4ABT_JOINT_NAMES[7]})f18({K4ABT_JOINT_NAMES[7]},{K4ABT_JOINT_NAMES[8]})f19({K4ABT_JOINT_NAMES[7]},{K4ABT_JOINT_NAMES[10]})f20({K4ABT_JOINT_NAMES[8]},{K4ABT_JOINT_NAMES[9]})f21({K4ABT_JOINT_NAMES[11]},{K4ABT_JOINT_NAMES[12]})f22({K4ABT_JOINT_NAMES[12]},{K4ABT_JOINT_NAMES[13]})f23({K4ABT_JOINT_NAMES[13]},{K4ABT_JOINT_NAMES[14]})f24({K4ABT_JOINT_NAMES[14]},{K4ABT_JOINT_NAMES[15]})f25({K4ABT_JOINT_NAMES[14]},{K4ABT_JOINT_NAMES[17]})f26({K4ABT_JOINT_NAMES[15]},{K4ABT_JOINT_NAMES[16]})f27({K4ABT_JOINT_NAMES[1]})f28({K4ABT_JOINT_NAMES[2]})f29({K4ABT_JOINT_NAMES[3]})f30({K4ABT_JOINT_NAMES[4]})f31({K4ABT_JOINT_NAMES[5]})f32({K4ABT_JOINT_NAMES[6]})f33({K4ABT_JOINT_NAMES[7]})f34({K4ABT_JOINT_NAMES[8]})f35({K4ABT_JOINT_NAMES[9]})f36({K4ABT_JOINT_NAMES[10]})f37({K4ABT_JOINT_NAMES[11]})f38({K4ABT_JOINT_NAMES[12]})f39({K4ABT_JOINT_NAMES[13]})f40({K4ABT_JOINT_NAMES[14]})f41({K4ABT_JOINT_NAMES[15]})f42({K4ABT_JOINT_NAMES[16]})f43({K4ABT_JOINT_NAMES[17]})f44({K4ABT_JOINT_NAMES[18]})f45({K4ABT_JOINT_NAMES[19]})f46({K4ABT_JOINT_NAMES[20]})f47({K4ABT_JOINT_NAMES[21]})f48({K4ABT_JOINT_NAMES[22]})f49({K4ABT_JOINT_NAMES[23]})f50({K4ABT_JOINT_NAMES[24]})f51({K4ABT_JOINT_NAMES[25]})f52({K4ABT_JOINT_NAMES[26]})')
        # 对齐
        T = np.loadtxt(f'param/0_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        joints_ak_0 = np.dot(joints_ak_0, r.T) + t  # 在ak0的rgb坐标系中了
        T = np.loadtxt(f'param/1_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        joints_ak_1 = np.dot(joints_ak_1, r.T) + t  # 在ak1的rgb坐标系中了
        joints_ak_1 = utils.sub2MasterRT(1, joints_ak_1)
        T = np.loadtxt(f'param/2_rgb_ex.txt')
        r = T[:, :3].copy()
        t = T[:, 3].copy().reshape([1, 3])
        joints_ak_2 = np.dot(joints_ak_2, r.T) + t  # 在ak2的rgb坐标系中了
        joints_ak_2 = utils.sub2MasterRT(2, joints_ak_2)
        # 离散化
        affine_l = [1, 2, 3, 4, 5, 6, 7, 8]
        dist_set = []
        for lamda in affine_l:
            for beta in affine_l[:9 - lamda]:
                joint_bar = lamda * joints_ak_0 / 10 + beta * joints_ak_1 / 10 + (10 - lamda - beta) * joints_ak_2 / 10
                dist_set.append(joint_bar)
        joint_bar_set = np.array(dist_set).reshape(-1, 32, 3)
        distri = o3d.geometry.PointCloud()
        joint_bar_set = joint_bar_set.reshape([-1,3])
        dcl = np.zeros(joint_bar_set.shape)
        distri.points = o3d.utility.Vector3dVector(joint_bar_set)
        distri.colors = o3d.utility.Vector3dVector(dcl)
        distri.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        joint_bar_set = joint_bar_set.reshape(-1, 32, 3)
        # 骨长
        bl0 = utils.bonesLengthMid(np.loadtxt('./data/fw/data0.txt'))
        bl1 = utils.bonesLengthMid(np.loadtxt('./data/fw/data1.txt'))
        bl2 = utils.bonesLengthMid(np.loadtxt('./data/fw/data2.txt'))
        bld = utils.dictAtMidValue(bl0,bl1,bl2)
        # BP
        joints_score = utils.threeAKs(joints_ak_0, joints_ak_1, joints_ak_2, joint_bar_set)
        # 27个(36,36)的骨长关系也加上
        # 0-1 0-18 0-22
        f0 = factor([K4ABT_JOINT_NAMES[0]], joints_score[:, 0])  # joint 36个状态
        f27 = factor([K4ABT_JOINT_NAMES[1]], joints_score[:, 1])
        distance01 = utils.distriDistance(joint_bar_set[:, 0, :], joint_bar_set[:, 1, :],
                                          bld['0-1'])  # pairwise (36,36)个状态
        f1 = factor([K4ABT_JOINT_NAMES[0], K4ABT_JOINT_NAMES[1]], distance01)

        f44 = factor([K4ABT_JOINT_NAMES[18]], joints_score[:, 18])
        distance018 = utils.distriDistance(joint_bar_set[:, 0, :], joint_bar_set[:, 18, :], bld['0-18'])
        f2 = factor([K4ABT_JOINT_NAMES[0], K4ABT_JOINT_NAMES[18]], distance018)

        f48 = factor([K4ABT_JOINT_NAMES[22]], joints_score[:, 22])
        distance022 = utils.distriDistance(joint_bar_set[:, 0, :], joint_bar_set[:, 22, :], bld['0-22'])
        f3 = factor([K4ABT_JOINT_NAMES[0], K4ABT_JOINT_NAMES[22]], distance022)

        # 18-19
        f45 = factor([K4ABT_JOINT_NAMES[19]], joints_score[:, 19])
        distance1819 = utils.distriDistance(joint_bar_set[:, 18, :], joint_bar_set[:, 19, :], bld['18-19'])
        f4 = factor([K4ABT_JOINT_NAMES[18], K4ABT_JOINT_NAMES[19]], distance1819)

        # 19-20
        f46 = factor([K4ABT_JOINT_NAMES[20]], joints_score[:, 20])
        distance1920 = utils.distriDistance(joint_bar_set[:, 19, :], joint_bar_set[:, 20, :], bld['19-20'])
        f5 = factor([K4ABT_JOINT_NAMES[19], K4ABT_JOINT_NAMES[20]], distance1920)

        # 20-21
        f47 = factor([K4ABT_JOINT_NAMES[21]], joints_score[:, 21])
        distance2021 = utils.distriDistance(joint_bar_set[:, 20, :], joint_bar_set[:, 21, :], bld['20-21'])
        f6 = factor([K4ABT_JOINT_NAMES[20], K4ABT_JOINT_NAMES[21]], distance2021)

        # 22-23
        f48 = factor([K4ABT_JOINT_NAMES[22]], joints_score[:, 22])
        f49 = factor([K4ABT_JOINT_NAMES[23]], joints_score[:, 23])
        distance2223 = utils.distriDistance(joint_bar_set[:, 22, :], joint_bar_set[:, 23, :], bld['22-23'])
        f7 = factor([K4ABT_JOINT_NAMES[22], K4ABT_JOINT_NAMES[23]], distance2223)

        # 23-24
        f50 = factor([K4ABT_JOINT_NAMES[24]], joints_score[:, 24])
        distance2324 = utils.distriDistance(joint_bar_set[:, 23, :], joint_bar_set[:, 24, :], bld['23-24'])
        f8 = factor([K4ABT_JOINT_NAMES[24], K4ABT_JOINT_NAMES[23]], distance2324)

        # 24-25
        f51 = factor([K4ABT_JOINT_NAMES[25]], joints_score[:, 25])
        distance2425 = utils.distriDistance(joint_bar_set[:, 24, :], joint_bar_set[:, 25, :], bld['24-25'])
        f9 = factor([K4ABT_JOINT_NAMES[24], K4ABT_JOINT_NAMES[25]], distance2425)

        # 1-2
        f28 = factor([K4ABT_JOINT_NAMES[2]], joints_score[:, 2])
        distance12 = utils.distriDistance(joint_bar_set[:, 1, :], joint_bar_set[:, 2, :], bld['1-2'])
        f10 = factor([K4ABT_JOINT_NAMES[2], K4ABT_JOINT_NAMES[1]], distance12)

        # 2-3 2-4 2-11
        f29 = factor([K4ABT_JOINT_NAMES[3]], joints_score[:, 3])
        f30 = factor([K4ABT_JOINT_NAMES[4]], joints_score[:, 4])
        f37 = factor([K4ABT_JOINT_NAMES[11]], joints_score[:, 11])
        distance23 = utils.distriDistance(joint_bar_set[:, 2, :], joint_bar_set[:, 3, :], bld['2-3'])
        f11 = factor([K4ABT_JOINT_NAMES[2], K4ABT_JOINT_NAMES[3]], distance23)

        distance24 = utils.distriDistance(joint_bar_set[:, 2, :], joint_bar_set[:, 4, :], bld['2-4'])
        f12 = factor([K4ABT_JOINT_NAMES[2], K4ABT_JOINT_NAMES[4]], distance24)

        distance211 = utils.distriDistance(joint_bar_set[:, 2, :], joint_bar_set[:, 11, :], bld['2-11'])
        f13 = factor([K4ABT_JOINT_NAMES[2], K4ABT_JOINT_NAMES[11]], distance211)

        # 3-26
        f52 = factor([K4ABT_JOINT_NAMES[26]], joints_score[:, 26])
        distance326 = utils.distriDistance(joint_bar_set[:, 3, :], joint_bar_set[:, 26, :], bld['3-26'])
        f14 = factor([K4ABT_JOINT_NAMES[3], K4ABT_JOINT_NAMES[26]], distance326)

        # 4-5
        f31 = factor([K4ABT_JOINT_NAMES[5]], joints_score[:, 5])
        distance45 = utils.distriDistance(joint_bar_set[:, 4, :], joint_bar_set[:, 5, :], bld['4-5'])
        f15 = factor([K4ABT_JOINT_NAMES[5], K4ABT_JOINT_NAMES[4]], distance45)

        # 5-6
        f32 = factor([K4ABT_JOINT_NAMES[6]], joints_score[:, 6])
        distance56 = utils.distriDistance(joint_bar_set[:, 4, :], joint_bar_set[:, 5, :], bld['5-6'])
        f16 = factor([K4ABT_JOINT_NAMES[5], K4ABT_JOINT_NAMES[6]], distance56)

        # 6-7
        f33 = factor([K4ABT_JOINT_NAMES[7]], joints_score[:, 7])
        distance67 = utils.distriDistance(joint_bar_set[:, 6, :], joint_bar_set[:, 7, :], bld['6-7'])
        f17 = factor([K4ABT_JOINT_NAMES[6], K4ABT_JOINT_NAMES[7]], distance67)

        # 7-8 7-10
        f34 = factor([K4ABT_JOINT_NAMES[8]], joints_score[:, 8])
        distance78 = utils.distriDistance(joint_bar_set[:, 7, :], joint_bar_set[:, 8, :], bld['7-8'])
        f18 = factor([K4ABT_JOINT_NAMES[7], K4ABT_JOINT_NAMES[8]], distance78)

        f36 = factor([K4ABT_JOINT_NAMES[10]], joints_score[:, 10])
        distance710 = utils.distriDistance(joint_bar_set[:, 7, :], joint_bar_set[:, 10, :], bld['7-10'])
        f19 = factor([K4ABT_JOINT_NAMES[10], K4ABT_JOINT_NAMES[7]], distance710)

        # 8-9
        f35 = factor([K4ABT_JOINT_NAMES[9]], joints_score[:, 9])
        distance89 = utils.distriDistance(joint_bar_set[:, 8, :], joint_bar_set[:, 9, :], bld['8-9'])
        f20 = factor([K4ABT_JOINT_NAMES[8], K4ABT_JOINT_NAMES[9]], distance89)

        # 11-12
        f38 = factor([K4ABT_JOINT_NAMES[12]], joints_score[:, 12])
        distance1112 = utils.distriDistance(joint_bar_set[:, 11, :], joint_bar_set[:, 12, :], bld['11-12'])
        f21 = factor([K4ABT_JOINT_NAMES[11], K4ABT_JOINT_NAMES[12]], distance1112)

        # 12-13
        f39 = factor([K4ABT_JOINT_NAMES[13]], joints_score[:, 13])
        distance1213 = utils.distriDistance(joint_bar_set[:, 12, :], joint_bar_set[:, 13, :], bld['12-13'])
        f22 = factor([K4ABT_JOINT_NAMES[12], K4ABT_JOINT_NAMES[13]], distance1213)

        # 13-14
        f40 = factor([K4ABT_JOINT_NAMES[14]], joints_score[:, 14])
        distance1314 = utils.distriDistance(joint_bar_set[:, 13, :], joint_bar_set[:, 14, :], bld['13-14'])
        f23 = factor([K4ABT_JOINT_NAMES[13], K4ABT_JOINT_NAMES[14]], distance1314)

        # 14-15 14-17
        f41 = factor([K4ABT_JOINT_NAMES[15]], joints_score[:, 15])
        distance1415 = utils.distriDistance(joint_bar_set[:, 14, :], joint_bar_set[:, 15, :], bld['14-15'])
        f24 = factor([K4ABT_JOINT_NAMES[14], K4ABT_JOINT_NAMES[15]], distance1415)

        f43 = factor([K4ABT_JOINT_NAMES[17]], joints_score[:, 17])
        distance1417 = utils.distriDistance(joint_bar_set[:, 14, :], joint_bar_set[:, 17, :], bld['14-17'])
        f25 = factor([K4ABT_JOINT_NAMES[14], K4ABT_JOINT_NAMES[17]], distance1417)

        # 15-16
        f42 = factor([K4ABT_JOINT_NAMES[16]], joints_score[:, 16])
        distance1516 = utils.distriDistance(joint_bar_set[:, 15, :], joint_bar_set[:, 16, :], bld['15-16'])
        f26 = factor([K4ABT_JOINT_NAMES[16], K4ABT_JOINT_NAMES[15]], distance1516)
        names = locals()
        for n in range(53):  # 动态变量名
            mrf.change_factor_distribution(f'f{n}', names['f%s' % n])
        bp = belief_propagation(mrf)
        answer = []
        for jidex, j in enumerate(K4ABT_JOINT_NAMES):
            if j in ["nose", "left eye", "left ear",
                     "right eye", "right ear", "left hand", "left handtip", "left thumb","right hand", "right handtip", "right thumb"]:
                continue
            score_array = bp.belief(j).get_distribution()
            idx = np.argmax(score_array)
            answer.append(joint_bar_set[idx, jidex, :])
        bpStep1 = np.array(answer)
        '''画图 start'''
        line = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 7], [11, 2], [12, 11],
                [13, 12], [14, 13], [15, 14], [16, 15], [17, 14], [18, 0], [19, 18], [20, 19], [21, 20], [22, 0],
                [23, 22], [24, 23], [25, 24], [26, 3], [27, 26], [28, 26], [29, 26], [30, 26], [31, 26]]
        cl0 = [[1, 0, 0] for i in range(len(line))]
        cl1 = [[0, 1, 0] for j in range(len(line))]
        cl2 = [[0, 0, 1] for k in range(len(line))]
        cs0 = np.zeros([32, 3])
        cs0[:, 0] = 1  # red
        cs1 = np.zeros([32, 3])
        cs1[:, 1] = 1  # green
        cs2 = np.zeros([32, 3])
        cs2[:, 2] = 1  # blue
        line_bp = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 4], [6, 5], [7, 6], [8, 2], [8, 9],
                   [9, 10], [10, 11], [12, 0], [12, 13], [13, 14], [14, 15], [16, 0],
                   [16, 17], [17, 18], [18, 19], [20, 3]]
        cl_bp = [[0, 0, 0] for l in range(len(line_bp))]
        test0_pcd = o3d.geometry.PointCloud()
        test0_pcd.points = o3d.utility.Vector3dVector(joints_ak_0)  # 定义点云坐标位置
        test0_pcd.colors = o3d.utility.Vector3dVector(cs0)  # 定义点云的颜色
        test0_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lines0_pcd = o3d.geometry.LineSet()
        lines0_pcd.lines = o3d.utility.Vector2iVector(line)
        lines0_pcd.colors = o3d.utility.Vector3dVector(cl0)  # 线条颜色
        lines0_pcd.points = o3d.utility.Vector3dVector(joints_ak_0)
        lines0_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        test1_pcd = o3d.geometry.PointCloud()
        test1_pcd.points = o3d.utility.Vector3dVector(joints_ak_1)  # 定义点云坐标位置
        test1_pcd.colors = o3d.utility.Vector3dVector(cs1)  # 定义点云的颜色
        test1_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lines1_pcd = o3d.geometry.LineSet()
        lines1_pcd.lines = o3d.utility.Vector2iVector(line)
        lines1_pcd.colors = o3d.utility.Vector3dVector(cl1)  # 线条颜色
        lines1_pcd.points = o3d.utility.Vector3dVector(joints_ak_1)
        lines1_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        test2_pcd = o3d.geometry.PointCloud()
        test2_pcd.points = o3d.utility.Vector3dVector(joints_ak_2)  # 定义点云坐标位置
        test2_pcd.colors = o3d.utility.Vector3dVector(cs2)  # 定义点云的颜色
        test2_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lines2_pcd = o3d.geometry.LineSet()
        lines2_pcd.lines = o3d.utility.Vector2iVector(line)
        lines2_pcd.colors = o3d.utility.Vector3dVector(cl2)  # 线条颜色
        lines2_pcd.points = o3d.utility.Vector3dVector(joints_ak_2)
        lines2_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        bp_pcd = o3d.geometry.LineSet()
        bp_pcd.lines = o3d.utility.Vector2iVector(line_bp)
        bp_pcd.colors = o3d.utility.Vector3dVector(cl_bp)  # 线条颜色
        bp_pcd.points = o3d.utility.Vector3dVector(bpStep1)
        bpp_pcd = o3d.geometry.PointCloud()
        bpp_pcd.points = o3d.utility.Vector3dVector(bpStep1)
        bpp_pcd.colors = o3d.utility.Vector3dVector(cl_bp)
        bp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        bpp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries(
            [test0_pcd, lines0_pcd,test1_pcd, lines1_pcd,test2_pcd, lines2_pcd],
            window_name="ak points")
        o3d.visualization.draw_geometries(
            [bp_pcd, bpp_pcd, distri],
            window_name="BP points")
        '''画图end 头没有画 '''

        bg = cv2.imread(f'./data/fw/image0/image{frame_num}.bmp')
        in_param = np.loadtxt('./param/0_rgb_in.txt')
        uv1 = np.dot(in_param, bpStep1.T) / bpStep1.T[2]
        for i in range(uv1.shape[1]):
            u = int(uv1[0, i])
            v = int(uv1[1, i])
            cv2.circle(bg, (u, v), 2, (0, 255, 0))
        cv2.imshow('p', bg)
        cv2.waitKey(0)
def testnp():
    s = time.time()
    a = np.arange(300)
    print(np.median(a),time.time()-s)
    s = time.time()
    a = np.arange(100)
    b = np.arange(100)
    c = np.arange(100)
    print(np.median([a,b,c]),time.time()-s)
if __name__ == '__main__':
    # 验证一下ak的人体追踪 √
    # DataSelect.record()
    # 得到内参 √
    # DataProcess.getAKintrisics(0,'param')
    # 验证一下data经过rt，然后画到对应的rgb上
    # DataProcess.verfyJointsData(1, 50)
    # 验证λ和β的效果
    # DataProcess.jointsIn3D_interpolation()
    # BP测速
    DataProcess.BPapplication()
    # 验证BP使用
    # DataProcess.verifyBP(90)
    # 测试numpy
    # testnp()
    pass
