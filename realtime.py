import time

K4ABT_JOINT_NAMES = ["pelvis", "spine - navel", "spine - chest", "neck", "left clavicle", "left shoulder",
                     "left elbow",
                     "left wrist", "left hand", "left handtip", "left thumb", "right clavicle",
                     "right shoulder",
                     "right elbow",
                     "right wrist", "right hand", "right handtip", "right thumb", "left hip", "left knee",
                     "left ankle",
                     "left foot",
                     "right hip", "right knee", "right ankle", "right foot", "head", "nose", "left eye",
                     "left ear",
                     "right eye", "right ear"]


class Realtime(object):
    def __init__(self):
        pass

    @classmethod
    def start(cls):
        from dataProcess import DataProcess
        from ak.camera_synchronous.core_threading import MulDeviceSynCapture
        from dataStructure import string2factor_graph
        import numpy as np
        import cv2
        midFilter = []
        cv2.namedWindow("master", 0);
        cv2.resizeWindow("master", 640, 360);
        cv2.namedWindow("sub1", 0);
        cv2.resizeWindow("sub1", 640, 360);
        cv2.namedWindow("sub2", 0);
        cv2.resizeWindow("sub2", 640, 360);
        cv2.namedWindow("BP", 0);
        cv2.resizeWindow("BP", 640, 360);
        T = locals()
        for i in range(3):
            T['%s' % i] = np.loadtxt(f'param/{i}_rgb_ex.txt')
        K0 = np.loadtxt('./param/0_rgb_in.txt')
        mrf = string2factor_graph(
            f'f0({K4ABT_JOINT_NAMES[0]})f1({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[1]})f2({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[18]})f3({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[22]})f4({K4ABT_JOINT_NAMES[18]},{K4ABT_JOINT_NAMES[19]})f5({K4ABT_JOINT_NAMES[20]},{K4ABT_JOINT_NAMES[19]})f6({K4ABT_JOINT_NAMES[21]},{K4ABT_JOINT_NAMES[20]})f7({K4ABT_JOINT_NAMES[22]},{K4ABT_JOINT_NAMES[23]})f8({K4ABT_JOINT_NAMES[23]},{K4ABT_JOINT_NAMES[24]})f9({K4ABT_JOINT_NAMES[24]},{K4ABT_JOINT_NAMES[25]})f10({K4ABT_JOINT_NAMES[1]},{K4ABT_JOINT_NAMES[2]})f11({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[3]})f12({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[4]})f13({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[11]})f14({K4ABT_JOINT_NAMES[3]},{K4ABT_JOINT_NAMES[26]})f15({K4ABT_JOINT_NAMES[4]},{K4ABT_JOINT_NAMES[5]})f16({K4ABT_JOINT_NAMES[5]},{K4ABT_JOINT_NAMES[6]})f17({K4ABT_JOINT_NAMES[6]},{K4ABT_JOINT_NAMES[7]})f21({K4ABT_JOINT_NAMES[11]},{K4ABT_JOINT_NAMES[12]})f22({K4ABT_JOINT_NAMES[12]},{K4ABT_JOINT_NAMES[13]})f23({K4ABT_JOINT_NAMES[13]},{K4ABT_JOINT_NAMES[14]})f27({K4ABT_JOINT_NAMES[1]})f28({K4ABT_JOINT_NAMES[2]})f29({K4ABT_JOINT_NAMES[3]})f30({K4ABT_JOINT_NAMES[4]})f31({K4ABT_JOINT_NAMES[5]})f32({K4ABT_JOINT_NAMES[6]})f33({K4ABT_JOINT_NAMES[7]})f37({K4ABT_JOINT_NAMES[11]})f38({K4ABT_JOINT_NAMES[12]})f39({K4ABT_JOINT_NAMES[13]})f40({K4ABT_JOINT_NAMES[14]})f44({K4ABT_JOINT_NAMES[18]})f45({K4ABT_JOINT_NAMES[19]})f46({K4ABT_JOINT_NAMES[20]})f47({K4ABT_JOINT_NAMES[21]})f48({K4ABT_JOINT_NAMES[22]})f49({K4ABT_JOINT_NAMES[23]})f50({K4ABT_JOINT_NAMES[24]})f51({K4ABT_JOINT_NAMES[25]})f52({K4ABT_JOINT_NAMES[26]})')
        mc = MulDeviceSynCapture(0, [1, 2])
        # mc = MulDeviceSynCapture(0)
        j0, j1, j2 = np.zeros([32, 3]), np.zeros([32, 3]), np.zeros([32, 3])
        Rt10 = np.loadtxt('./param/1_0_abt_Rt.txt')
        Rt20 = np.loadtxt('./param/2_0_abt_Rt.txt')
        key = 0
        lines = []
        jl = []
        while key & 0xFF != 27:
            s = time.time()
            ret = mc.get()
            if ret[0][2] == 1 and ret[1][2] == 1 and ret[2][2] == 1:
                for i in range(32):
                    j0[i] = np.array(
                        [ret[0][3].joints[i].position.x, ret[0][3].joints[i].position.y,
                         ret[0][3].joints[i].position.z])
                    j1[i] = np.array(
                        [ret[1][3].joints[i].position.x, ret[1][3].joints[i].position.y,
                         ret[2][3].joints[i].position.z])
                    j2[i] = np.array(
                        [ret[2][3].joints[i].position.x, ret[2][3].joints[i].position.y,
                         ret[2][3].joints[i].position.z])
                # todo add affine_l division
                bpj = DataProcess.singleFrame(T['0'], T['1'], T['2'], mrf, j0, j1, j2, [1, 2, 3, 4, 5, 6, 7, 8], Rt10,
                                              Rt20)
                # midFilter.append(bpj)
                # if len(midFilter) < 3:
                #     continue
                # else:
                #     bpj = np.array(midFilter)
                #     bpj = np.median(bpj,axis=0)
                #     midFilter.pop(0)
                uv1 = np.dot(K0, bpj.T) / bpj.T[2]
                for i in range(uv1.shape[1]):
                    u = int(uv1[0, i])
                    v = int(uv1[1, i])
                    cv2.circle(ret[0][1], (u, v), 5, (0, 255, 0), 3)
                    lines.append((u, v))
                cv2.line(ret[0][1], lines[0], lines[1], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[1], lines[2], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[2], lines[3], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[2], lines[4], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[4], lines[5], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[5], lines[6], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[6], lines[7], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[2], lines[8], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[8], lines[9], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[9], lines[10], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[10], lines[11], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[0], lines[12], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[12], lines[13], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[13], lines[14], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[14], lines[15], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[0], lines[16], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[16], lines[17], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[17], lines[18], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[18], lines[19], (0, 255, 0), 3)
                cv2.line(ret[0][1], lines[3], lines[20], (0, 255, 0), 3)
                jl.append([j0.copy()])
                jl.append([j1.copy()])
                jl.append([j2.copy()])
                jl.append([bpj])
            else:
                pass
            cv2.imshow('master', ret[0][4])
            cv2.imshow('sub1', ret[1][4])
            cv2.imshow('sub2', ret[2][4])
            cv2.imshow('BP', ret[0][1])
            key = cv2.waitKey(1)
            lines.clear()
        mc.close()
        cv2.destroyAllWindows()
        jw = np.array(jl)
        np.savetxt('./data/zsjrt.txt', jw,delimiter=' ' ,fmt = '%s')

def draw(start):
    import numpy as np
    import open3d as o3d
    '''加载数据'''
    joints = np.loadtxt('./data/zsjrt.txt')
    joints = joints.reshape((-1,117,3))
    '''加载数据完毕'''

    '''open3d显示用的，点的颜色和线的颜色'''
    cs0 = np.zeros([32, 3])
    cs1 = np.zeros([32, 3])
    cs2 = np.zeros([32, 3])
    cs3 = np.zeros([21, 3])
    cs0[:, 0] = 1
    cs1[:, 1] = 1
    cs2[:, 2] = 1
    line = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 7], [11, 2], [12, 11],
            [13, 12], [14, 13], [15, 14], [16, 15], [17, 14], [18, 0], [19, 18], [20, 19], [21, 20], [22, 0],
            [23, 22], [24, 23], [25, 24], [26, 3], [27, 26], [28, 26], [29, 26], [30, 26], [31, 26]]
    cl0 = [[1, 0, 0] for i in range(len(line))]
    cl1 = [[0, 1, 0] for j in range(len(line))]
    cl2 = [[0, 0, 1] for k in range(len(line))]
    line_bp = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 4], [6, 5], [7, 6], [8, 2], [8, 9],
               [9, 10], [10, 11], [12, 0], [12, 13], [13, 14], [14, 15], [16, 0],
               [16, 17], [17, 18], [18, 19], [20, 3]]
    cl_bp = [[0, 0, 0] for l in range(len(line_bp))]
    '''open3d显示用的'''

    '''加载参数，三个dep到rgb的rt和两个rgb到rgb的rt'''
    T0 = np.loadtxt(f'param/0_rgb_ex.txt')
    T1 = np.loadtxt(f'param/1_rgb_ex.txt')
    T2 = np.loadtxt(f'param/2_rgb_ex.txt')
    Rt10 = np.loadtxt('./param/1_0_abt_Rt.txt')
    Rt20 = np.loadtxt('./param/2_0_abt_Rt.txt')
    '''加载参数，三个dep到rgb的rt和两个rgb到rgb的rt，加载完毕'''

    affine_l = [1, 2, 3, 4, 5, 6, 7, 8]
    for f in range(start,joints.shape[0]):
        '''所有的点都放到主相机rgb空间'''
        ak0 = joints[f,:32,:]
        ak1 = joints[f,32:64,:]
        ak2 = joints[f,64:96,:]
        ak0 = np.dot(ak0, T0[:, :3].T) + T0[:, 3].reshape([1, 3])
        ak1 = np.dot(ak1, T1[:, :3].T) + T1[:, 3].reshape([1, 3])
        ak2 = np.dot(ak2, T2[:, :3].T) + T2[:, 3].reshape([1, 3])
        ak1 = np.dot(ak1, Rt10[:3, :3].reshape([3, 3]).T) + Rt10[:3, 3].reshape(1, 3)
        ak2 = np.dot(ak2, Rt20[:3, :3].reshape([3, 3]).T) + Rt20[:3, 3].reshape(1, 3)
        bp = joints[f,96:117,:]
        '''所有的点都放到主相机rgb空间'''
        # 无论对齐与否，bp结果都应在观察值之内
        dist_set = []
        for lamda in affine_l:
            for beta in affine_l[:9 - lamda]:
                joint_bar = lamda * ak0 / 10 + beta * ak1 / 10 + (10 - lamda - beta) * ak2 / 10
                dist_set.append(joint_bar)
        joint_bar_set = np.array(dist_set).reshape(-1, 3)
        '''点云，0-red，1-green，2-blue，3-black'''
        test0_pcd = o3d.geometry.PointCloud()
        test0_pcd.points = o3d.utility.Vector3dVector(ak0)  # 定义点云坐标位置
        test0_pcd.colors = o3d.utility.Vector3dVector(cs0)  # 定义点云的颜色
        test0_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        test1_pcd = o3d.geometry.PointCloud()
        test1_pcd.points = o3d.utility.Vector3dVector(ak1)  # 定义点云坐标位置
        test1_pcd.colors = o3d.utility.Vector3dVector(cs1)  # 定义点云的颜色
        test1_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        test2_pcd = o3d.geometry.PointCloud()
        test2_pcd.points = o3d.utility.Vector3dVector(ak2)  # 定义点云坐标位置
        test2_pcd.colors = o3d.utility.Vector3dVector(cs2)  # 定义点云的颜色
        test2_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        test3_pcd = o3d.geometry.PointCloud()
        test3_pcd.points = o3d.utility.Vector3dVector(bp)  # 定义点云坐标位置
        test3_pcd.colors = o3d.utility.Vector3dVector(cs3)  # 定义点云的颜色
        test3_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        test4_pcd = o3d.geometry.PointCloud()
        test4_pcd.points = o3d.utility.Vector3dVector(joint_bar_set)  # 定义点云坐标位置
        test4_pcd.colors = o3d.utility.Vector3dVector(cs3)  # 定义点云的颜色
        test4_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        '''点云，0-red，1-green，2-blue，3-black，设置完毕'''

        '''连线，0-red，1-green，2-blue，3-black'''
        lines0_pcd = o3d.geometry.LineSet()
        lines0_pcd.lines = o3d.utility.Vector2iVector(line)
        lines0_pcd.colors = o3d.utility.Vector3dVector(cl0)  # 线条颜色
        lines0_pcd.points = o3d.utility.Vector3dVector(ak0)
        lines0_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lines1_pcd = o3d.geometry.LineSet()
        lines1_pcd.lines = o3d.utility.Vector2iVector(line)
        lines1_pcd.colors = o3d.utility.Vector3dVector(cl1)  # 线条颜色
        lines1_pcd.points = o3d.utility.Vector3dVector(ak1)
        lines1_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lines2_pcd = o3d.geometry.LineSet()
        lines2_pcd.lines = o3d.utility.Vector2iVector(line)
        lines2_pcd.colors = o3d.utility.Vector3dVector(cl2)  # 线条颜色
        lines2_pcd.points = o3d.utility.Vector3dVector(ak2)
        lines2_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        lines3_pcd = o3d.geometry.LineSet()
        lines3_pcd.lines = o3d.utility.Vector2iVector(line_bp)
        lines3_pcd.colors = o3d.utility.Vector3dVector(cl_bp)  # 线条颜色
        lines3_pcd.points = o3d.utility.Vector3dVector(bp)
        lines3_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        '''连线，0-red，1-green，2-blue，3-black，设置完毕'''
        o3d.visualization.draw_geometries(
            [test0_pcd, lines0_pcd,test1_pcd, lines1_pcd,test2_pcd, lines2_pcd,test3_pcd, lines3_pcd,test4_pcd],
            window_name=f"BP joints points {f}")
if __name__ == '__main__':
    Realtime.start()
    # draw(50)