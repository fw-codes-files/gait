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
        cv2.namedWindow("master", 0);
        cv2.resizeWindow("master", 640, 480);
        cv2.namedWindow("sub1", 0);
        cv2.resizeWindow("sub1", 640, 480);
        cv2.namedWindow("sub2", 0);
        cv2.resizeWindow("sub2", 640, 480);
        cv2.namedWindow("BP", 0);
        cv2.resizeWindow("BP", 640, 480);
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
        n = 0
        lines = []
        while key & 0xFF != 27:
            s = time.time()
            ret = mc.get()
            for i in range(32):
                j0[i] = np.array(
                    [ret[0][2].joints[i].position.x, ret[0][2].joints[i].position.y, ret[0][2].joints[i].position.z])
                j1[i] = np.array(
                    [ret[1][2].joints[i].position.x, ret[1][2].joints[i].position.y, ret[2][2].joints[i].position.z])
                j2[i] = np.array(
                    [ret[2][2].joints[i].position.x, ret[2][2].joints[i].position.y, ret[2][2].joints[i].position.z])
            bpj = DataProcess.singleFrame(T['0'], T['1'], T['2'], mrf, j0, j1, j2, [1, 2, 3, 4, 5, 6, 7, 8], Rt10, Rt20)
            cv2.imshow('master', ret[0][3])
            cv2.imshow('sub1', ret[1][3])
            cv2.imshow('sub2', ret[2][3])
            uv1 = np.dot(K0, bpj.T) / bpj.T[2]
            for i in range(uv1.shape[1]):
                u = int(uv1[0, i])
                v = int(uv1[1, i])
                cv2.circle(ret[0][1], (u, v), 5, (0, 255, 0), 5)
                lines.append((u, v))
            cv2.line(ret[0][1], lines[0], lines[1], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[1], lines[2], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[2], lines[3], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[2], lines[4], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[4], lines[5], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[5], lines[6], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[6], lines[7], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[2], lines[8], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[8], lines[9], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[9], lines[10], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[10], lines[11], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[0], lines[12], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[12], lines[13], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[13], lines[14], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[14], lines[15], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[0], lines[16], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[16], lines[17], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[17], lines[18], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[18], lines[19], (0, 255, 0), 5)
            cv2.line(ret[0][1], lines[3], lines[20], (0, 255, 0), 5)
            cv2.imshow('BP', ret[0][1])
            key = cv2.waitKey(1)
            lines.clear()
            n += 1
            print(time.time() - s)
        mc.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Realtime.start()
