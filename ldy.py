from dataProcess import DataProcess
from ak.camera_synchronous.core_threading import MulDeviceSynCapture
from dataStructure import string2factor_graph
import numpy as np

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
        self.mc = MulDeviceSynCapture(0, [1, 2])
        self.T0 = np.loadtxt(f'param/0_rgb_ex.txt')
        self.T1 = np.loadtxt(f'param/1_rgb_ex.txt')
        self.T2 = np.loadtxt(f'param/2_rgb_ex.txt')
        self.K0 = np.loadtxt('./param/0_rgb_in.txt')
        self.Rt10 = np.loadtxt('./param/1_0_abt_Rt.txt')
        self.Rt20 = np.loadtxt('./param/2_0_abt_Rt.txt')
        self.mrf = string2factor_graph(
            f'f0({K4ABT_JOINT_NAMES[0]})f1({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[1]})f2({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[18]})f3({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[22]})f4({K4ABT_JOINT_NAMES[18]},{K4ABT_JOINT_NAMES[19]})f5({K4ABT_JOINT_NAMES[20]},{K4ABT_JOINT_NAMES[19]})f6({K4ABT_JOINT_NAMES[21]},{K4ABT_JOINT_NAMES[20]})f7({K4ABT_JOINT_NAMES[22]},{K4ABT_JOINT_NAMES[23]})f8({K4ABT_JOINT_NAMES[23]},{K4ABT_JOINT_NAMES[24]})f9({K4ABT_JOINT_NAMES[24]},{K4ABT_JOINT_NAMES[25]})f10({K4ABT_JOINT_NAMES[1]},{K4ABT_JOINT_NAMES[2]})f11({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[3]})f12({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[4]})f13({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[11]})f14({K4ABT_JOINT_NAMES[3]},{K4ABT_JOINT_NAMES[26]})f15({K4ABT_JOINT_NAMES[4]},{K4ABT_JOINT_NAMES[5]})f16({K4ABT_JOINT_NAMES[5]},{K4ABT_JOINT_NAMES[6]})f17({K4ABT_JOINT_NAMES[6]},{K4ABT_JOINT_NAMES[7]})f21({K4ABT_JOINT_NAMES[11]},{K4ABT_JOINT_NAMES[12]})f22({K4ABT_JOINT_NAMES[12]},{K4ABT_JOINT_NAMES[13]})f23({K4ABT_JOINT_NAMES[13]},{K4ABT_JOINT_NAMES[14]})f27({K4ABT_JOINT_NAMES[1]})f28({K4ABT_JOINT_NAMES[2]})f29({K4ABT_JOINT_NAMES[3]})f30({K4ABT_JOINT_NAMES[4]})f31({K4ABT_JOINT_NAMES[5]})f32({K4ABT_JOINT_NAMES[6]})f33({K4ABT_JOINT_NAMES[7]})f37({K4ABT_JOINT_NAMES[11]})f38({K4ABT_JOINT_NAMES[12]})f39({K4ABT_JOINT_NAMES[13]})f40({K4ABT_JOINT_NAMES[14]})f44({K4ABT_JOINT_NAMES[18]})f45({K4ABT_JOINT_NAMES[19]})f46({K4ABT_JOINT_NAMES[20]})f47({K4ABT_JOINT_NAMES[21]})f48({K4ABT_JOINT_NAMES[22]})f49({K4ABT_JOINT_NAMES[23]})f50({K4ABT_JOINT_NAMES[24]})f51({K4ABT_JOINT_NAMES[25]})f52({K4ABT_JOINT_NAMES[26]})')
        self.j0 = np.zeros([32, 3])
        self.j1 = np.zeros([32, 3])
        self.j2 = np.zeros([32, 3])

    def get(self, turnOn):
        """
        :param turnOn:Ture是获取此帧BP数据，False是关闭AK
        :return: 返回关节点3d坐标估计，索引顺序为
                     "pelvis", "spine - navel", "spine - chest",
                     "neck", "left clavicle", "left shoulder",
                     "left elbow",
                     "left wrist",  "right clavicle",
                     "right shoulder",
                     "right elbow",
                     "right wrist",  "left hip", "left knee",
                     "left ankle",
                     "left foot",
                     "right hip", "right knee", "right ankle", "right foot"
        """
        if turnOn:
            ret = self.mc.get()
            joints = []
            for i in range(32):
                self.j0[i] = np.array(
                    [ret[0][2].joints[i].position.x, ret[0][2].joints[i].position.y, ret[0][2].joints[i].position.z])
                self.j1[i] = np.array(
                    [ret[1][2].joints[i].position.x, ret[1][2].joints[i].position.y, ret[2][2].joints[i].position.z])
                self.j2[i] = np.array(
                    [ret[2][2].joints[i].position.x, ret[2][2].joints[i].position.y, ret[2][2].joints[i].position.z])
                print(ret[0][2].joints[i])
            bpj = DataProcess.singleFrame(self.T0, self.T1, self.T2, self.mrf, self.j0, self.j1, self.j2,
                                          [1, 2, 3, 4, 5, 6, 7, 8], self.Rt10, self.Rt20)
            joints.append(bpj)
            return joints
        else:
            self.mc.close()
            return


if __name__ == '__main__':
    rl = Realtime()
    for i in range(20):
        rl.get(True)
    rl.get(False)
