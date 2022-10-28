import logging
import sys
import time
import numpy as np
from dataProcess import K4ABT_JOINT_NAMES,cfg

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            level=int(cfg['Logging_level']), filename='dev.log', filemode='a')

class RealtimeVariAK(object):
    def __init__(self):
        pass

    @classmethod
    def openAK(cls):
        from ak.camera_synchronous.core_threading import MulDeviceSynCapture
        import cv2

        # all params,setups and verifies
        ak_numbers = 0
        for i in range(10):
            stream = cv2.VideoCapture(i)
            grab = stream.grab()
            stream.release()
            if not grab:
                break
            ak_numbers += 1
        if ak_numbers:
            logging.info(f'detect {ak_numbers} ak. program is going to continue.')
        else:
            logging.info(f'detect {ak_numbers} ak. program is breaking out')
            return
        sub_ak_lst = [ak for ak in range(1, ak_numbers)]
        try:
            mc = MulDeviceSynCapture(0, sub_ak_lst)
        except Exception:
            mc.close()
        return mc, ak_numbers

    @classmethod
    def esitBoneLength(cls, participant_name):
        '''
        Args:
            participant_name: a string
        Returns:
        '''
        import os
        from dataProcess import utils
        import json
        mc, ak_numbers = RealtimeVariAK.openAK()
        if os.path.exists(f'./data/{participant_name}'):
            pass
        else:
            os.mkdir(f'./data/{participant_name}')
            os.mkdir(f'./data/{participant_name}/bone_length')
            assert os.path.exists(f'./data/{participant_name}/bone_length'), 'make directory failed!'
        # The participants should face ak and stand up straight and flat stretch their arms ,then twist their body in a small a angle. Then do once again with back to ak.
        J = locals()
        bl_t = np.zeros((ak_numbers, 100, 20, 1))  # dim1 equals to the following range value
        for ak in range(ak_numbers):
            J[f'joints{ak}'] = np.zeros((0, 3))
        for time in range(100):  # just face ak for an example
            ret = mc.get()
            for ak in range(ak_numbers):
                temp = np.zeros((0, 3))
                for i in range(32):
                    temp = np.concatenate((temp,np.array([ret[ak][3].joints[i].position.x, ret[ak][3].joints[i].position.y,
                                          ret[ak][3].joints[i].position.z]).reshape((1,-1))),axis=0)
                J[f'joints{ak}'] = np.concatenate((J[f'joints{ak}'], temp))
        mc.close()
        for k in range(ak_numbers):
            bl_t[k] = utils.bonesLengthMid(J[f'joints{ak}'])  # (100, 20, 1)
        bld = utils.dictMidValueWithMultiAK(np.concatenate(bl_t, axis=1))
        json.dump(bld, open(f'./data/{participant_name}/bone_length/mid_value.json','a'))
    @classmethod
    def viewino3d(cls,res):
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(res)
        o3d.visualization.draw(pc)
    @classmethod
    def start(cls, participant_name, total):
        '''
        Args:
            participant_name: must same as  functing esitBoneLength()
            total: how many frames to obtain
        Returns:
        '''
        from dataProcess import DataProcess, utils
        from dataStructure import string2factor_graph
        import numpy as np
        # from udpProcess import udpPose
        import json
        import cv2
        # start AK
        mc, ak_numbers = RealtimeVariAK.openAK()

        # all params and setups
        K = np.zeros((ak_numbers, 3, 3))
        T = np.zeros((ak_numbers, 3, 4))
        Rt = np.zeros((ak_numbers, 4, 4))
        Cxy = np.zeros((ak_numbers, 1, 3))
        J = np.zeros((ak_numbers, 21, 3))
        Rt[0] = np.eye(4)
        Cxy[0] = np.zeros((1, 3))
        for p in range(ak_numbers):
            preK = np.loadtxt(f'param/{p}_rgb_in.txt')
            T[p] = np.loadtxt(f'param/{p}_rgb_ex.txt')
            if p != 0:
                Rt[p] = np.loadtxt(f'param/{p}_0_abt_Rt.txt')  # no major ak
                Cxy[p] = preK[:, 2].reshape((1, 3))
            K[p] = np.linalg.inv(preK)
        mrf = string2factor_graph(
            f'f0({K4ABT_JOINT_NAMES[0]})f1({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[1]})f2({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[18]})f3({K4ABT_JOINT_NAMES[0]},{K4ABT_JOINT_NAMES[22]})f4({K4ABT_JOINT_NAMES[18]},{K4ABT_JOINT_NAMES[19]})f5({K4ABT_JOINT_NAMES[20]},{K4ABT_JOINT_NAMES[19]})f6({K4ABT_JOINT_NAMES[21]},{K4ABT_JOINT_NAMES[20]})f7({K4ABT_JOINT_NAMES[22]},{K4ABT_JOINT_NAMES[23]})f8({K4ABT_JOINT_NAMES[23]},{K4ABT_JOINT_NAMES[24]})f9({K4ABT_JOINT_NAMES[24]},{K4ABT_JOINT_NAMES[25]})f10({K4ABT_JOINT_NAMES[1]},{K4ABT_JOINT_NAMES[2]})f11({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[3]})f12({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[4]})f13({K4ABT_JOINT_NAMES[2]},{K4ABT_JOINT_NAMES[11]})f14({K4ABT_JOINT_NAMES[3]},{K4ABT_JOINT_NAMES[26]})f15({K4ABT_JOINT_NAMES[4]},{K4ABT_JOINT_NAMES[5]})f16({K4ABT_JOINT_NAMES[5]},{K4ABT_JOINT_NAMES[6]})f17({K4ABT_JOINT_NAMES[6]},{K4ABT_JOINT_NAMES[7]})f21({K4ABT_JOINT_NAMES[11]},{K4ABT_JOINT_NAMES[12]})f22({K4ABT_JOINT_NAMES[12]},{K4ABT_JOINT_NAMES[13]})f23({K4ABT_JOINT_NAMES[13]},{K4ABT_JOINT_NAMES[14]})f27({K4ABT_JOINT_NAMES[1]})f28({K4ABT_JOINT_NAMES[2]})f29({K4ABT_JOINT_NAMES[3]})f30({K4ABT_JOINT_NAMES[4]})f31({K4ABT_JOINT_NAMES[5]})f32({K4ABT_JOINT_NAMES[6]})f33({K4ABT_JOINT_NAMES[7]})f37({K4ABT_JOINT_NAMES[11]})f38({K4ABT_JOINT_NAMES[12]})f39({K4ABT_JOINT_NAMES[13]})f40({K4ABT_JOINT_NAMES[14]})f44({K4ABT_JOINT_NAMES[18]})f45({K4ABT_JOINT_NAMES[19]})f46({K4ABT_JOINT_NAMES[20]})f47({K4ABT_JOINT_NAMES[21]})f48({K4ABT_JOINT_NAMES[22]})f49({K4ABT_JOINT_NAMES[23]})f50({K4ABT_JOINT_NAMES[24]})f51({K4ABT_JOINT_NAMES[25]})f52({K4ABT_JOINT_NAMES[26]})')
        bld = json.load(open(f'./data/{participant_name}/bone_length/mid_value.json', 'r'))
        first_frame = True
        initial = np.zeros((0,3))
        space2plane_K = np.linalg.inv(K[0])

        # frame difference
        affine_table = np.tile(np.arange(1, 12 - ak_numbers), (ak_numbers, 1))
        rs_lst = []
        tem_lst = []
        utils.recurisive(ak_numbers - 1, affine_table, 0, rs_lst, tem_lst)
        affine_np = np.array(rs_lst)

        affine_table_n = np.tile(np.arange(1, 11 - ak_numbers), (ak_numbers+1, 1))
        rs_lst_n = []
        tem_lst_n = []
        utils.recurisive(ak_numbers, affine_table_n, 0, rs_lst_n, tem_lst_n)
        affine_np_n = np.array(rs_lst_n)

        # another thread for UDP
        img = np.zeros((ak_numbers, 720, 1280, 3))

        # gobal variable that need to be remolded in pyqt5
        key = 0  # key to stop process
        detect = False  # whether all ak detect participant. if False, just show origin image; if True, show BP estimation
        BP_esti = np.zeros((21, 2))  # BP estimation on image plane
        frame_num =0
        save_lst = []
        rgb_save_lst = []
        try:
            while frame_num < total:
                s = time.time()
                ret = mc.get()
                print('start')
                for ak in range(ak_numbers):
                    if ret[ak][2] == 0:
                        detect = False
                    img[ak] = cv2.cvtColor(ret[ak][1], cv2.COLOR_RGBA2RGB)
                    rgb_save_lst.append(img[ak].copy())

                if not detect:
                    detect = True
                    continue

                # UDP = udpPose(img) # should be asynchronous
                for akn in range(ak_numbers):
                    for idx,i in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26]):
                        J[akn][idx] = np.array([ret[akn][3].joints[i].position.x, ret[akn][3].joints[i].position.y,
                                              ret[akn][3].joints[i].position.z])

                if first_frame:
                    ret = DataProcess.singleFrameWithVariAK(np.eye(1), K, T, mrf, J, Rt, Cxy, affine_np, bld, first_frame, initial)
                else:
                    ret = DataProcess.singleFrameWithVariAK(np.eye(1), K, T, mrf, J, Rt, Cxy, affine_np_n, bld, first_frame, last_bp)
                last_bp = ret.copy()
                if first_frame//3 ==0:
                    save_lst.append(ret)
                BP_esti = (space2plane_K @ last_bp.T / last_bp.T[2,:])[:2,:]
                logging.info(f'2 ak process time consumes {time.time() - s}')
                frame_num +=1
                print(frame_num)
            save_txt = np.array(save_lst).reshape((-1,3))
            np.savetxt(f'./data/{participant_name}/{time.time()}.txt',save_txt)
            for imdex,im in enumerate(rgb_save_lst):
                if imdex%2==0:
                    cv2.imwrite(f'./data/{participant_name}/img0/{imdex/2}.jpg',im) # img0 diretory is created manully!!!
                else:
                    cv2.imwrite(f'./data/{participant_name}/img1/{(imdex-1)/2}.jpg',im)
                print(imdex)
            return
        except KeyboardInterrupt:
            mc.close()

if __name__ == '__main__':
    # RealtimeVariAK.esitBoneLength('FW_1')
    RealtimeVariAK.start('FW_1')