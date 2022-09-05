import cv2
from dataProcess import DataProcess, utils
import os
import json
import numpy as np
from tqdm.auto import tqdm


class Evaluation(object):
    @classmethod
    def dataload(cls):
        path_DATA = 'E:/attitude/twice/fm/image0/DATA/image26.json'
        path_img = 'E:/attitude/twice/fm/image0/image26.bmp'
        mark = json.load(open(path_DATA, encoding='utf-8'))
        img = cv2.imread(path_img)
        j_len = len(mark['dots'])
        for j in range(j_len):
            cv2.circle(img, (int(mark['dots'][j]['Point']['X']), int(mark['dots'][j]['Point']['Y'])), 2, [0, 255, 0])
        cv2.imshow('26', img)
        cv2.waitKey(0)

    @classmethod
    def evalBP(cls):
        '''
        加载标注数据，两遍的进行均值。
        根据加载数据运行算法，再投影，再算欧式距离就是单个样本的误差
        :return: 总误差/样本数量
        '''
        frame_count = 100
        sample_json_l = []
        sample_data_l = []
        dir = 'E:/attitude/twice/'
        name_list = os.listdir(dir)
        bpStep1 = np.zeros([21, 3])
        K0 = np.array([[6.020215454101562500e+02, 0.000000000000000000e+00, 6.371241455078125000e+02],
                       [0.000000000000000000e+00, 6.018470458984375000e+02, 3.665009155273437500e+02],
                       [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
        K1 = np.array([[6.052656860351562500e+02, 0.000000000000000000e+00, 6.425632934570312500e+02],
                       [0.000000000000000000e+00, 6.050792236328125000e+02, 3.652657165527343750e+02],
                       [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
        K2 = np.array([[6.072551269531250000e+02, 0.000000000000000000e+00, 6.365347900390625000e+02],
                       [0.000000000000000000e+00, 6.072070312500000000e+02, 3.670882873535156250e+02],
                       [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
        Rt10 = np.loadtxt('./param/1_0_abt_Rt.txt')
        Rt20 = np.loadtxt('./param/2_0_abt_Rt.txt')
        Rt10inv = np.linalg.inv(Rt10)
        Rt20inv = np.linalg.inv(Rt20)
        slice_window0 = []
        slice_window1 = []
        slice_window2 = []
        drop_frame = False
        for n in tqdm(name_list, desc=f'name processor:', leave=True, colour='green', ncols=100, position=0):
            # if n != 'fm':
            #     continue
            for f in tqdm(range(20, frame_count), desc=f'frame processor:', leave=True, colour='red', ncols=100,
                          position=1):
                for ak in range(3):
                    img = cv2.imread(f'{dir}{n}/image{ak}/image{f}.bmp')
                    try:
                        json_file = json.load(open(f'{dir}{n}/image{ak}/DATA/image{f}.json', encoding='utf-8'))
                    except FileNotFoundError:
                        drop_frame = True
                        # print(f'no file: {n},image{ak}/{f}.bmp')
                        break
                    if json_file['dots'] is None or len(json_file['dots']) != 30:
                        drop_frame = True
                        # print(f'label not enough:{n},image{ak}/{f}.bmp',)
                        break
                    sample_json_l.append(json_file)
                    sample_data_l.append(img)

                if drop_frame:
                    sample_json_l = sample_json_l[:len(sample_json_l) - (len(sample_json_l) % 3)]
                    drop_frame = False
                    continue
                bpStep1 = DataProcess.BPapplication(n, f, f + 1, bpStep1)  # 人，帧的bp估计,主相机rgb空间
                bp_ak0_normal = bpStep1.T / bpStep1.T[2]
                uv1_ak0_panel = np.dot(K0, bp_ak0_normal)

                bpStep1_ak1 = np.dot(Rt10inv[:3, :3], bpStep1.T) + Rt10inv[:3, 3].reshape(3, 1)
                bpStep1_ak1_normal = bpStep1_ak1 / bpStep1_ak1[2]
                uv1_ak1_panel = np.dot(K1, bpStep1_ak1_normal)

                bpStep1_ak2 = np.dot(Rt20inv[:3, :3], bpStep1.T) + Rt20inv[:3, 3].reshape(3, 1)
                bpStep1_ak2_normal = bpStep1_ak2 / bpStep1_ak2[2]
                uv1_ak2_panel = np.dot(K2, bpStep1_ak2_normal)
                # 一帧三张loss加起来才是一帧的loss,21里挑15个
                uv1_ak0_panel = np.delete(uv1_ak0_panel, [3, 4, 8, 15, 19, 20], axis=1)
                uv1_ak1_panel = np.delete(uv1_ak1_panel, [3, 4, 8, 15, 19, 20], axis=1)
                uv1_ak2_panel = np.delete(uv1_ak2_panel, [3, 4, 8, 15, 19, 20], axis=1)

                # for uv in range(uv1_ak2_panel.shape[1]): # bp 投下来的
                #     u = int(round(uv1_ak0_panel[0, uv], 0))
                #     v = int(round(uv1_ak0_panel[1, uv], 0))
                #     cv2.circle(sample_data_l[-3], (u, v), 2, (0, 255, 0))
                #     u = int(round(uv1_ak1_panel[0, uv], 0))
                #     v = int(round(uv1_ak1_panel[1, uv], 0))
                #     cv2.circle(sample_data_l[-2], (u, v), 2, (0, 255, 0))
                #     u = int(round(uv1_ak2_panel[0, uv], 0))
                #     v = int(round(uv1_ak2_panel[1, uv], 0))
                #     cv2.circle(sample_data_l[-1], (u, v), 2, (0, 255, 0))
                # cv2.imshow(f'ak0', sample_data_l[-3])
                # cv2.imshow(f'ak1', sample_data_l[-2])
                # cv2.imshow(f'ak2', sample_data_l[-1])
                # cv2.waitKey(0)

                variables = locals()
                for v in range(3):
                    variables['slice_window%s' % v].append(variables['uv1_ak%s_panel' % v])
        if len(sample_json_l) == 0:
            return

        filtered_uv = []
        for i in range(len(slice_window0)):
            filtered_uv.append(utils.BPmidFilter(
                [slice_window0[max(i - 1, 0)], slice_window0[i], slice_window0[min(len(slice_window0) - 1, i + 1)]])[:2,
                               :])

            filtered_uv.append(utils.BPmidFilter(
                [slice_window1[max(i - 1, 0)], slice_window1[i], slice_window1[min(len(slice_window1) - 1, i + 1)]])[:2,
                               :])

            filtered_uv.append(utils.BPmidFilter(
                [slice_window2[max(i - 1, 0)], slice_window2[i], slice_window2[min(len(slice_window2) - 1, i + 1)]])[:2,
                               :])
        gt = []
        for l in range(len(sample_json_l)):
            gtxy0 = np.zeros([2, 15])
            gtxy1 = np.zeros([2, 15])

            for idx in range(15):
                if sample_json_l[l]['dots'][idx]['Label'].startswith('00'):
                    gtxy0[0, 0] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 0] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('01'):
                    gtxy0[0, 1] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 1] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('02'):
                    gtxy0[0, 2] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 2] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('05'):
                    gtxy0[0, 3] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 3] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('06'):
                    gtxy0[0, 4] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 4] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('07'):
                    gtxy0[0, 5] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 5] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('12'):
                    gtxy0[0, 6] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 6] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('13'):
                    gtxy0[0, 7] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 7] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('14'):
                    gtxy0[0, 8] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 8] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('18'):
                    gtxy0[0, 9] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 9] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('19'):
                    gtxy0[0, 10] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 10] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('20'):
                    gtxy0[0, 11] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 11] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('22'):
                    gtxy0[0, 12] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 12] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('23'):
                    gtxy0[0, 13] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 13] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('24'):
                    gtxy0[0, 14] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 14] = sample_json_l[l]['dots'][idx]['Point']['Y']
            for idx in range(15, 30):
                if sample_json_l[l]['dots'][idx]['Label'].startswith('00'):
                    gtxy1[0, 0] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 0] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('01'):
                    gtxy1[0, 1] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 1] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('02'):
                    gtxy1[0, 2] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 2] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('05'):
                    gtxy1[0, 3] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 3] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('06'):
                    gtxy1[0, 4] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 4] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('07'):
                    gtxy1[0, 5] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 5] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('12'):
                    gtxy1[0, 6] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 6] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('13'):
                    gtxy1[0, 7] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 7] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('14'):
                    gtxy1[0, 8] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 8] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('18'):
                    gtxy1[0, 9] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 9] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('19'):
                    gtxy1[0, 10] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 10] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('20'):
                    gtxy1[0, 11] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 11] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('22'):
                    gtxy1[0, 12] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 12] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('23'):
                    gtxy1[0, 13] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 13] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('24'):
                    gtxy1[0, 14] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 14] = sample_json_l[l]['dots'][idx]['Point']['Y']
            gt.append((gtxy0 + gtxy1) / 2)
            # gtxy2 = np.zeros([2, 15])
            # for idx in range(30, 45):
            #     if sample_json_l[l]['dots'][idx]['Label'].startswith('00'):
            #         gtxy2[0, 0] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 0] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('01'):
            #         gtxy2[0, 1] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 1] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('02'):
            #         gtxy2[0, 2] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 2] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('05'):
            #         gtxy2[0, 3] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 3] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('06'):
            #         gtxy2[0, 4] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 4] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('07'):
            #         gtxy2[0, 5] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 5] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('12'):
            #         gtxy2[0, 6] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 6] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('13'):
            #         gtxy2[0, 7] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 7] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('14'):
            #         gtxy2[0, 8] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 8] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('18'):
            #         gtxy2[0, 9] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 9] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('19'):
            #         gtxy2[0, 10] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 10] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('20'):
            #         gtxy2[0, 11] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 11] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('22'):
            #         gtxy2[0, 12] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 12] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('23'):
            #         gtxy2[0, 13] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 13] = sample_json_l[l]['dots'][idx]['Point']['Y']
            #     elif sample_json_l[l]['dots'][idx]['Label'].startswith('24'):
            #         gtxy2[0, 14] = sample_json_l[l]['dots'][idx]['Point']['X']
            #         gtxy2[1, 14] = sample_json_l[l]['dots'][idx]['Point']['Y']
            # gt.append((gtxy0+gtxy1+gtxy2)/3)

        output = np.array(filtered_uv)
        label = np.array(gt)

        score = np.sqrt(np.sum((output - label) ** 2, axis=1))
        print(np.sum(score) / score.shape[0] / score.shape[1])

    @classmethod
    def evalAK(cls):
        frame_count = 100
        sample_json_l = []
        dir = 'E:/attitude/twice/'
        name_list = os.listdir(dir)
        drop_frame = False
        ak_uv = []
        Rt10 = np.loadtxt('./param/1_0_abt_Rt.txt')
        Rt20 = np.loadtxt('./param/2_0_abt_Rt.txt')
        Rt10inv = np.linalg.inv(Rt10)
        Rt20inv = np.linalg.inv(Rt20)
        for n in tqdm(name_list, desc=f'name processor:', leave=True, colour='green', ncols=100, position=0):
            # if n != 'zjh':
            #     continue
            for f in tqdm(range(20, frame_count), desc=f'frame processor:', leave=True, colour='red', ncols=100,
                          position=1):
                for ak in range(3):
                    try:
                        json_file = json.load(open(f'{dir}{n}/image{ak}/DATA/image{f}.json', encoding='utf-8'))
                    except FileNotFoundError:
                        drop_frame = True
                        break
                    if json_file['dots'] is None or len(json_file['dots']) != 30:
                        drop_frame = True
                        break
                    sample_json_l.append(json_file)

                if drop_frame:
                    sample_json_l = sample_json_l[:len(sample_json_l) - (len(sample_json_l) % 3)]
                    drop_frame = False
                    continue
                ak_uv.append(DataProcess.calculateAKDistance2Label(0, f, n, Rt10, Rt20, Rt10inv, Rt20inv)[:2, :])
                ak_uv.append(DataProcess.calculateAKDistance2Label(1, f, n, Rt10, Rt20, Rt10inv, Rt20inv)[:2, :])
                ak_uv.append(DataProcess.calculateAKDistance2Label(2, f, n, Rt10, Rt20, Rt10inv, Rt20inv)[:2, :])
        gt = []
        for l in range(len(sample_json_l)):
            gtxy0 = np.zeros([2, 15])
            gtxy1 = np.zeros([2, 15])
            for idx in range(15):
                if sample_json_l[l]['dots'][idx]['Label'].startswith('00'):
                    gtxy0[0, 0] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 0] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('01'):
                    gtxy0[0, 1] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 1] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('02'):
                    gtxy0[0, 2] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 2] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('05'):
                    gtxy0[0, 3] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 3] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('06'):
                    gtxy0[0, 4] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 4] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('07'):
                    gtxy0[0, 5] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 5] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('12'):
                    gtxy0[0, 6] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 6] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('13'):
                    gtxy0[0, 7] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 7] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('14'):
                    gtxy0[0, 8] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 8] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('18'):
                    gtxy0[0, 9] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 9] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('19'):
                    gtxy0[0, 10] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 10] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('20'):
                    gtxy0[0, 11] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 11] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('22'):
                    gtxy0[0, 12] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 12] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('23'):
                    gtxy0[0, 13] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 13] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('24'):
                    gtxy0[0, 14] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy0[1, 14] = sample_json_l[l]['dots'][idx]['Point']['Y']
            for idx in range(15, 30):
                if sample_json_l[l]['dots'][idx]['Label'].startswith('00'):
                    gtxy1[0, 0] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 0] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('01'):
                    gtxy1[0, 1] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 1] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('02'):
                    gtxy1[0, 2] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 2] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('05'):
                    gtxy1[0, 3] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 3] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('06'):
                    gtxy1[0, 4] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 4] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('07'):
                    gtxy1[0, 5] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 5] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('12'):
                    gtxy1[0, 6] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 6] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('13'):
                    gtxy1[0, 7] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 7] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('14'):
                    gtxy1[0, 8] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 8] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('18'):
                    gtxy1[0, 9] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 9] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('19'):
                    gtxy1[0, 10] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 10] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('20'):
                    gtxy1[0, 11] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 11] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('22'):
                    gtxy1[0, 12] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 12] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('23'):
                    gtxy1[0, 13] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 13] = sample_json_l[l]['dots'][idx]['Point']['Y']
                elif sample_json_l[l]['dots'][idx]['Label'].startswith('24'):
                    gtxy1[0, 14] = sample_json_l[l]['dots'][idx]['Point']['X']
                    gtxy1[1, 14] = sample_json_l[l]['dots'][idx]['Point']['Y']
            gt.append((gtxy0 + gtxy1) / 2)
        output = np.array(ak_uv)
        label = np.array(gt)
        score = np.sqrt(np.sum((output - label) ** 2, axis=1))
        print(np.sum(score) / score.shape[0] / score.shape[1])


def checkLabel():
    for n in range(3):
        for i in range(20, 100):
            json_fs = json.load(open(f'E:/attitude/twice/ls/image{n}/DATA/image{i}.json', encoding='utf-8'))
            img = cv2.imread(f'E:/attitude/twice/ls/image{n}/image{i}.bmp')
            for i in range(len(json_fs['dots'])):
                cv2.circle(img, (int(json_fs['dots'][i]['Point']['X']), int(json_fs['dots'][i]['Point']['Y'])), 2,
                           [0, 255, 0])
            cv2.imshow('img', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    # 加载数据看一下标注的效果
    # Evaluation.dataload()
    # 与标注数据进行对比
    # Evaluation.evalBP()
    # Evaluation.evalAK()
    checkLabel()
