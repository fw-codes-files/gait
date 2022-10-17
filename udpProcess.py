import torch
import numpy as np
import cv2
from numpy.linalg import LinAlgError

def udpPose(self, img_np: np):
    rs_img = img_np.copy()
    results = []
    # 输入是tensor
    # 1.放gpu上
    img = torch.tensor(img_np.copy()).to(self.config['device'])  # (3,720,1280,3)
    # 2.数据标准化 先bbox，再标准化
    # 3.转成dataset数据结构
    # 3.1. 先初始化bbox模型并得到bbox
    img = torch.permute(img, (0, 3, 1, 2)) / 255  # (3,3,720,1280)

    with torch.no_grad():
        pred = self.box_model(img)
    pred_bbox = torch.zeros([img_np.shape[0], 4]).cuda()  # (bs,4)
    for i in range(img_np.shape[0]):
        if pred[i]['boxes'].shape == 0:
            return self.empty_bbox
        pred_bbox[i] = pred[i]['boxes'][0].detach()
    # 求变换矩阵
    center = torch.zeros([img_np.shape[0], 2])
    center[:, 0] = (pred_bbox[:, 2] - pred_bbox[:, 0]) / 2 + pred_bbox[:, 0]
    center[:, 1] = (pred_bbox[:, 3] - pred_bbox[:, 1]) / 2 + pred_bbox[:, 1]
    scale = torch.zeros([img_np.shape[0], 2])
    scale[:, 0] = (pred_bbox[:, 2] - pred_bbox[:, 0]) / 200 * self.config['test_x_extention']
    scale[:, 1] = (pred_bbox[:, 3] - pred_bbox[:, 1]) / 200 * self.config['test_y_extention']
    img_np_affined = np.zeros([img_np.shape[0], self.config['input_shape'][0], self.config['input_shape'][1], 3])
    for s in range(scale.shape[0]):
        if scale[s, 0] > scale[s, 1] * self.config['w_h_ratio']:
            scale[s, 1] = scale[s, 0] * 1.0 / self.config['w_h_ratio']
        else:
            scale[s, 0] = scale[s, 1] * 1.0 * self.config['w_h_ratio']
        trans = torch.zeros([2, 3])
        trans[0, 0] = scale[s, 0] * 200.0 / (self.config['input_shape'][1] - 1)
        trans[0, 2] = -0.5 * scale[s, 0] * 200 + 0.5 * center[s, 0] * 2
        trans[1, 0] = -trans[1, 0]
        trans[1, 1] = scale[s, 1] * 200.0 / (self.config['input_shape'][0] - 1)
        trans[1, 2] = -0.5 * scale[s, 1] * 200 + 0.5 * center[s, 1] * 2
        M = np.array([[trans[0, 0].item(), trans[0, 1].item(), trans[0, 2].item()],
                      [trans[1, 0].item(), trans[1, 1].item(), trans[1, 2].item()]])
        img_np_affined_s = cv2.warpAffine(img_np[s], M,
                                          (self.config['input_shape'][1], self.config['input_shape'][0]),
                                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        img_np_affined[s] = img_np_affined_s
    img_np_affined = torch.permute((torch.tensor(img_np_affined).cuda() / 255 - self.means) / self.stds,
                                   (0, 3, 1, 2)).type(torch.float32)
    img_np_affined_flip = torch.flip(img_np_affined, [3])
    # 正反两个图像过网络，改成batchsize=2*bs
    img_np_affined = torch.vstack((img_np_affined, img_np_affined_flip))  # (bs*2,3,256,192)
    with torch.no_grad():
        outputs = self.model(img_np_affined)
        half_batch_idx = int(outputs.shape[0] / 2)
        outputs[half_batch_idx:] = torch.flip(outputs[half_batch_idx:], [3])
        for p in self.config['flip_pairs']:
            tmp = outputs[half_batch_idx:, p[0], :, :].clone().detach()
            outputs[half_batch_idx:, p[0], :, :] = outputs[half_batch_idx:, p[1], :, :]
            outputs[half_batch_idx:, p[1], :, :] = tmp
        outputs_mean = (outputs[:half_batch_idx] + outputs[half_batch_idx:]) * 0.5 / 255

    outputs = outputs_mean.detach().cpu().numpy()
    # 特征处理
    heatmaps_reshaped = outputs.reshape((outputs.shape[0], outputs.shape[1], -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((outputs.shape[0], outputs.shape[1], 1))
    idx = idx.reshape((outputs.shape[0], outputs.shape[1], 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % outputs.shape[3]
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / outputs.shape[3])
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    # 后处理
    shape_pad = list(outputs.shape)
    shape_pad[2] = shape_pad[2] + 2
    shape_pad[3] = shape_pad[3] + 2
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            mapij = outputs[i, j, :, :]
            maxori = np.max(mapij)
            mapij = cv2.GaussianBlur(mapij, (7, 7), 0)
            max = np.max(mapij)
            min = np.min(mapij)
            mapij = (mapij - min) / (max - min) * maxori
            outputs[i, j, :, :] = mapij
    batch_heatmaps = np.clip(outputs, 0.001, 50)
    batch_heatmaps = np.log(batch_heatmaps)
    batch_heatmaps_pad = np.zeros(shape_pad, dtype=float)
    batch_heatmaps_pad[:, :, 1:-1, 1:-1] = batch_heatmaps
    batch_heatmaps_pad[:, :, 1:-1, -1] = batch_heatmaps[:, :, :, -1]
    batch_heatmaps_pad[:, :, -1, 1:-1] = batch_heatmaps[:, :, -1, :]
    batch_heatmaps_pad[:, :, 1:-1, 0] = batch_heatmaps[:, :, :, 0]
    batch_heatmaps_pad[:, :, 0, 1:-1] = batch_heatmaps[:, :, 0, :]
    batch_heatmaps_pad[:, :, -1, -1] = batch_heatmaps[:, :, -1, -1]
    batch_heatmaps_pad[:, :, 0, 0] = batch_heatmaps[:, :, 0, 0]
    batch_heatmaps_pad[:, :, 0, -1] = batch_heatmaps[:, :, 0, -1]
    batch_heatmaps_pad[:, :, -1, 0] = batch_heatmaps[:, :, -1, 0]
    I = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1 = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1 = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1y1 = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1_y1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1_ = np.zeros((shape_pad[0], shape_pad[1]))
    preds = preds.astype(np.int32)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            I[i, j] = batch_heatmaps_pad[i, j, preds[i, j, 1] + 1, preds[i, j, 0] + 1]
            Ix1[i, j] = batch_heatmaps_pad[i, j, preds[i, j, 1] + 1, preds[i, j, 0] + 2]
            Ix1_[i, j] = batch_heatmaps_pad[i, j, preds[i, j, 1] + 1, preds[i, j, 0]]
            Iy1[i, j] = batch_heatmaps_pad[i, j, preds[i, j, 1] + 2, preds[i, j, 0] + 1]
            Iy1_[i, j] = batch_heatmaps_pad[i, j, preds[i, j, 1], preds[i, j, 0] + 1]
            Ix1y1[i, j] = batch_heatmaps_pad[i, j, preds[i, j, 1] + 2, preds[i, j, 0] + 2]
            Ix1_y1_[i, j] = batch_heatmaps_pad[i, j, preds[i, j, 1], preds[i, j, 0]]
    dx = 0.5 * (Ix1 - Ix1_)
    dy = 0.5 * (Iy1 - Iy1_)
    D = np.zeros((shape_pad[0], shape_pad[1], 2))
    D[:, :, 0] = dx
    D[:, :, 1] = dy
    D.reshape((shape_pad[0], shape_pad[1], 2, 1))
    dxx = Ix1 - 2 * I + Ix1_
    dyy = Iy1 - 2 * I + Iy1_
    dxy = 0.5 * (Ix1y1 - Ix1 - Iy1 + I + I - Ix1_ - Iy1_ + Ix1_y1_)
    hessian = np.zeros((shape_pad[0], shape_pad[1], 2, 2))
    hessian[:, :, 0, 0] = dxx
    hessian[:, :, 1, 0] = dxy
    hessian[:, :, 0, 1] = dxy
    hessian[:, :, 1, 1] = dyy
    inv_hessian = np.zeros(hessian.shape)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            hessian_tmp = hessian[i, j, :, :]
            try:
                inv_hessian[i, j, :, :] = np.linalg.inv(hessian_tmp)
            except LinAlgError:
                inv_hessian[i, j, :, :] = np.zeros((2, 2))
    res = np.zeros(preds.shape)  # (bs,17,2)
    preds = preds.astype(np.float64)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            D_tmp = D[i, j, :]
            D_tmp = D_tmp[:, np.newaxis]
            shift = np.matmul(inv_hessian[i, j, :, :], D_tmp)
            res_tmp = preds[i, j, :] - shift.reshape((-1))
            res[i, j, :] = res_tmp
    # 拿到特征点坐标
    new_preds = np.zeros(res.shape)
    new_preds[:, :, 0] = res[:, :, 0] * scale[:, 0].detach().cpu().numpy().reshape(-1, 1) * 200 / 47 + center[:,
                                                                                                       0].detach().cpu().numpy().reshape(
        -1, 1) - scale[:,
                 0].detach().cpu().numpy().reshape(-1, 1) * 0.5 * 200
    new_preds[:, :, 1] = res[:, :, 1] * scale[:, 1].detach().cpu().numpy().reshape(-1, 1) * 200 / 63 + center[:,
                                                                                                       1].detach().cpu().numpy().reshape(
        -1, 1) - scale[:,
                 1].detach().cpu().numpy().reshape(-1, 1) * 0.5 * 200
    # 算分
    new_preds = np.concatenate((new_preds, maxvals), axis=2)
    for i in range(preds.shape[0]):
        keypoints = new_preds[i].reshape(-1).tolist()
        results.append(dict(keypoints=keypoints))
    joints = []
    for r in results:
        kps = np.array(r['keypoints']).reshape(-1, 3)
        i = 0
        while i < kps.shape[0]:
            if kps[i][2] <= 0:
                kps[i][0] = 0
                kps[i][1] = 0
            i += 1
        joints_single = np.delete(kps, 2, 1, )
        joints_single = joints_single.astype(int)
        joints.append(joints_single)
    # todo 17 joints minus 5 Facial joints
    return np.array(joints)