import numpy as np

def calculate_iou_matrix(box_pred, box_ground, threshold=0.5):
    # left top x, left top y, right bottom x, right bottom y
    x11, y11, x12, y12 = np.split(box_pred, 4, axis=1)
    x21, y21, x22, y22 = np.split(box_ground, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB-xA+1e-9),0)*np.maximum((yB-yA+1e-9),0)
    boxAArea = (x12-x11+1e-9)*(y12-y11+1e-9)
    boxBArea = (x22-x21+1e-9)*(y22-y21+1e-9)
    iou = interArea / (boxAArea + np.transpose(boxBArea)-interArea)
    return iou

def merge_iou(box_pred, threshold=0.3):
    iou = calculate_iou_matrix(box_pred, box_pred)
    visit = [0]*len(box_pred)
    cnt = 0
    res = []
    for i in range(len(box_pred)):
        if visit[i] == 0:
            visit[i] = 1
            cur_pt = i
            next_pt = i+1
            tmp = []
            tmp.append(box_pred[cur_pt])
            while cur_pt < len(box_pred) and next_pt < len(box_pred):
                if visit[next_pt] == 0 and iou[cur_pt, next_pt] > threshold:
                    cnt += 1
                    visit[next_pt] = 1
                    tmp.append(box_pred[next_pt])
                    cur_pt = next_pt
                next_pt += 1
            res.append(np.mean(tmp, axis=0))
    return np.array(res), cnt

def calculate_acc(box_pred, box_ground, threshold=0.4):
    box_pred_new, cnt = merge_iou(box_pred)
    while cnt > 0:
        box_pred_new, cnt = merge_iou(box_pred_new)
    iou = calculate_iou_matrix(box_pred_new, box_ground)
    tp_pred = box_ground[np.max(iou, axis=0)>=threshold]
    fp_pred = box_pred_new[np.max(iou, axis=1)<threshold]
    fn_ground = box_ground[np.max(iou, axis=0)<threshold]
    return tp_pred, fp_pred, fn_ground






