import scipy.io as sp
import torch
import os
import numpy as np
import cv2

def imageFolder(filepath, chn_first=False):# basic
    pics = []
    labels = []
    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    
    for k,item in class_to_idx.items():
        print('class:{} lable:{}'.format(k,item))

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(filepath, target)
        fs = os.listdir(d)
        for fname in fs:
            path = os.path.join(d, fname)
            img = cv2.imread(path)
            if chn_first:
                img = np.transpose(img, (2,0,1))
            pics.append(img)
            labels.append(class_to_idx[target])
    return np.array(pics, dtype='float')/255.0, np.array(labels)

def detectionFolder(filepath, chn_first=False):
    coords = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    coords.sort()
    coord_to_xys = {coord:np.loadtxt(os.path.join(filepath,coord+'.txt')) for coord in coords}
    coord_to_imgxys = {coord:[] for coord in coords}
    coord_to_imgs = {coord:[] for coord in coords}
    for target in coords:
        d = os.path.join(filepath, target)
        fs = os.listdir(d)
        for fname in fs:
            path = os.path.join(d, fname)
            img = cv2.imread(path)
            if chn_first:
                img = np.transpose(img, (2,0,1))
            coord_to_imgs[target].append(img)
            coord_to_imgxys[target].append([float(item) for item in fname.split('.')[0].split('_')])
        coord_to_imgs[target] = np.array(coord_to_imgs[target], dtype='float')/255.0
        coord_to_imgxys[target] = np.array(coord_to_imgxys[target], dtype='float')
    return coord_to_imgs, coord_to_imgxys, coord_to_xys


class OILPALM:
    def __init__(self, clf_path, detect_path, test_split=0.3, chn_first=False):
        self.chn_first = chn_first
        self.clf_path = clf_path
        self.detect_path = detect_path
        self.data, self.label, self.detect_data, self.detect_xy_candi, self.detect_xy, self.source_name = self.__load_data__()
        self.test_split = test_split
        self.data_train, self.data_test, self.label_train, self.label_test = self.__split_train_test__()
        
    def __load_data__(self):
        img_a, label_a = imageFolder(os.path.join(self.clf_path, '0'), self.chn_first)
        img_b, label_b = imageFolder(os.path.join(self.clf_path, '1'), self.chn_first)
        img_d, label_d = imageFolder(os.path.join(self.clf_path, '3'), self.chn_first)
        img_e, label_e = imageFolder(os.path.join(self.clf_path, '4'), self.chn_first)
        detect_a, candi_xy_a, xy_a = detectionFolder(os.path.join(self.detect_path, '0'), self.chn_first)
        detect_b, candi_xy_b, xy_b = detectionFolder(os.path.join(self.detect_path, '1'), self.chn_first)
        detect_d, candi_xy_d, xy_d = detectionFolder(os.path.join(self.detect_path, '3'), self.chn_first)
        detect_e, candi_xy_e, xy_e = detectionFolder(os.path.join(self.detect_path, '4'), self.chn_first)
        data = [img_a, img_b, img_d, img_e]
        label = [label_a, label_b, label_d, label_e]
        detect_data = [detect_a, detect_b, detect_d, detect_e]
        detect_xy = [xy_a, xy_b, xy_d, xy_e]
        detect_xy_candi = [candi_xy_a, candi_xy_b, candi_xy_d, candi_xy_e]
        return data, label, detect_data, detect_xy_candi, detect_xy, ['0' ,'1', '3', '4']

    def __split_train_test__(self):
        data_train = []
        data_test = []
        label_train = []
        label_test = []
        for i in range(len(self.data)):
            length = self.data[i].shape[0]
            test_size = int(self.test_split * length)
            test_idx = np.random.choice(np.arange(length), test_size, replace=False)
            data_test.append(self.data[i][test_idx])
            data_train.append(np.delete(self.data[i], test_idx, axis=0))
            label_test.append(self.label[i][test_idx])
            label_train.append(np.delete(self.label[i], test_idx, axis=0))
        return np.array(data_train), np.array(data_test), np.array(label_train), np.array(label_test)

    def generator(self, testSource, modeltype, batch_size=32):
        sourceId = self.source_name.index(testSource)
        trainSamples = np.delete(self.data_train, sourceId)
        trainLabels = np.delete(self.label_train, sourceId)
        trainDomainIds = [np.ones(trainLabels[i].shape)*i for i in range(len(trainLabels))]

        while True:
            sampleId = [np.random.choice(np.arange(len(item)), batch_size, replace=False) for item in trainLabels]
            batch_x = np.concatenate([trainSamples[i][sampleId[i]] for i in range(len(sampleId))], axis=0)
            batch_y = np.concatenate([trainLabels[i][sampleId[i]] for i in range(len(sampleId))], axis=0)
            batch_d = np.concatenate([trainDomainIds[i][sampleId[i]] for i in range(len(sampleId))], axis=0)
            list_x = [torch.Tensor(trainSamples[i][sampleId[i]]) for i in range(len(sampleId))]
            list_y = [torch.Tensor(trainLabels[i][sampleId[i]]) for i in range(len(sampleId))]
            if modeltype == 'MMD_DRCN':
                yield (batch_x.astype(float), [batch_x.astype(float), batch_d.astype(int), batch_y.astype(int)])
            elif modeltype == 'DANN':
                yield (list_x, list_y)
            elif modeltype == 'DRCN':
                yield (batch_x.astype(float), [batch_x.astype(float), batch_y.astype(int)])
            elif modeltype == 'Base': 
                yield (batch_x.astype(float), batch_y.astype(int))

                

    def getValData(self, testSource):
        sourceId = self.source_name.index(testSource)
        valSamples = np.delete(self.data_test, sourceId)
        valLabels = np.delete(self.label_test, sourceId)
        valDomainIds = [np.ones(valLabels[i].shape) * i for i in range(len(valLabels))]
        
        s, l, d = np.concatenate(valSamples, axis=0), \
                np.concatenate(valLabels, axis=0).astype(int), \
                np.concatenate(valDomainIds, axis=0).astype(int)
        index = np.arange(s.shape[0])
        np.random.shuffle(index)
        s = s[index]
        l = l[index]
        d = d[index]
        return s, l, d

    def getTestData(self, testSource):
        sourceId = self.source_name.index(testSource)
        testSamples = self.data_test[sourceId]
        testLabels = self.label_test[sourceId]
        return testSamples, testLabels.astype(int)

    def getDetectData(self, testSource):
        sourceId = self.source_name.index(testSource)
        detectSamples = self.detect_data[sourceId]
        detectXy = self.detect_xy[sourceId]
        detectXyCandi = self.detect_xy_candi[sourceId] 
        return detectSamples, detectXyCandi, detectXy

        
        



            
            




    

        
        


