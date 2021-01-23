import numpy as np
import pandas as pd
from dataSets import OILPALM
from networks import MMD_DRCN
from iouAcc import calculate_acc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.metrics import sparse_categorical_crossentropy
import os
import gc
from keras import backend as K
import sys
import argparse
from keras.models import load_model
# argument settings
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='Base')
# method  Base DRCN MMD_DRCN 
parser.add_argument('--cuda','-g', type=int, default=5)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--ae', type=float, default=0.5)
parser.add_argument('--mmd', type=float, default=5)
parser.add_argument('--cls', type=float, default=1.0)
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
patience = args.patience
num_domains = 4

datasets = OILPALM('/path/to/trainingdata', '/path/to/detectingdata', test_split = 0.3) 
source_name = datasets.source_name
# results list
best_acc = np.zeros([num_domains])
best_F1 = np.zeros([num_domains])
best_detect_F1 = np.zeros([num_domains])
best_detect_Rec = np.zeros([num_domains])
best_detect_Prec = np.zeros([num_domains])
detect_accs = []
# results dataframe save
acc_sum = {}
for k in ['method' ,'domain', 'area', 'TP', 'FP', 'FN', 'recall', 'precision', 'F1_score']:
    acc_sum[k] = []
# modeltype
modeltype = args.method
# leave one domain out strategy
for idxSource, name in enumerate(source_name):
    print('Testing on {}, training on remain...'.format(name))
    generator = datasets.generator(name, modeltype, batch_size=100) 
    #print(next(generator))
    val_data, val_y, val_domain = datasets.getValData(name)
    test_data, test_y = datasets.getTestData(name)
    detect_data, detect_xy_candi, detect_xy = datasets.getDetectData(name)

    mmd_drcn = MMD_DRCN(num_domains-1, [16,16,3], 2, ae=args.ae, mmd=args.mmd, cls=args.cls)
    if modeltype == 'DRCN':
        model = mmd_drcn.makeDRCN()
    elif modeltype == 'Base':
        model = mmd_drcn.makeBase()
    elif modeltype == 'MMD_DRCN':
        model = mmd_drcn.makeMMD_DRCN() 
        
    metrics_name = model.metrics_names

    step = 0
    count = 0
    while True:
        model.fit_generator(generator, steps_per_epoch=100, epochs=1, use_multiprocessing=True)
        step += 1
        if modeltype == 'MMD_DRCN':
            results = model.evaluate(val_data, [val_data, val_domain, val_y], batch_size = 300)
        elif modeltype == 'DRCN':
            results = model.evaluate(val_data,[val_data, val_y], batch_size = 300)
        elif modeltype == 'Base':
            results = model.evaluate(val_data, val_y, batch_size = 300)

        print('Step {}'.format(step))
        if modeltype == 'Base':
            print('Loss',results)
        else: 
            for idx, mname in enumerate(metrics_name):
                print('{}: {:.4f}'.format(mname, results[idx]))
            
        # test on target domain data
        if modeltype == 'Base':
            test_pred = model.predict(test_data)
        elif modeltype ==  'DRCN':
            _, test_pred = model.predict(test_data)
        elif modeltype ==  'MMD_DRCN':
            _, _, test_pred = model.predict(test_data)
        label = np.argmax(test_pred, axis = 1)
        acc = accuracy_score(test_y, label)
        prec = precision_score(test_y, label, pos_label=0)
        rec = recall_score(test_y, label, pos_label=0)
        f1 = f1_score(test_y, label, pos_label=0)
        print('[TEST] ACC: {:.4}'.format(acc*100))
        print('[TEST] PRECISION: {:.4}'.format(prec*100))
        print('[TEST] RECALL: {:.4}'.format(rec*100))
        print('[TEST] F1: {:.4}'.format(f1*100))

        if acc > best_acc[idxSource]:
            best_acc[idxSource] = acc
        if step >= patience:
            break
    # detection using new image
    recall_arr = []
    precision_arr =[]
    F1_score_arr = []
    for area in detect_data.keys(): 
        print("detect on area {}".format(area))
        if modeltype == 'Base':
            detect_pred = model.predict(detect_data[area])
        elif modeltype == 'DRCN':
            _, detect_pred = model.predict(detect_data[area])
        elif modeltype == 'MMD_DRCN':
            _, _, detect_pred = model.predict(detect_data[area])
        detect_label = np.argmax(detect_pred, axis=1)
        pred_xy = detect_xy_candi[area][detect_label==0]
        if len(pred_xy)==0:
             recall = 0
             precision = 0
             F1_score = 0
        else:
            TP_xy, FP_xy, FN_xy = calculate_acc(pred_xy, detect_xy[area], args.threshold)
            if len(TP_xy)==0:
                recall = 0
                precision = 0
                F1_score = 0
            else:
                recall = len(TP_xy)/(len(TP_xy)+len(FN_xy))
                precision = len(TP_xy)/(len(TP_xy)+len(FP_xy))
                F1_score = 2.0*precision*recall/(precision+recall)

        recall_arr.append(recall)
        precision_arr.append(precision)
        F1_score_arr.append(F1_score)
        bbox = [int(u) for u in area.split('testArea')[1].split('_')]
        print(bbox)
        acc_sum['domain'].append(name)
        acc_sum['area'].append(area)
        acc_sum['TP'].append(len(TP_xy))
        acc_sum['FP'].append(len(FP_xy))
        acc_sum['FN'].append(len(FN_xy))
        acc_sum['recall'].append(recall)
        acc_sum['precision'].append(precision)
        acc_sum['F1_score'].append(F1_score)
        acc_sum['method'].append(args.method)


    print("F1, Precision, Recall")
    print(F1_score_arr)
    print(precision_arr)
    print(recall_arr)
        
    detect_acc = {"domain":name,
            "avg_recall":np.array(recall_arr).mean(),
            "avg_precision":np.array(precision_arr).mean(),
            "avg_F1":np.array(F1_score_arr).mean()}
    detect_accs.append(detect_acc)
    print('detect Accuracy')
    for key, value in detect_acc.items():
        print('{}:{}'.format(key, value))

    best_detect_F1[idxSource] = detect_acc['avg_F1']
    best_detect_Rec[idxSource] = detect_acc['avg_recall']
    best_detect_Prec[idxSource] = detect_acc['avg_precision']
print(source_name)
print("validation on target acc")
print(best_acc)
print("validation on target F1")
print(best_F1)
print("detect on target F1 Recall Presion")
print(best_detect_F1)
print(best_detect_Rec)
print(best_detect_Prec)
acc_df = pd.DataFrame(acc_sum)
#acc_df.to_csv('detect_results/detect_acc_{}_mmd_{}_ae_{}_cls_{}.csv'.format(modeltype, args.mmd, args.ae, args.cls))
print("summary")
print(detect_accs)








    


