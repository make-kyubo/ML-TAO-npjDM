# predict the redness of the conjunctiva

import numpy as np
import pandas as pd
import cv2
import os
import datetime
import pickle
import multiprocessing as mp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline

def rgb_level(image):
    bgr_bin = 32
    bgr_step = image // (256//bgr_bin)
    b_step = bgr_step[:,:,0]
    g_step = bgr_step[:,:,1]
    r_step = bgr_step[:,:,2]
    return r_step.flatten(), g_step.flatten(), b_step.flatten()

def read_learn_test(iter):
    # to load two kinds of images
    dir_path_1 = 'cropped/Redness_of_conjunctiva/lateral/'
    img_size_1 = (150,150)
    dir_path_2 = 'cropped/Redness_of_conjunctiva/medial/'
    img_size_2 = (100,100)
    
    type_split = 'random'

    # It uses the randomly pre-split training/test sets
    with open('pre_split_'+type_split+'.pkl', 'rb') as file:
        test_set = pickle.load(file)
        train_set = pickle.load(file)

    symptom = "Red_conj"
    
    df_CAS = pd.read_excel('CAS_scoring.xlsx')
    df_CAS = df_CAS.dropna(axis=0)
    df_CAS = df_CAS.reset_index(drop=True)
    
    data_p_num = df_CAS['p_num'].values
    data_p_num = np.append(data_p_num, data_p_num)
    data_p_num.sort()
    
    data_score_Dr1_right = df_CAS[symptom+'_Dr1_right'][np.in1d(df_CAS['p_num'].values, data_p_num)].values
    data_score_Dr1_left = df_CAS[symptom+'_Dr1_left'][np.in1d(df_CAS['p_num'].values, data_p_num)].values
    data_score_Dr2_right = df_CAS[symptom+'_Dr2_right'][np.in1d(df_CAS['p_num'].values, data_p_num)].values
    data_score_Dr2_left = df_CAS[symptom+'_Dr2_left'][np.in1d(df_CAS['p_num'].values, data_p_num)].values
    data_score_Dr3_right = df_CAS[symptom+'_Dr3_right'][np.in1d(df_CAS['p_num'].values, data_p_num)].values
    data_score_Dr3_left = df_CAS[symptom+'_Dr3_left'][np.in1d(df_CAS['p_num'].values, data_p_num)].values
    data_score_right = df_CAS[symptom+'_Fin_right'][np.in1d(df_CAS['p_num'].values, data_p_num)].values
    data_score_left = df_CAS[symptom+'_Fin_left'][np.in1d(df_CAS['p_num'].values, data_p_num)].values
    
    data_score_Dr1 = np.empty((0,0))
    data_score_Dr2 = np.empty((0,0))
    data_score_Dr3 = np.empty((0,0))
    data_score = np.empty((0,0))
    for temp_i in range(0,len(data_score_right)):
        data_score_Dr1 = np.append(data_score_Dr1, data_score_Dr1_right[temp_i])
        data_score_Dr1 = np.append(data_score_Dr1, data_score_Dr1_left[temp_i])
        data_score_Dr2 = np.append(data_score_Dr2, data_score_Dr2_right[temp_i])
        data_score_Dr2 = np.append(data_score_Dr2, data_score_Dr2_left[temp_i])
        data_score_Dr3 = np.append(data_score_Dr3, data_score_Dr3_right[temp_i])
        data_score_Dr3 = np.append(data_score_Dr3, data_score_Dr3_left[temp_i])
        data_score = np.append(data_score, data_score_right[temp_i])
        data_score = np.append(data_score, data_score_left[temp_i])

    print("@@@@ iteration #"+str(iter)+" @@@@ ("+str(datetime.datetime.now())+")")

    train_p_num = train_set[iter,:]
    test_p_num = test_set[iter,:]

    train_p_num.sort()
    test_p_num.sort()
    
    train_p_num = list(train_p_num)
    test_p_num = list(test_p_num)
    
    X_train_doc = np.empty((0,img_size_1[0]*img_size_1[1]*3+img_size_2[0]*img_size_2[1]*3),dtype=np.uint8) # input data (images) used to train per-doc model
    Y_train_Dr1 = data_score_Dr1[np.in1d(data_p_num, train_p_num)].astype(np.uint8) # output data (labels) used to train per-doc model
    Y_train_Dr2 = data_score_Dr2[np.in1d(data_p_num, train_p_num)].astype(np.uint8)
    Y_train_Dr3 = data_score_Dr3[np.in1d(data_p_num, train_p_num)].astype(np.uint8)
    Y_train = data_score[np.in1d(data_p_num, train_p_num)].astype(np.uint8) # It will be used to determine class weights 
    for p_num in train_p_num:
        for l_or_r in ('right', 'left'):
            img_1 = cv2.imread(dir_path_1+str(p_num)+'_'+l_or_r+'.jpg')
            r_data_1, g_data_1, b_data_1 = rgb_level(img_1)
            r_data_1.astype(np.uint8)
            g_data_1.astype(np.uint8)
            b_data_1.astype(np.uint8)
            img_2 = cv2.imread(dir_path_2+str(p_num)+'_'+l_or_r+'.jpg')
            r_data_2, g_data_2, b_data_2 = rgb_level(img_2)
            r_data_2.astype(np.uint8)
            g_data_2.astype(np.uint8)
            b_data_2.astype(np.uint8)
            X_train_doc = np.vstack([X_train_doc, np.hstack([r_data_1, r_data_2, g_data_1, g_data_2, b_data_1, b_data_2])])
    print('FIN: making training set')
    
    X_test_doc = np.empty((0,img_size_1[0]*img_size_1[1]*3+img_size_2[0]*img_size_2[1]*3),dtype=np.uint8) # input data (images) used to test per-doc model
    Y_test_Dr1 = data_score_Dr1[np.in1d(data_p_num, test_p_num)].astype(np.uint8) # output data (labels) used to test per-doc model
    Y_test_Dr2 = data_score_Dr2[np.in1d(data_p_num, test_p_num)].astype(np.uint8)
    Y_test_Dr3 = data_score_Dr3[np.in1d(data_p_num, test_p_num)].astype(np.uint8)
    for p_num in test_p_num:
        for l_or_r in ('right', 'left'):
            img_1 = cv2.imread(dir_path_1+str(p_num)+'_'+l_or_r+'.jpg')
            r_data_1, g_data_1, b_data_1 = rgb_level(img_1)
            r_data_1.astype(np.uint8)
            g_data_1.astype(np.uint8)
            b_data_1.astype(np.uint8)
            img_2 = cv2.imread(dir_path_2+str(p_num)+'_'+l_or_r+'.jpg')
            r_data_2, g_data_2, b_data_2 = rgb_level(img_2)
            r_data_2.astype(np.uint8)
            g_data_2.astype(np.uint8)
            b_data_2.astype(np.uint8)
            X_test_doc = np.vstack([X_test_doc, np.hstack([r_data_1, r_data_2, g_data_1, g_data_2, b_data_1, b_data_2])])
    print('FIN: making test set (per-doc)')
    
    dir_path_save = 'result/'+symptom+'/'

    print('Doc: Dr1')
    sc_Dr1 = StandardScaler()
    kpca_Dr1 = KernelPCA(kernel='linear')
    svm_linear_Dr1 = svm.SVC(kernel='linear', C=0.001, probability=True,
                             class_weight={0:np.sqrt(len(Y_train)/sum(Y_train==0)),
                                           1:np.sqrt(len(Y_train)/sum(Y_train==1))})

    pipe_Dr1 = Pipeline(steps=[("scaler", sc_Dr1), ("pca", kpca_Dr1), ('svm', svm_linear_Dr1)])
    params = {"pca__n_components": [100, 200, 300]} # the list should be adjusted
    grid_svm_linear_Dr1 = GridSearchCV(pipe_Dr1, param_grid = params, cv = 3)
    grid_svm_linear_Dr1.fit(X_train_doc,Y_train_Dr1)
    prediction_Dr1 = grid_svm_linear_Dr1.predict(X_test_doc)
    pred_prob_Dr1 = grid_svm_linear_Dr1.predict_proba(X_test_doc)
    pred_prob_Dr1 = pred_prob_Dr1[:,1]

    conf_mat = confusion_matrix(Y_test_Dr1, prediction_Dr1)
    print(conf_mat)

    acc = (conf_mat[0,0]+conf_mat[1,1]) / len(prediction_Dr1)
    sens = conf_mat[1,1]/(conf_mat[1,1]+conf_mat[1,0])
    spec = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])

    print('acc:', acc)
    print('sens:', sens)
    print('spec:', spec)

    save_df = pd.DataFrame(
        {
            'p_num':test_set[iter,:],
            symptom+'_right':(Y_test_Dr1.reshape(-1,2))[:,0],
            symptom+'_left':(Y_test_Dr1.reshape(-1,2))[:,1],
            symptom+'_score':(np.logical_or((Y_test_Dr1.reshape(-1,2))[:,0],(Y_test_Dr1.reshape(-1,2))[:,1])).astype(np.uint8),
            symptom+'_right_pred':(prediction_Dr1.reshape(-1,2))[:,0],
            symptom+'_left_pred':(prediction_Dr1.reshape(-1,2))[:,1],
            symptom+'_score_pred':(np.logical_or((prediction_Dr1.reshape(-1,2))[:,0],(prediction_Dr1.reshape(-1,2))[:,1])).astype(np.uint8),
            symptom+'_right_pred_prob':(pred_prob_Dr1.reshape(-1,2))[:,0],
            symptom+'_left_pred_prob':(pred_prob_Dr1.reshape(-1,2))[:,1]
        }
    )
    save_df.to_excel(dir_path_save+str(iter)+'_Dr1.xlsx', index=False)
    print()
    
    print('Doc: Dr2')
    sc_Dr2 = StandardScaler()
    kpca_Dr2 = KernelPCA(kernel='linear')
    svm_linear_Dr2 = svm.SVC(kernel='linear', C=0.001, probability=True,
                             class_weight={0:np.sqrt(len(Y_train)/sum(Y_train==0)),
                                           1:np.sqrt(len(Y_train)/sum(Y_train==1))})

    pipe_Dr2 = Pipeline(steps=[("scaler", sc_Dr2), ("pca", kpca_Dr2), ('svm', svm_linear_Dr2)])
    params = {"pca__n_components": [100, 200, 300]} # the list should be adjusted
    grid_svm_linear_Dr2 = GridSearchCV(pipe_Dr2, param_grid = params, cv = 3)
    grid_svm_linear_Dr2.fit(X_train_doc,Y_train_Dr2)
    prediction_Dr2 = grid_svm_linear_Dr2.predict(X_test_doc)
    pred_prob_Dr2 = grid_svm_linear_Dr1.predict_proba(X_test_doc)
    pred_prob_Dr2 = pred_prob_Dr2[:,1]

    conf_mat = confusion_matrix(Y_test_Dr2, prediction_Dr2)
    print(conf_mat)

    acc = (conf_mat[0,0]+conf_mat[1,1]) / len(prediction_Dr2)
    sens = conf_mat[1,1]/(conf_mat[1,1]+conf_mat[1,0])
    spec = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])

    print('acc:', acc)
    print('sens:', sens)
    print('spec:', spec)

    save_df = pd.DataFrame(
        {
            'p_num':test_set[iter,:],
            symptom+'_right':(Y_test_Dr2.reshape(-1,2))[:,0],
            symptom+'_left':(Y_test_Dr2.reshape(-1,2))[:,1],
            symptom+'_score':(np.logical_or((Y_test_Dr2.reshape(-1,2))[:,0],(Y_test_Dr2.reshape(-1,2))[:,1])).astype(np.uint8),
            symptom+'_right_pred':(prediction_Dr2.reshape(-1,2))[:,0],
            symptom+'_left_pred':(prediction_Dr2.reshape(-1,2))[:,1],
            symptom+'_score_pred':(np.logical_or((prediction_Dr2.reshape(-1,2))[:,0],(prediction_Dr2.reshape(-1,2))[:,1])).astype(np.uint8),
            symptom+'_right_pred_prob':(pred_prob_Dr2.reshape(-1,2))[:,0],
            symptom+'_left_pred_prob':(pred_prob_Dr2.reshape(-1,2))[:,1]
        }
    )
    save_df.to_excel(dir_path_save+str(iter)+'_Dr2.xlsx', index=False)
    print()
    
    print('Doc: Dr3')
    sc_Dr3 = StandardScaler()
    kpca_Dr3 = KernelPCA(kernel='linear')
    svm_linear_Dr3 = svm.SVC(kernel='linear', C=0.001, probability=True,
                             class_weight={0:np.sqrt(len(Y_train)/sum(Y_train==0)),
                                           1:np.sqrt(len(Y_train)/sum(Y_train==1))})

    pipe_Dr3 = Pipeline(steps=[("scaler", sc_Dr3), ("pca", kpca_Dr3), ('svm', svm_linear_Dr3)])
    params = {"pca__n_components": [100, 200, 300]} # the list should be adjusted
    grid_svm_linear_Dr3 = GridSearchCV(pipe_Dr3, param_grid = params, cv = 3)
    grid_svm_linear_Dr3.fit(X_train_doc,Y_train_Dr3)
    prediction_Dr3 = grid_svm_linear_Dr3.predict(X_test_doc)
    pred_prob_Dr3 = grid_svm_linear_Dr1.predict_proba(X_test_doc)
    pred_prob_Dr3 = pred_prob_Dr3[:,1]

    conf_mat = confusion_matrix(Y_test_Dr3, prediction_Dr3)
    print(conf_mat)

    acc = (conf_mat[0,0]+conf_mat[1,1]) / len(prediction_Dr3)
    sens = conf_mat[1,1]/(conf_mat[1,1]+conf_mat[1,0])
    spec = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])

    print('acc:', acc)
    print('sens:', sens)
    print('spec:', spec)

    save_df = pd.DataFrame(
        {
            'p_num':test_set[iter,:],
            symptom+'_right':(Y_test_Dr3.reshape(-1,2))[:,0],
            symptom+'_left':(Y_test_Dr3.reshape(-1,2))[:,1],
            symptom+'_score':(np.logical_or((Y_test_Dr3.reshape(-1,2))[:,0],(Y_test_Dr3.reshape(-1,2))[:,1])).astype(np.uint8),
            symptom+'_right_pred':(prediction_Dr3.reshape(-1,2))[:,0],
            symptom+'_left_pred':(prediction_Dr3.reshape(-1,2))[:,1],
            symptom+'_score_pred':(np.logical_or((prediction_Dr3.reshape(-1,2))[:,0],(prediction_Dr3.reshape(-1,2))[:,1])).astype(np.uint8),
            symptom+'_right_pred_prob':(pred_prob_Dr3.reshape(-1,2))[:,0],
            symptom+'_left_pred_prob':(pred_prob_Dr3.reshape(-1,2))[:,1]
        }
    )
    save_df.to_excel(dir_path_save+str(iter)+'_Dr3.xlsx', index=False)
    print()

    # Additional codes will be needed to choose the majority of results
    
    print("@@@@ Results saved @@@@ ("+str(datetime.datetime.now())+")")
    print()

pool = mp.Pool(3)
pool.map(read_learn_test,range(0,30))