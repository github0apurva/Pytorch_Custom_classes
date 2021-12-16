import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve,  roc_auc_score, accuracy_score,   auc, confusion_matrix



def get_all_metric (train_y_predprob, test_y_predprob, train_y1, test_y1,th) :
    train_y_pred =  pd.Series( np.ravel( np.where(train_y_predprob > th, 1 , 0)) )   
    test_y_pred =   pd.Series( np.ravel( np.where(test_y_predprob > th, 1 , 0)) ) 
    
    train_y = pd.Series( np.ravel( train_y1 ) )   
    test_y  = pd.Series( np.ravel( test_y1 ) )
    
    print ("                          accc  f1 recall precision ")
    print (" metrics for train basic ", round(accuracy_score(train_y,train_y_pred),2), round(f1_score(train_y,train_y_pred),2),  round(recall_score(train_y,train_y_pred),2),  round(precision_score(train_y,train_y_pred),2) )
    print (" metrics for test basic  ", round(accuracy_score(test_y,test_y_pred),2), round(f1_score(test_y,test_y_pred),2),  round(recall_score(test_y,test_y_pred),2),  round(precision_score(test_y,test_y_pred),2) )
    print
    print (" confusion for train  & test basic ")
    print ( pd.concat ( [ pd.DataFrame([" "," "],columns=['train:act-->']), pd.crosstab ( train_y_pred, train_y ) , pd.DataFrame([" "," "],columns=[' ']), pd.DataFrame([" "," "],columns=['test:act-->']), pd.crosstab ( test_y_pred, test_y ) ], axis = 1 )  )

    get_roc_curve (  test_y , test_y_predprob, train_y, train_y_predprob)

    thres,score_xgb= get_metric_score(test_y, test_y_predprob )
    threst,score_xgbt= get_metric_score(train_y, train_y_predprob )
    print (thres)
    print ()
    train_y_pred =  pd.Series( np.ravel( np.where(train_y_predprob > thres["threshold"], 1 , 0)) )   
    test_y_pred =   pd.Series( np.ravel( np.where(test_y_predprob > thres["threshold"], 1 , 0)) ) 

    print ("                          accc  f1 recal preci ")
    print (" metrics for train basic ", round(accuracy_score(train_y,train_y_pred),2), round(f1_score(train_y,train_y_pred),2),  round(recall_score(train_y,train_y_pred),2),  round(precision_score(train_y,train_y_pred),2) )
    print (" metrics for test basic  ", round(accuracy_score(test_y,test_y_pred),2), round(f1_score(test_y,test_y_pred),2),  round(recall_score(test_y,test_y_pred),2),  round(precision_score(test_y,test_y_pred),2) )
    print
    print (" confusion for train  & test basic ")
    print ( pd.concat ( [ pd.DataFrame([" "," "],columns=['train:act-->']), pd.crosstab ( train_y_pred, train_y ) , 
                         pd.DataFrame([" "," "],columns=[' ']), pd.DataFrame([" "," "],columns=['test:act-->']), 
                         pd.crosstab ( test_y_pred, test_y ) ], axis = 1 )  )

    get_pre_rec_curve(score_xgb,score_xgbt)
    return score_xgb, score_xgbt


def get_roc_curve(y_test, y_score , y_train, y_train_score ):
    # Compute ROC curve and ROC area for each class

    fpr, tpr, thresholds =  roc_curve(y_test, y_score  )
    fprt, tprt, thresholdst =  roc_curve(y_train, y_train_score  )

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC test curve (area = %0.2f)' % auc (fpr, tpr))
    plt.plot(fprt, tprt, color='darkblue',
             lw=lw, label='ROC train curve (area = %0.2f)' % auc (fprt, tprt))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return

def get_metric_score(y_true, y_proba):
    '''
    actual and predicted prob ( input )

    '''
    best_threshold = 0
    best_score = 0
    accuracy,precision,recall,f1,fpr,thresholds,tnl,fpl,fnl,tpl = [],[],[],[],[],[],[],[],[],[]
#     for threshold in tqdm([i * 0.01 for i in range(1,100)]):
    for threshold in [i * 0.01 for i in range(1,100)]:
        y_pred = (y_proba>threshold).astype(int)
        score=f1_score(y_true=y_true, y_pred=y_pred)
        f1.append(score)
        accuracy.append(accuracy_score(y_true,y_pred))
        precision.append(precision_score(y_true,y_pred))
        recall.append(recall_score(y_true,y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr.append(fp/(fp+tn))
        tnl.append(tn)
        fpl.append(fp)
        fnl.append(fn)
        tpl.append(tp)
        thresholds.append(threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
        #-----
    model_score_df = pd.DataFrame([thresholds, tpl, fpl, tnl, fnl, accuracy,precision,recall,f1,fpr]).T
    model_score_df.columns = ['threshold', 'tp', 'fp', 'tn', 'fn','accuracy','precision','recall','f1','fpr']
    model_score_df = model_score_df.sort_values(by='threshold',ascending=False)
    search_result = {'threshold': best_threshold, 'f1': best_score}
    #------- roc -auc curve ---
#     fpr_rf, tpr_rf, _ = roc_curve(y_true, y_proba)
    return search_result,model_score_df

def get_pre_rec_curve(score_xgb,score_xgbt):
    plt.title('Precision (red) -- Recall (blue)')
    plt.plot( score_xgb.threshold, score_xgb.precision,'r--', score_xgb.threshold, score_xgb.recall , 'b--',
            score_xgbt.threshold, score_xgbt.precision,'r:', score_xgbt.threshold, score_xgbt.recall , 'b:' )
