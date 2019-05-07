from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,auc

def getAucScore(predict,target):
    all_auc=0
    for i in range(len(target)):
        all_auc=all_auc+roc_auc_score(target[i],predict[i])
    return all_auc/len(target)

def getFlattenAucScore(predict,target,category=6):
    if category == 5:
        predict=predict[:,1:6]
        target=target[:,1:6]
    # # all_auc=0
    # for i in range(len(target)):
    #     all_auc=all_auc+roc_auc_score(target[i],predict[i])
    # predict=predict.flatten()
    # target=target.flatten()
    all_auc=roc_auc_score(target,predict,average="micro")
    return all_auc

