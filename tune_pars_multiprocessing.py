# -*- coding: utf-8 -*-
"""
@author: Fabio Sigrist
"""

try:
    import xgboost as xgb
except ImportError:
    pass
from joblib import Parallel, delayed
import sys
try:
   import cPickle as pickle
except:
   import pickle
import numpy as np
from sklearn.metrics import accuracy_score

##Run this file from command line
def do_tune_boost(lr,model,Xtrain,ytrain,seed,dataset,Xvalid,yvalid,n_estimators=1000,
                printres=False,min_weight_leaf=1.,min_samples_leaf=1,mod="GN",max_depth=5):
    PoissonTask=["poisson","insurance","poisson_r"]
    GammaTask=["gamma","gamma_r"]
    TobitTask=["tobit","tobit_r"]
    MSRTask=["msr_f3","msr_r","birthweight","malnutrition","head_circumf","bodyfat"]
    RegrDatasets=PoissonTask+GammaTask+TobitTask+MSRTask
    BinClassifDatasets=["bin_clasif","bin_clasif_fht1",
                     "ionosphere","sonar","adult","titanic","bank","cancer","ijcnn"]
    MultiClassifDatasets=["multi_clasif","multi_clasif_fht","multi_clasif_cos",
                   "digits","car","glass","satimage","letter","smartphone","covtype","usps"]
    n_estimators=int(n_estimators)
    min_samples_leaf=int(min_samples_leaf)
    nests=range(1,n_estimators+1,1)           
    model.set_params(learning_rate=lr,n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,min_weight_leaf=min_weight_leaf)
    
    test_score=np.empty(n_estimators+1)
    test_score[:]=np.nan
    if mod in ["GG","GN","NN"]:
        model.fit(Xtrain, ytrain)
        if dataset in RegrDatasets:
            for ind, pred in enumerate(model.staged_predict(Xvalid)):
                i=ind+1##i=nb of trees
                if i in nests:
                    test_score[i] = model.loss_(yvalid, pred)
        if dataset in BinClassifDatasets+MultiClassifDatasets:
            for ind, pred in enumerate(model.staged_decision_function(Xvalid)):
                i=ind+1##i=nb of trees
                if i in nests:
                    if dataset in BinClassifDatasets: test_score[i] = 1-accuracy_score(yvalid, (pred>0).astype(float))
                    if dataset in MultiClassifDatasets: test_score[i] = 1-accuracy_score(yvalid, np.argmax(pred,axis=1))
    ##XGBoost            
    if mod=="XG":
        dtrain = xgb.DMatrix(Xtrain, label=ytrain)
        dvalid = xgb.DMatrix(Xvalid, label=yvalid)
        if dataset in BinClassifDatasets:  
            objective='binary:logistic'
            score_name='error'
        if dataset in MultiClassifDatasets: 
            objective='multi:softprob'
            score_name='merror'
        if dataset in PoissonTask: 
            objective='count:poisson'
            score_name='poisson-nloglik'
        if dataset in GammaTask: 
            objective='reg:gamma'
            score_name="gamma-nloglik"
        param = {'max_depth': max_depth, 'eta': lr, 'lambda': 0,
                 'objective': objective, 'min_child_weight': min_weight_leaf,
                 'tree_method': 'exact', 'seed':seed,
                 'max_delta_step': 0, 'eval_metric': [score_name]}
        if dataset in MultiClassifDatasets: param['num_class']=len(np.unique(ytrain))
        watchlist = [(dvalid, 'eval')]
        evals_result=dict()
        try:
            xgb.train(params=param, dtrain=dtrain, num_boost_round=n_estimators,
                            evals=watchlist,verbose_eval=False,evals_result=evals_result)
            test_score[nests] = evals_result['eval'][score_name]
        except ValueError as e:
            print("XGBoost error in evaluation for "+ str(dataset)+" & LR="+str(lr)+": \""+str(e)+"\"")
            test_score[nests] = 999999999
    opt_score=np.nanmin(test_score[nests])
    try:
        opt_nest_score=nests[np.nanargmin(test_score[nests])]
    except ValueError:
        opt_nest_score=n_estimators
    if printres:
        print("lr,min_weight_leaf,min_samples_leaf,opt_nest_score,opt_score")
        print([lr,min_weight_leaf,min_samples_leaf,opt_nest_score,opt_score])
    return([lr,min_weight_leaf,min_samples_leaf,opt_nest_score,opt_score])


if __name__ == '__main__':
    n_estimators=int(sys.argv[1])
    seed=int(sys.argv[2])
    ncores=int(sys.argv[3])
    dataset=str(sys.argv[4])
    path=str(sys.argv[5])
    mod=str(sys.argv[6])

    param_grid_cross_prod=pickle.load(open(path+'param_grid_cross_prod.p', 'rb'))
    Xtrain=pickle.load(open(path+'Xtrain.p', 'rb'))
    ytrain=pickle.load(open(path+'ytrain.p', 'rb'))
    Xvalid=pickle.load(open(path+'Xvalid.p', 'rb'))
    yvalid=pickle.load(open(path+'yvalid.p', 'rb'))
    model=pickle.load(open(path+'model.p', 'rb'))
    
    results = Parallel(n_jobs=ncores)(delayed(do_tune_boost)(lr=lr,min_weight_leaf=mw,min_samples_leaf=ms,
                       model=model,Xtrain=Xtrain,ytrain=ytrain,seed=seed,dataset=dataset,
                       n_estimators=n_estimators,Xvalid=Xvalid,yvalid=yvalid,mod=mod) for (lr,mw,ms) in param_grid_cross_prod)
    pickle.dump(results, open(path+'results.p', 'wb'))
#    print(results)
