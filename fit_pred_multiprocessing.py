# -*- coding: utf-8 -*-
"""
@author: Fabio Sigrist
"""

##Run this file from command line
from joblib import Parallel, delayed
import sys
try:
   import cPickle as pickle
except:
   import pickle

def fit_pred(nest,lr,mw,ms,model,Xtrain,ytrain,Xtest,dataset,mod):
    PoissonTask=["poisson","insurance","poisson_r"]
    GammaTask=["gamma","gamma_r"]
    TobitTask=["tobit","tobit_r"]
    MSRTask=["msr_f3","msr_r","birthweight","malnutrition","head_circumf","bodyfat"]
    RegrDatasets=PoissonTask+GammaTask+TobitTask+MSRTask
    BinClassifDatasets=["bin_clasif","bin_clasif_fht1",
                     "ionosphere","sonar","adult","titanic","bank","cancer","ijcnn"]
    MultiClassifDatasets=["multi_clasif","multi_clasif_fht","multi_clasif_cos",
                   "digits","car","glass","satimage","letter","smartphone","covtype","usps"]
    model.set_params(learning_rate=lr,n_estimators=int(nest),min_samples_leaf=int(ms),min_weight_leaf=mw)
    model.fit(Xtrain, ytrain)
    if dataset in RegrDatasets: pred=model.predict(Xtest)
    if dataset in BinClassifDatasets+MultiClassifDatasets: pred=model.decision_function(Xtest)
    return(pred)

if __name__ == '__main__':
    ncores=int(sys.argv[1])
    dataset=str(sys.argv[2])
    mod=str(sys.argv[3])
    path=str(sys.argv[4])
    
    opt_params=pickle.load(open(path+'opt_params.p', 'rb'))
    model=pickle.load(open(path+'model.p', 'rb'))
    Xtrain=pickle.load(open(path+'Xtrain.p', 'rb'))
    ytrain=pickle.load(open(path+'ytrain.p', 'rb'))
    Xtest=pickle.load(open(path+'Xtest.p', 'rb'))

    results = Parallel(n_jobs=ncores)(delayed(fit_pred)(nest=nest,lr=lr,mw=mw,ms=ms,
                       model=model,Xtrain=Xtrain,ytrain=ytrain,Xtest=Xtest,dataset=dataset,mod=mod) for (nest,lr,mw,ms) in opt_params)
    pickle.dump(results, open(path+'preds.p', 'wb'))
