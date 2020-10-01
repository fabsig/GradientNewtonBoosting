# -*- coding: utf-8 -*-
"""
@author: Fabio Sigrist

This is the code to reproduce the results of Sigrist (2018) 
    "Gradient and Newton Boosting for Classification and Regression".
    We compare gradient and Newton boosting, as well as hybrid gradient-Newton
    boosting with trees as base learners using various datasets and loss functions.
    See https://arxiv.org/abs/1808.03064 for more details.

Running all experiments with all settings for real-world and simulated datasets
    takes some time. The parameter 'which_data' specifies which experiments are run. 
    The file(s) 'results_summary_simulation=.csv' in the results folder 
    contains the summary of the results.
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
import sklearn.datasets as datasets
import copy
import KTBoost.KTBoost as ktb
try:
   import cPickle as pickle
except:
   import pickle
import subprocess
import os
from sklearn.metrics import accuracy_score
import xgboost as xgb# xgboost version 0.7 has been used in the paper
from scipy.special import logit

## Path where the code is
path_code="/code/"
## Path where the data is
path_data="/data/"
## Path where temporary data is saved during parameter tuning
path_temp="/data/"
## Results path
path_results="/results/"

"""
Functions used in the experiments below
"""
## Auxiliary function for laoding data. It returns the response variables and the predictor variabels in a design matrix
def get_response_design_matrix(data,indcat,indresp=0,naind=None,respToCat=False,narm=False):
    itransform = indcat + [indresp] if respToCat else indcat
    for i in itransform:
        le = preprocessing.LabelEncoder()
        le.fit(data[:,i])
        data[:,i]=le.transform(data[:,i])
    data[np.where(data=="NULL")]=np.nan
    data=data.astype(np.float)
    if naind is not None:
        for ina in naind:  data[np.isnan(data[:,ina]),ina]=np.median(data[~np.isnan(data[:,ina]),ina])
    if narm:
        mask = ~np.any(np.isnan(data), axis=1)
        data = data[mask]
    contvars=np.delete(data, [indresp]+indcat, axis=1)
    y=data[:,indresp]
    if len(indcat)>0:
        enc = preprocessing.OneHotEncoder()
        enc.fit(data[:,indcat])
        dummies=enc.transform(data[:,indcat]).toarray()
        X=np.append(contvars,dummies,axis=1)
    else:
        X=contvars
    return [y,X]

## Auxiliary function for simulting data
def sum_sq_fct(ntot,p=10,dist="norm"):
    if dist=="norm": X=np.random.normal(size=(ntot*p)).reshape((ntot, p))
    if dist=="unif": X=np.random.rand(ntot,p)
    r=np.zeros(X.shape[0])
    for i in range(0,X.shape[1]): r+=(X[:,i])**2
    return(X,r)
    
## Function for laoding and simulting of data
def load_simulate_data(dataset,n,gamma=2,sigma=1,seed=None,
                  yl=0,yu=1,path=None,npart=3,FullDataSet=False):
    ntot = npart*n
    if seed is not None: np.random.seed(seed)
    """
    Real regression data
    """
    if dataset=="birthweight":##n=150, p=5
        # Source: tbm package (see Load_Hothorn_data.R file)
        data = np.genfromtxt(path+'birthweight.csv', delimiter=',',dtype=float,skip_header=1)
        X=data[:,1:6]
        y=data[:,0]
    if dataset=="malnutrition":##n=24166, p=42
        # Source: tbm package (see Load_Hothorn_data.R file)
        data = np.genfromtxt(path+'malnutrition.csv', delimiter=',',dtype=float,skip_header=1)
        y,X=get_response_design_matrix(data,indresp=0,indcat=range(7,21),respToCat=False)
    if dataset in ["insurance","liberty"]:##n=50999
        #Liberty Mutual Group: Property Inspection Prediction
        #Source:  https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction
        data = np.genfromtxt(path+'insurance_data.csv', delimiter=',',skip_header=1,dtype=str)
        indcat=[4,5,6,7,8,9,11,12,14,15,16,17,20,22,25,28,29,30]
        y,X=get_response_design_matrix(data,indresp=0,indcat=indcat)
        y=y-1##Shift by one, otherwise the Poisson distribution cannot fit since there are no zeros
    """
    Real classification data
    """
    if dataset=="digits":##n=5620
        # Source: https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
        data = np.genfromtxt(path+'digits.csv', delimiter=',',dtype=str)
        y,X=get_response_design_matrix(data,indresp=64,indcat=[],respToCat=True)
    if dataset=="cancer":##n=699
        # Source: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
        data = np.genfromtxt(path+'cancer_data.csv', delimiter=',')
        X=data[:,1:10]
        y=data[:,10]
        y[y==2]=0
        y[y==4]=1
    if dataset=="bank":##n=41188
        # Source: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
        ##Variable duration removed in csv as suggested on the website
        data = np.genfromtxt(path+'bank.csv', delimiter=',',dtype=str,skip_header=1)
        indcat=range(1,10)+[13]
        y,X=get_response_design_matrix(data,indresp=19,indcat=indcat,respToCat=True)
    if dataset=="car":##n=1728
        ##source: https://archive.ics.uci.edu/ml/datasets/car+evaluation
        data = np.genfromtxt(path+'car_data.csv', delimiter=',',dtype=str)
        indcat=[0,1,2,3,4,5]
        y,X=get_response_design_matrix(data,indresp=6,indcat=indcat,respToCat=True)
    if dataset=="adult":##n=48842
        # Source: http://archive.ics.uci.edu/ml/datasets/Adult
        datatrain = np.genfromtxt(path+'adult_train_data.csv', delimiter=',',dtype=str)
        datatest = np.genfromtxt(path+'adult_test_data.csv', delimiter=',',dtype=str)
        data=np.append(datatrain,datatest,axis=0)
        indcat=[1,3,5,6,7,8,9,13]
        y,X=get_response_design_matrix(data,indresp=14,indcat=indcat,respToCat=True)
    if dataset=="glass":##n=214
        data = np.genfromtxt(path+'glass.csv', delimiter=',',dtype=str)
        y,X=get_response_design_matrix(data,indresp=9,indcat=[],respToCat=True)
    if dataset=="ionosphere":##n=351
        ##source https://archive.ics.uci.edu/ml/datasets/ionosphere
        data = np.genfromtxt(path+'ionosphere.csv', delimiter=',',dtype=str)
        y,X=get_response_design_matrix(data,indresp=34,indcat=[],respToCat=True)
    if dataset=="letter":##n=20000
        ##source: https://archive.ics.uci.edu/ml/datasets/letter+recognition
        data = np.genfromtxt(path+'letter.csv', delimiter=',',dtype=str)
        y,X=get_response_design_matrix(data,indresp=0,indcat=[],respToCat=True)
    if dataset=="sonar":##n=208
        ##source: http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
        data = np.genfromtxt(path+'sonar.csv', delimiter=',',dtype=str)
        y,X=get_response_design_matrix(data,indresp=60,indcat=[],respToCat=True)
    if dataset=="satimage":##n=6438
        ##source: https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)
        data = np.genfromtxt(path+'satimage.csv', delimiter=',',dtype=str)
        y,X=get_response_design_matrix(data,indresp=36,indcat=[],respToCat=True)
    if dataset=="smartphone":##n=10299
        # Source: https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
        Xtrain = np.genfromtxt(path+'smartphone_X_train.csv', delimiter=',')
        Xtest = np.genfromtxt(path+'smartphone_X_test.csv', delimiter=',')
        ytrain = np.genfromtxt(path+'smartphone_y_train.csv', delimiter=',')
        ytest = np.genfromtxt(path+'smartphone_y_test.csv', delimiter=',')
        X=np.append(Xtrain,Xtest,axis=0)
        y=np.append(ytrain,ytest,axis=0)
        y=y-1
    if dataset=="usps":##n=9298, p=256, classes=10
        ##https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
        train = load_svmlight_file(path+'usps.bz2')
        test = load_svmlight_file(path+'usps.t.bz2')
        X=np.append(train[0].toarray(),test[0].toarray(),axis=0)
        y=np.append(train[1],test[1],axis=0)
        y=y-1
    if dataset=="ijcnn":##n=141691, p=22, classes=2
        ##https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
        test = load_svmlight_file(path+'ijcnn1.t.bz2')
        valid = load_svmlight_file(path+'ijcnn1.val.bz2')
        train = load_svmlight_file(path+'ijcnn1.tr.bz2')
        X=np.append(np.append(train[0].toarray(),valid[0].toarray(),axis=0),test[0].toarray(),axis=0)
        y=np.append(np.append(train[1],valid[1],axis=0),test[1],axis=0)
        y[y==-1]=0
    if dataset=="covtype":##n=581012, p=54, classes=7
        ##https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
        data = load_svmlight_file(path+'covtype.bz2')
        X=data[0].toarray()
        y=data[1]
        y=y-1
    """
    Simulated regression data
    """
    if dataset in ["poisson","gamma","tobit"]: 
        X, lp = datasets.make_friedman3(n_samples=ntot)
        lp=lp*5+0.2
        if dataset=="tobit":
            y=np.random.normal(loc=lp,scale=sigma)
            y[y>=yu]=yu
            y[y<=yl]=yl
        if dataset=="poisson": y=np.random.poisson(lp)
        if dataset=="gamma": y=np.random.gamma(scale=lp/gamma,shape=gamma)
    if dataset in ["msr_f3","msr_r"]:
        if dataset in ["msr_f3"]:
            X_mean, lp_mean = datasets.make_friedman3(n_samples=ntot)
            X_sd, lp_sd = datasets.make_friedman3(n_samples=ntot)
            lp_sd=lp_sd*5+0.2
            lp_mean=lp_mean*5+0.2
        if dataset in ["msr_r"]: ##Simulate according to Ridgeway (1999) PhD thesis
            X_mean=np.random.rand(ntot,2)
            lp_mean=np.exp(2*np.sin(3*X_mean[:,0]+5*X_mean[:,0]**2)-2*np.sin(3*(X_mean[:,1]+0.1)+5*(X_mean[:,1]+0.1)**2))
            X_sd=np.random.rand(ntot,2)
            lp_sd=np.exp(2*np.sin(3*X_sd[:,0]+5*X_sd[:,0]**2)-2*np.sin(3*(X_sd[:,1]+0.1)+5*(X_sd[:,1]+0.1)**2))
        y=np.random.normal(loc=lp_mean,scale=lp_sd)
        X=np.append(X_mean,X_sd,axis=1)
    if dataset in ["poisson_r","gamma_r","tobit_r"]: ##Simulate according to Ridgeway (1999) PhD thesis
        X=np.random.rand(ntot,2)
        lp=np.exp(2*np.sin(3*X[:,0]+5*X[:,0]**2)-2*np.sin(3*(X[:,1]+0.1)+5*(X[:,1]+0.1)**2))
        if dataset=="poisson_r": y=np.random.poisson(lp)
        if dataset=="gamma_r": y=np.random.gamma(scale=lp/gamma,shape=gamma)
        if dataset=="tobit_r":
            y=np.random.normal(loc=lp,scale=sigma)
            y[y>=yu]=yu
            y[y<=yl]=yl
    """
    Simulated classification data
    """
    if dataset=="bin_clasif": 
        X, y = datasets.make_classification(n_samples=ntot,n_informative=10,n_redundant=0,n_features=10)
    if dataset=="bin_clasif_fht1":##Simulate according to FHT00 page361 and MWB07 (JMLR) p416
        X=np.random.normal(size=(ntot*10)).reshape((ntot, 10))
        lp=np.zeros(ntot)
        lp+=1
        for i in range(0,6):
            lp+=(-1)**(i+1)*X[:,i]
        lp=lp*np.sum(X[:,0:6],axis=1)*10
        prob=1/(1+np.exp(-lp))
        y=np.random.binomial(1,prob)
    if dataset=="multi_clasif":
        X, y = datasets.make_classification(n_samples=ntot,n_classes=5,n_informative=10,n_redundant=0,n_features=10)
    if dataset=="multi_clasif_fht": ##Simulate according to FHT00 p358 
        X,r=sum_sq_fct(ntot,p=10)
        y=r.copy()
        cuts=[-1e50,6.2,8.3,10.5,13.4,1e50]
        for i in range(0,len(cuts)-1): y[(r>=cuts[i]) & (r<cuts[i+1])]=i
    ##Drop NAs
    incl=~np.isnan(X).any(axis=1)
    X=X[incl]
    y=y[incl]
    ## Dndex for splitting data
    randind = list(np.arange(len(y)))
    ##Shuffle data
    np.random.shuffle(randind)
    ##Define different data for training, validation, and testing
    if FullDataSet: ntot = len(y)
    ntot=np.min([ntot,len(y)])
    ##Check the minimal number of samples per class
    if dataset in BinClassifDatasets+MultiClassifDatasets:
        ue, counts = np.unique(y[randind[0:int(ntot/npart)]], return_counts=True)
        while ((len(ue)<len(np.unique(y))) | (np.min(counts)<2)):
            np.random.shuffle(randind)
            ue, counts = np.unique(y[randind[0:int(ntot/npart)]], return_counts=True)
    ntrain=int(ntot/npart)
    nvalid=int(ntot/npart) if npart==3 else 0
    indtrain=randind[0:ntrain]
    indvalid=randind[ntrain:(ntrain+nvalid)]
    indtest=randind[(ntrain+nvalid):ntot]
    if npart==2:
        return(X[indtrain,:],X[indtest,:],y[indtrain],y[indtest])
    if npart==3:
        return(X[indtrain,:],X[indvalid,:],X[indtest,:],y[indtrain],y[indvalid],y[indtest])

def return_limits_tobit(dataset,sigma=1):
    if dataset in TobitTask:
        X1,X2,X3,y1,y2,y3=load_simulate_data(dataset=dataset,n=100000,sigma=sigma,seed=1,yl=-1e30,yu=1e30)
        yl,yu=np.percentile(np.concatenate((y1,y2,y3)),q=[33.33,66.66])
    else:
        yl=-1e30
        yu=1e30
    return(yl,yu)

## Creates boosting model
def get_ktboost_model(mod,dataset,max_depth,n_estimators,seed,yl=0,yu=1,gamma=1,
                      learning_rate=0.1,min_samples_leaf=1,min_weight_leaf=1.):
    if dataset in PoissonTask: loss="poisson"
    if dataset in BinClassifDatasets+MultiClassifDatasets: loss='deviance'
    if dataset in GammaTask: loss="gamma"
    if dataset in TobitTask: loss="tobit"
    if dataset in MSRTask: loss="msr"
    if mod in ['GG',"XG"]: update_step="gradient"
    if mod in ['GN']: update_step="hybrid"
    if mod in ['NN']: update_step="newton"
    if dataset in RegrDatasets: 
        model = ktb.BoostingRegressor(loss=loss,criterion='mse',max_depth=max_depth,min_weight_leaf=min_weight_leaf,min_samples_leaf=min_samples_leaf,
                                             n_estimators=n_estimators,learning_rate=learning_rate,random_state=seed,
                                             update_step=update_step,yl=yl,yu=yu,gamma=gamma)
    if dataset in BinClassifDatasets+MultiClassifDatasets: 
        model = ktb.BoostingClassifier(loss=loss,criterion='mse',max_depth=max_depth,min_weight_leaf=min_weight_leaf,min_samples_leaf=min_samples_leaf,
                                              n_estimators=n_estimators,learning_rate=learning_rate,random_state=seed,
                                              update_step=update_step)
    return(model)

## Creates parameter grid for tuning
def get_param_grid_cross_prod(param_grid):
    cross_prod=[]
    for lr in param_grid['learning_rate']:
        for mw in param_grid['min_weight_leaf']:
            for ms in param_grid['min_samples_leaf']:
                cross_prod+=[(lr,mw,ms)]
    return(cross_prod)    

## Auxiliary function
def get_data_param_XGBoost(Xtrain,Xtest,ytrain,ytest,dataset,max_depth,min_weight_leaf,learning_rate=0.1,seed=0):
    dtrain = xgb.DMatrix(Xtrain, label=ytrain)
    dtest = xgb.DMatrix(Xtest, label=ytest)
    if dataset in BinClassifDatasets:  objective='binary:logistic'
    if dataset in MultiClassifDatasets: objective='multi:softprob'
    if dataset in PoissonTask: objective='count:poisson'
    if dataset in GammaTask: objective='reg:gamma'
    param = {'max_depth': max_depth, 'eta': learning_rate, 'lambda': 0,
             'objective': objective, 'min_child_weight': min_weight_leaf,
             'tree_method': 'exact', 'seed': seed, 'max_delta_step': 0}
    if dataset in MultiClassifDatasets: param['num_class']=len(np.unique(ytrain))
    return(dtrain,dtest,param)
    
## Find optimal tuning parameters and calculate test errors
def tune_pars_calc_test_error(mod,max_depth,n_estimators,n,seed,param_grid,
                              dataset,path_temp,path_code,Xtrain,Xtest,Xvalid,ytrain,ytest,yvalid,
                              ncores=1,yl=0,yu=1,gamma=1,calcOptLoss=True):
    model=get_ktboost_model(mod=mod,dataset=dataset,max_depth=max_depth,n_estimators=n_estimators,
                            seed=seed,yl=yl,yu=yu,gamma=gamma)
    param_grid_cross_prod=get_param_grid_cross_prod(param_grid)
    pickle.dump(param_grid_cross_prod, open(path_temp+'param_grid_cross_prod.p', 'wb'))
    pickle.dump(Xtrain, open(path_temp+'Xtrain.p', 'wb'))##For use below in CV function
    pickle.dump(ytrain, open(path_temp+'ytrain.p', 'wb'))
    pickle.dump(Xvalid, open(path_temp+'Xvalid.p', 'wb'))
    pickle.dump(yvalid, open(path_temp+'yvalid.p', 'wb'))
    pickle.dump(Xtest, open(path_temp+'Xtest.p', 'wb'))
    pickle.dump(model, open(path_temp+'model.p', 'wb'))
    ##Choose tuning parameters
    os.chdir(path_code)
    cmd="python tune_pars_multiprocessing.py "+str(n_estimators)+" "+str(seed)+" "+str(ncores)+" "+dataset+" "+path_temp+" "+mod
    status = subprocess.call(cmd, shell=True)       
    if not status == 0: raise ValueError("Error in parameter tuning for " + str(dataset) + " & " + str(mod))
    cvscores=pickle.load(open(path_temp+'results.p', 'rb'))
    cvscores=np.array(cvscores)
    opt_pars_score=list(cvscores[np.nanargmin(cvscores[:,4]),[3,0,1,2]])
    ## Results when setting the number of samples per leaf parameter to its default value
    cvscores_noW=cvscores[(cvscores[:,1]==np.min(cvscores[:,1])) & (cvscores[:,2]==np.min(cvscores[:,2])),:]
    opt_pars_score_noW=list(cvscores_noW[np.nanargmin(cvscores_noW[:,4]),[3,0,1,2]])
    opt_params=[opt_pars_score,opt_pars_score_noW]
    ##Train best model and make predictions for evaluation
    if calcOptLoss:
        if mod == 'XG':
            dtrain,dtest,param=get_data_param_XGBoost(Xtrain=Xtrain,Xtest=Xtest,ytrain=ytrain,ytest=ytest,
                       dataset=dataset,max_depth=max_depth,min_weight_leaf=1.,seed=seed)
            preds=[]
            for nest,lr,mw,ms in opt_params:
                param['eta']=lr
                param['min_child_weight']=mw
                modxgb = xgb.train(param, dtrain, int(nest))
                pred = modxgb.predict(dtest)
                if (dataset in RegrDatasets) & (dataset not in TobitTask+MSRTask): pred = np.log(pred)
                if dataset in BinClassifDatasets: pred=logit(pred)##logit link
                if dataset in MultiClassifDatasets: pred=np.log(pred)##softmax (up to a constant that cancels)
                preds+=[pred]
        else:
            pickle.dump(opt_params, open(path_temp+'opt_params.p', 'wb'))
            cmd="python fit_pred_multiprocessing.py "+" "+str(ncores)+" "+dataset+" "+mod+" "+path_temp
            status = subprocess.call(cmd, shell=True)
            if not status == 0: raise ValueError("Error in CV score calculation for " + str(dataset) + " & " + str(mod))
            preds=pickle.load(open(path_temp+'preds.p', 'rb'))
    else:
        preds=None  
    ##Determine optimal negll / error rate
    if calcOptLoss:
        if dataset in BinClassifDatasets: 
            score = 1-accuracy_score(ytest, (preds[0]>0.).astype(float))
            score_noW = 1-accuracy_score(ytest, (preds[1]>0.).astype(float))
        if dataset in MultiClassifDatasets: 
            score = 1-accuracy_score(ytest, np.argmax(preds[0],axis=1))
            score_noW = 1-accuracy_score(ytest, np.argmax(preds[1],axis=1))
        if dataset in RegrDatasets:
            dummy_model=get_ktboost_model(mod="GG",dataset=dataset,max_depth=max_depth,
                                          n_estimators=n_estimators,seed=seed,yl=yl,yu=yu,gamma=gamma)
            dummy_model.set_params(learning_rate=0.01,n_estimators=1)
            dummy_model.fit(Xtrain, ytrain)
            score = dummy_model.loss_(ytest, preds[0])
            score_noW = dummy_model.loss_(ytest, preds[1])
    else:
         score=None
         score_noW=None
    ##Return results
    res=[dataset,mod,len(ytrain)]+opt_pars_score+[score]+opt_pars_score_noW+[score_noW]
    return(res)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


"""
Specify data sets in different simulation runs
"""
PoissonTask=["poisson","insurance","poisson_r"]
GammaTask=["gamma","gamma_r"]
TobitTask=["tobit","tobit_r"]
MSRTask=["msr_f3","msr_r","birthweight","malnutrition","head_circumf","bodyfat"]
RegrDatasets=PoissonTask+GammaTask+TobitTask+MSRTask
BinClassifDatasets=["bin_clasif","bin_clasif_fht1",
                 "ionosphere","sonar","adult","titanic","bank","cancer","ijcnn"]
MultiClassifDatasets=["multi_clasif","multi_clasif_fht","multi_clasif_cos",
                   "digits","car","glass","satimage","letter","smartphone","covtype","usps"]
Simulated=["bin_clasif","bin_clasif_fht1","multi_clasif","multi_clasif_fht",
           "gamma","gamma_r","poisson","poisson_r",
           "tobit","tobit_r","tobit_f1","msr_f3","msr_r"]
RealData=["ionosphere","sonar","adult","bank","cancer","ijcnn","digits",
          "car","glass","satimage","insurance","letter","covtype","usps","smartphone",
          "birthweight","malnutrition"]
AdditionalDepth=["bin_clasif","multi_clasif_fht","gamma_r","poisson_r","tobit_r","msr_r"]


"""
Specify parameters for experiments
"""
n_estimators=1000## Maximal number of boosting iterations
nreps={"ionosphere":100,"sonar":100,"glass":100,"cancer":100,"titanic":100,"car":20,
       "waveform":20,"digits":20,"satimage":20,"birthweight":100,"bodyfat":100,"head_circumf":20}
param_grid_non_newt = {'learning_rate': [1,0.1,0.01,0.001], 'min_samples_leaf': [1,5,25,100], 'min_weight_leaf': [1.]}
param_grid_newton = {'learning_rate': [1,0.1,0.01,0.001], 'min_samples_leaf': [1], 'min_weight_leaf': [1.,5.,25.,100.]}
seed=20
ncores=6##Number of parallel cores
gamma=10 ## Shape parameter of gamma model
mods=['GG','GN','NN','XG']## Algorithms considered, GG = gradient boosting, GN = hybrid gradient-Newton boosting, NN = Newton boosting, XG = XGBoost
nsim=5000# number of simulated samples
nreal=20000# maximal number of samples for real-world data
max_depth=5# maximal depth of trees


"""
Run experiments and compare methods
"""
## Choose datasets for which comparison should be made
which_data="real_data"# real world data
which_data="simulated"# simulated data
which_data="simulated_small"# simulated data with smaller sample size
which_data="depth1"# different tree depths
which_data="depth3"# different tree depths
which_data="depth8"# different tree depths
which_data="depth20"# different tree depths
which_data="small_test"# small toy run

if which_data=="real_data":
    DataSetsExperiment=RealData
if which_data=="simulated":
    DataSetsExperiment=Simulated
if which_data=="simulated_small":
    nsim=500
    DataSetsExperiment=Simulated 
if which_data[:5]=="depth":
    nsim=5000
    nreal=20000
    DataSetsExperiment=AdditionalDepth
    max_depth=int(which_data[5:])
if which_data=="small_test":
    nsim=500
    DataSetsExperiment=["bin_clasif","digits"]

colnames=["Iter",'Dataset','Method','n','n_est_score','LR_score','min_w_score','min_s_score','score',
          'n_est_score_noW','LR_score_noW','min_w_score_noW','min_s_score_noW','score_noW']
## 'noW' refers to the situation where then minimum number of (weighted) samples per leaf parameter is not tuned and set to 1
results=pd.DataFrame(columns=colnames)
for dataset in DataSetsExperiment:
    n = nreal if dataset in RealData else nsim
    if which_data=="small_test":
        nrep = 5
        n = 500
    elif dataset in nreps:
        nrep = nreps[dataset]
    elif which_data=="simulated_small":
        nrep = 100
    else:
        nrep=10
    ensure_dir(path_temp)
    for indrep in range(0,nrep):
        for mod in mods:
            if (mod=='XG') & (dataset in TobitTask+MSRTask): continue
            yl,yu=return_limits_tobit(dataset)
            Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest=load_simulate_data(dataset=dataset,n=n,seed=(10*indrep+200),yl=yl,yu=yu,gamma=gamma,path=path_data,npart=3)
            param_grid = copy.deepcopy(param_grid_newton) if mod in ['NN','XG'] else copy.deepcopy(param_grid_non_newt)
            if dataset in MSRTask:##Minimum number of samples per leaf needs to be at least 10 for mean-scale regression (to avoid identifiability problems)
                if len(param_grid['min_samples_leaf'])>1: 
                    param_grid['min_samples_leaf'] = [ms for ms in param_grid['min_samples_leaf'] if not ms in [1,5]]
            param_grid['min_weight_leaf']=list(np.unique(np.minimum(param_grid['min_weight_leaf'],float(int(len(ytrain))))))
            param_grid['min_samples_leaf']=list(np.unique(np.minimum(param_grid['min_samples_leaf'],int(len(ytrain)))))
            print("\n *** "+dataset+" & "+mod+": run number "+ str(indrep+1) + " out of "+str(nrep) +" ***")
            res=tune_pars_calc_test_error(mod=mod,max_depth=max_depth,n_estimators=n_estimators,n=n,
                                               seed=seed,param_grid=param_grid,dataset=dataset,path_temp=path_temp,path_code=path_code,
                                               Xtrain=Xtrain,Xtest=Xtest,Xvalid=Xvalid,ytrain=ytrain,ytest=ytest,
                                               yvalid=yvalid,ncores=ncores,yl=yl,yu=yu,gamma=gamma,calcOptLoss=True)
            addrow=pd.DataFrame([[indrep+1]+res],columns=colnames)
            results=results.append(addrow)
    method_new_names = {'GG':'Gradient','GN':'Gradient-Newton','NN':'Newton','XG':'XGBoost'}
    for m in method_new_names:
        results.loc[results['Method']==m,'Method'] = method_new_names[m]
    results.to_csv(path_results+"results_detailed_simulation="+which_data+".csv",index=False)
    ## Aggregate results
    results_summary = results[['Dataset','Method','score']].groupby(['Dataset','Method']).agg(['mean','std'])
    results_summary.columns = ['mean', 'std']
    results_summary.to_csv(path_results+"results_summary_simulation="+which_data+".csv",index=True)
