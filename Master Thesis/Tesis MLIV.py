# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:23:23 2024

@author: eduar
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from econml.iv.dml import OrthoIV
from econml.iv.dml import DMLIV
from econml.iv.dr import LinearDRIV
from econml.iv.dr import ForestDRIV
from econml.iv.dr import DRIV
from shap.plots import beeswarm
from shap import summary_plot


 

df =  pd.read_stata("C:/Users/eduar/OneDrive/Escritorio/Tesis Masterado/Stata Datasets/combined.dta")
print(df.dtypes)

df_covariates = df[["i_age_bfsr","p_age_bfsr","contraceptive","highlands_region","coastal_region","amazon_region","afro_ecuadorian","indigenous","mestizo","white","urban"]]

y = df["years_educ"]
z = df["menarchy"]
t = df["teen_pregnancy"]

def split_dataset(dataset, selected_variables):
    covariates = dataset[selected_variables]
    remaining_variables = [var for var in dataset.columns if var not in selected_variables]
    w = dataset[remaining_variables]
    return covariates, w

covariates, w = split_dataset(df_covariates, ["i_age_bfsr"])

########################################OrthoIV all linear regressions as in 2SLS###################################
ortho_iv_linear = OrthoIV(projection=False, discrete_treatment=True, discrete_instrument=False,random_state=(180397), 
                   model_y_xw = LinearRegression(), model_t_xw = LinearRegression(), model_z_xw = LinearRegression(), mc_iters=(1))

##escenario sin controles y sin cate
ortho_iv_linear.fit(Y = y,T = t,Z = z)
ortho_iv_linear.summary(alpha=0.05)
ortho_iv_linear.ate(X = None, T0 = 0, T1 =1)

##escenario con controles y con cate
ortho_iv_linear.fit(Y = y,T = t,Z = z, X=df_covariates)
ortho_iv_linear.summary(alpha=0.05)
ortho_iv_linear.ate(X = df_covariates, T0 = 0, T1 =1)

te_predict = ortho_iv_linear.effect(X = df_covariates,T0 = 0, T1 = 1)
te_interval = ortho_iv_linear.effect_interval(X = df_covariates, T0 = 0, T1 =1)


########################################OrthoIV linear regressions and Logit treatment ###################################
ortho_iv_logit = OrthoIV(projection=False, discrete_treatment=True, discrete_instrument=False,random_state=(180397), 
                   model_y_xw = LinearRegression(), model_t_xw = LogisticRegression(solver = "newton-cg"), model_z_xw = LinearRegression())

##escenario sin controles y sin cate
ortho_iv_logit.fit(Y = y,T = t,Z = z, X = None, W = None)
ortho_iv_logit.summary(alpha=0.05)
ortho_iv_logit.ate(X = None, T0 = 0, T1 =1)

##escenario con controles y con cate
ortho_iv_logit.fit(Y = y,T = t,Z = z, X=df_covariates)
ortho_iv_logit.summary(alpha=0.05)
ortho_iv_logit.ate(X = df_covariates, T0 = 0, T1 =1)


########################################train_test_splits ##########################################################

y_x_train, y_x_test, y_train, y_test = train_test_split(df_covariates, y, test_size=0.2, random_state=42)
t_x_train, t_x_test, t_train, t_test = train_test_split(df_covariates, t, test_size=0.2, random_state=42)
z_x_train, z_x_test, z_train, z_test = train_test_split(df_covariates, z, test_size=0.2, random_state=42)


########################################OrthoIV Random Forests ########################################################

##model y_wx
param_random_forest = {'n_estimators': [int(x) for x in [100,300,500,700,1000] ],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in [10,30,50,70,100]],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True,False]}

rf_y_wx = RandomForestRegressor(random_state=(180397))
rf_y_wx_tune = RandomizedSearchCV(estimator = rf_y_wx, param_distributions= param_random_forest,n_iter = 10,cv = 3)
rf_y_wx_tune.fit(X = df_covariates,y = y)
rf_y_wx_tune.best_params_

rf_t_wx = RandomForestClassifier(random_state=(180397))
rf_t_wx_tune = RandomizedSearchCV(estimator = rf_t_wx, param_distributions= param_random_forest,n_iter = 10,cv = 3)
rf_t_wx_tune.fit(X = df_covariates,y = t)
rf_t_wx_tune.best_params_

rf_z_wx = RandomForestRegressor(random_state=(180397))
rf_z_wx_tune = RandomizedSearchCV(estimator = rf_z_wx, param_distributions= param_random_forest,n_iter = 10,cv = 3)
rf_z_wx_tune.fit(X = df_covariates,y = z)
rf_z_wx_tune.best_params_

rf_y_wx = RandomForestRegressor(n_estimators= 300, min_samples_split=5, min_samples_leaf= 4,max_features = 'sqrt',max_depth= 100,bootstrap= True)
rf_t_wx = RandomForestClassifier(n_estimators= 500, min_samples_split= 10, min_samples_leaf= 2,max_features = 'auto',max_depth= 10,bootstrap= True)
rf_z_wx = RandomForestRegressor(n_estimators= 100, min_samples_split= 2, min_samples_leaf= 2,max_features = 'sqrt',max_depth= 10,bootstrap= False)

ortho_iv_rf = OrthoIV(projection=False, discrete_treatment=True, discrete_instrument=False,random_state=(180397), 
                   model_y_xw = rf_y_wx, model_t_xw = rf_t_wx, model_z_xw = rf_z_wx,mc_iters = 1)

ortho_iv_rf.fit(Y = y,T = t,Z = z, X = df_covariates)
ortho_iv_rf.summary(alpha=0.05)
ortho_iv_rf.ate(X = df_covariates, T0 = 0, T1 =1)

########################################OrthoIV XGBOOST ########################################################
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
}

xgb_t_wx = XGBClassifier()
xgb_t_wx_tune = BayesSearchCV(estimator = xgb_t_wx,search_spaces= search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=8) 
xgb_t_wx_tune.best_estimator_


########################################DMLIV Random Forests ########################################################

dr_iv_linear = DRIV(discrete_treatment=True, discrete_instrument=False,random_state=(180397), 
                   model_y_xw = LinearRegression(), model_t_xw = LinearRegression(), model_z_xw= LinearRegression(), 
                   model_tz_xw = LinearRegression(), model_final = LinearRegression())

dr_iv_linear.fit(Y = y,T = t,Z = z,X = df_covariates,W = df_covariates)
dr_iv_linear.ate(X= df_covariates,T0 = 0,T1 =1)
shap_values = dr_iv_linear.shap_values(X = df_covariates)
summary_plot(shap_values['years_educ']['teen_pregnancy_1.0'])

df_covariates_test = pd.DataFrame({
    "i_age_bfsr" : np.linspace(10, 24, 100),
    "p_age_bfsr" : np.linspace(11, 60, 100),
    "contraceptive" : np.random.choice([0, 1], size=100, p=[1 - 0.4002 , 0.4002]),
    "highlands_region" : np.random.choice([0, 1], size=100, p=[1 - 0.3654 , 0.3654]),
    "coastal_region" : np.random.choice([0, 1], size=100, p=[1 - 0.3814 , 0.3814]),
    "amazon_region" : np.random.choice([0, 1], size=100, p=[1 - 0.2329  , 0.2329]),
    "afro_ecuadorian" : np.random.choice([0, 1], size=100, p=[1 - 0.0467  , 0.0467]),
    "indigenous" : np.random.choice([0, 1], size=100, p=[1 - 0.1369  , 0.1369]),
    "mestizo" : np.random.choice([0, 1], size=100, p=[1 - 0.7618  , 0.7618]),
    "white" : np.random.choice([0, 1], size=100, p=[1 - 0.0124  , 0.0124]),
    "urban" : np.random.choice([0, 1], size=100, p=[1 - 0.6051  , 0.6051])})
                        

te_predict = dr_iv_linear.effect(X = df_covariates,T0 = 0, T1 = 1)
te_interval = dr_iv_linear.effect_interval(X = df_covariates, T0 = 0, T1 =1)



linear_driv = LinearDRIV(discrete_instrument=False, discrete_treatment=True)
linear_driv.fit(Y = y,T = t,Z = z,X = covariates)
linear_driv.summary()
linear_driv.ate(X = covariates, T0 = 0, T1 =1)


forest_driv = ForestDRIV(discrete_instrument=False, discrete_treatment=True)
forest_driv.fit(Y = y,T = t,Z = z,X = covariates,W = df_covariates)
forest_driv.summary()



####################################################plots##########################################################
te_predict = ortho_iv.effect(X = np.linspace(10, 24, 100).reshape(-1, 1), T0 = 0, T1 =1)
te_interval = ortho_iv.effect_interval(X = np.linspace(10, 24, 100).reshape(-1, 1), T0 = 0, T1 =1)

plt.figure(figsize=(18, 12))
plt.plot(np.linspace(10, 24, 100).reshape(-1, 1), te_predict, label= "predicted_treatment_effect", alpha=0.6)
plt.fill_between(
    np.linspace(10, 24, 100).reshape(-1, 1).flatten(),
    te_interval[0],
    te_interval[1],
    alpha=0.2,
    label="95% Confidence Interval",
)
plt.xlabel("Age before first sexual relationship")
plt.ylabel("Treatment Effect")
plt.legend()
plt.show()

te_predict = ortho_iv.effect(X = np.array([0,1]).reshape(-1, 1), T0 = 0, T1 =1)
te_interval = ortho_iv.effect_interval(X = np.array([0,1]).reshape(-1, 1), T0 = 0, T1 =1)

plt.figure(figsize=(18, 12))
plt.errorbar(y = te_predict,x = np.array([0,1]).reshape(-1, 1), yerr=[te_predict - te_interval[0], te_interval[1] - te_predict],
                 color='blue', capsize=8, capthick=2)
plt.xlabel("Age before first sexual relationship")
plt.ylabel("Treatment Effect")
plt.legend()
plt.show()

