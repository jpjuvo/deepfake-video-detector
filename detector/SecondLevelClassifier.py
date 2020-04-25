import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from Util.Timer import Timer
from Util.FeatureStats import removeOutliers

class SecondLevelClassifier:
    
    def __init__(self, 
                 second_level_xgb_paths,
                 second_level_logreg_paths,
                 second_level_lgb_paths,
                 verbose=0):
        self.second_level_xgb_models = [self._load_xgb_model(xgb_path) for xgb_path in second_level_xgb_paths]
        self.second_level_logreg_models = [joblib.load(open(filename, 'rb')) for filename in second_level_logreg_paths]
        self.second_level_lgb_models = [lgb.Booster(model_file=filename) for filename in second_level_lgb_paths]
        self.verbose=verbose
        print("Loaded {0} second level xgb classifier models, {1} logistic regression models and {2} LightGBM models".format(len(self.second_level_xgb_models), 
                                                                                                                             len(self.second_level_logreg_models),
                                                                                                                             len(self.second_level_lgb_models)))

    def _load_xgb_model(self, path):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(path)
        return xgb_model

    def predict(self, feats, featureClassifiers=['xgb', 'logreg', 'lightgbm'], eps=1e-6):
        timer = Timer()
        assert len(self.second_level_xgb_models) > 0, 'Second level models are not initialized'
        assert len(featureClassifiers) > 0, 'No feature classifiers selected'

        preds_list = []

        preds_xgb = [model.predict_proba([feats])[0,1] for model in self.second_level_xgb_models]
        preds_xgb = list(np.clip(np.array(preds_xgb),eps,1-eps))
        if self.verbose>1:
            print('XGB predictions {0}'.format(preds_xgb))
        if 'xgb' in featureClassifiers:
            preds_list += preds_xgb

        preds_logred = [model.predict_proba([feats])[0,1] for model in self.second_level_logreg_models]
        preds_logred = list(np.clip(np.array(preds_logred),eps,1-eps))
        if self.verbose>1:
            print('Logistic reg. predictions {0}'.format(preds_logred))
        if 'logreg' in featureClassifiers:
            preds_list += preds_logred

        preds_lgb = [model.predict([feats])[0] for model in self.second_level_lgb_models]
        preds_lgb = list(np.clip(np.array(preds_lgb),eps,1-eps))
        if self.verbose>1:
            print('LightGBM predictions {0}'.format(preds_lgb))
        if 'lightgbm' in featureClassifiers:
            preds_list += preds_lgb

        # remove the most deviating 10%
        preds_list = removeOutliers(preds_list, remove_n=len(preds_list)//10)

        timer.print_elapsed(self.__class__.__name__, verbose=self.verbose)
        return np.mean(np.array(preds_list))
