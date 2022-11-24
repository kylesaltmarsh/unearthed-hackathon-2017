import AudioProc as AP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

class Classify:
    def __init__(self, classifer):
        if classifer == 'GBDT':
            self.clf = GradientBoostingClassifier()
        elif classifer == 'MLP':
            self.clf = MLPClassifier(solver='adam', hidden_layer_sizes=(6), random_state=1, max_iter=200)
        elif classifer == 'AutoML':
            self.clf = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30, tmp_folder="/tmp/autosklearn_classification_example_tmp")

    def process_audio(self, fname, window = 50, step = 50):
        # get the audio
        c = AP.InstrumentNoise()
        c.get_audio(fname, 0, 30)
        # Add features
        c.cal_ber(window, step, frel = [1210], freu = [1240],Enhance = False, AddFeature = True)
        c.cal_ber(window, step, frel = [100], freu = [400],Enhance = False, AddFeature = True)
        c.cal_ber(window, step, frel = [3000], freu = [8000],Enhance = False, AddFeature = True)
        c.cal_fcc(window, step, AddFeature = True)
        c.cal_flatness(window, step, AddFeature = True,Enhance = False)
        c.cal_hef(window, step, AddFeature = True)
        return c.Feature
    
    def train(self, features, label):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, label, test_size=0.3, random_state=0)
        self.clf.fit(self.X_train, self.y_train)

    def predict(self):
        print(self.clf.score(self.X_test, self.y_test))
        print(confusion_matrix(self.y_test, self.clf.predict(self.X_test)))
        # print(self.clf.show_models())
 
if __name__ == "__main__":
    classify = Classify('GBDT')
    filename = 'data/Corroded bearing Fin Fan 150 rpm.wav'
    feature_1 = classify.process_audio(filename)
    label_1 = np.repeat(0, len(feature_1))
    filename = 'data/Outer race bearing fault fin fan 150 rpm.wav'
    feature_2 = classify.process_audio(filename)
    label_2 = np.repeat(1, len(feature_1))
    filename = 'data/Health Fan.wav'
    feature_3 = classify.process_audio(filename)
    label_3 = np.repeat(2, len(feature_1))

    features = normalize(np.concatenate((feature_1, feature_2, feature_3), axis = 0))
    labels = np.concatenate((label_1, label_2, label_3), axis = 0)

    classify.train(features, labels)
    classify.predict()