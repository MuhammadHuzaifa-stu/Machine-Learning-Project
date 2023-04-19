import cv2
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

path = glob.glob("C:/Users/Muhammad Huzaifa/Desktop/Machin_Learning/cats/*.JPG")
path_d = glob.glob("C:/Users/Muhammad Huzaifa/Desktop/Machin_Learning/dogs/*.JPG")
path_h = glob.glob("C:/Users/Muhammad Huzaifa/Desktop/Machin_Learning/horses/*.JPG")
L = [path, path_d, path_h]
#------------- HOG SIFT SURF ----------------
hog = cv2.HOGDescriptor()
sift = cv2.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

features = []
image_features = []
image_features_surf = []
for i in L:
    for file in i:
        img = cv2.imread(file)
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        # compute the HOG features for the grayscale image
        hog_features = hog.compute(gray)
        # append the HOG features to the features list
        features.append(hog_features)

        keypoints, descriptors = sift.detectAndCompute(gray, None)
        descriptors = np.mean(descriptors, axis=0)
        image_features.append(descriptors)
    
        keypoints, descriptors = surf.detectAndCompute(gray, None)
        descriptors = np.mean(descriptors, axis=0)
        image_features_surf.append(descriptors)

# convert the features list to a NumPy array
features = np.array(features)
labels = np.zeros((features.shape[0]))
labels[0:202] = 0
labels[202:404] = 1
labels[404:606] = 2
#----------------- FOR HOG ------------------
x_tr, x_te, y_tr, y_te = train_test_split(features, labels, test_size=0.3)
clf = RFC(n_estimators=100)
ann = MLP(hidden_layer_sizes=(500,), activation='logistic', max_iter=1000)

clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_te)
acc_clf = accuracy_score(y_te, y_pred)
print("Random Forest Classifier accuracy (HOG): {:.2f}%".format(acc_clf*100))
ann.fit(x_tr, y_tr)
y_ann_pred = ann.predict(x_te)
acc_ann = accuracy_score(y_te, y_ann_pred)
print("Multilayer Perceptron Classifier accuracy (HOG): {:.2f}%".format(acc_ann*100))

#----------------- FOR SIFT ------------------
x_tr, x_te, y_tr, y_te = train_test_split(image_features, labels, test_size=0.3)
clf = RFC(n_estimators=100)
ann = MLP(hidden_layer_sizes=(500,), activation='logistic', max_iter=1000)
clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_te)
acc_clf = accuracy_score(y_te, y_pred)
print("Random Forest Classifier accuracy (SIFT): {:.2f}%".format(acc_clf*100))
ann.fit(x_tr, y_tr)
y_ann_pred = ann.predict(x_te)
acc_ann = accuracy_score(y_te, y_ann_pred)
print("Multilayer Perceptron Classifier accuracy (SIFT): {:.2f}%".format(acc_ann*100))

#----------------- FOR SURF ------------------
x_tr, x_te, y_tr, y_te = train_test_split(image_features_surf, labels, test_size=0.3)
clf = RFC(n_estimators=100)
ann = MLP(hidden_layer_sizes=(500,), activation='logistic', max_iter=1000)
clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_te)
acc_clf = accuracy_score(y_te, y_pred)
print("Random Forest Classifier accuracy (SURF): {:.2f}%".format(acc_clf*100))
ann.fit(x_tr, y_tr)
y_ann_pred = ann.predict(x_te)
acc_ann = accuracy_score(y_te, y_ann_pred)
print("Multilayer Perceptron Classifier accuracy (SURF): {:.2f}%".format(acc_ann*100))
