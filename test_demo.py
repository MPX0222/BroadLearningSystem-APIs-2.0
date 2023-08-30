from BoradLearningSystem import *
from sklearn.datasets import load_iris, load_breast_cancer

regressor = BLSRegressor(NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30)
classifier = BLSClassifier()

# dataFile = 'code/mnist.mat'
# data = scio.loadmat(dataFile)
# traindata = np.double(data['train_x']/255)
# trainlabel = np.double(data['train_y'])
# testdata = np.double(data['test_x']/255)
# testlabel = np.double(data['test_y'])

# dataset = load_iris()
# data, label = dataset['data'], dataset['target']
# print(label)

cls_dataset = load_breast_cancer()
cls_data, cls_label = cls_dataset['data'], cls_dataset['target']
print(cls_data.shape, cls_label.shape)
print(cls_label)


# train_output = regressor.fit(data, label)
# print(train_output.shape)
# predict_output = regressor.predict(data)
# print(predict_output.shape)

train_output = classifier.fit(cls_data, cls_label, is_excel_label=True)
# print(train_output)
predict_output = classifier.predict(cls_data)
print(predict_output)

