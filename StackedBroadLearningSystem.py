import numpy as np

from BoradLearningSystem import *

class StackBLS(BLS):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2 ** -30, NumBlock=3, is_argmax=False):
        super().__init__()
        self.NumBlock = NumBlock

class StackBLSClassifier(StackBLS):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2 ** -30, NumBlock=3):
        super().__init__()

    def fit(self, train_x, train_y, is_excel_label=False):

        if is_excel_label:
            train_y = [[i] for i in train_y]
            encoder = preprocessing.OneHotEncoder()
            encoder.fit(train_y)
            train_y = encoder.transform(train_y).toarray()

        self.block_list = []

        block_pred_list = []
        block_train_y = train_y
        
        for i in range(self.NumBlock):
            block_module = BLSClassifier(is_argmax=False)
            block_fit_pred = block_module.fit(train_x, block_train_y, is_excel_label=False)
            # print(block_fit_pred)
            block_pred_list.append(block_fit_pred)

            block_res = block_fit_pred - block_train_y
            block_train_y = block_res
            
            self.block_list.append(block_module)

        pred_res_sum = np.array(block_pred_list).sum(axis=0)
        predlabel = pred_res_sum.argmax(axis=1)


        return predlabel


    def predict(self, test_x):
        if len(self.block_list) == 0:
            assert 'Please Train Model before Prediction'
        
        pred_test_list = []

        for trained_block in self.block_list:
            pred = trained_block.predict(test_x)
            pred_test_list.append(pred)

        pred_res_sum = np.array(pred_test_list).sum(axis=0)
        predlabel = pred_res_sum.argmax(axis=1)

        return predlabel
