from BoradLearningSystem import *

class StackBLS(BLS):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2 ** -30, NumBlock=3, is_argmax=False):
        super().__init__()
        self.NumBlock = NumBlock

class StackBLSClassifier(StackBLS):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2 ** -30, NumBlock=3):
        super().__init__()

    def fit(self, train_x, train_y):
        self.block_list = []

        block_pred_list = []
        block_train_y = train_y
        
        for i in range(self.NumBlock):
            block_module = BLSClassifier(is_argmax=False)
            block_fit_pred = block_module.fit(train_x, block_train_y)
            block_pred_list.append(block_fit_pred)

            block_res = block_fit_pred - block_train_y
            block_train_y = block_res
            
            self.block_list.append(block_module)
        
        pred_res_sum = [sum(x) for x in zip(*block_pred_list)]
        predlabel = pred_res_sum.argmax(axis=1)

        # 预测标签解嵌套
        predlabel = [int(i) for j in predlabel for i in j]

        return predlabel


    def predict(self, test_x):
        if len(self.block_list) == 0:
            assert 'Please Train Model before Prediction'
        
        pred_test_list = []

        for trained_block in self.block_list:
            pred = trained_block.predict(test_x)
            pred_test_list.append(pred)

        pred_res_sum = [sum(x) for x in zip(*pred_test_list)]
        predlabel = pred_res_sum.argmax(axis=1)

        # 预测标签解嵌套
        predlabel = [int(i) for j in predlabel for i in j]

        return predlabel
