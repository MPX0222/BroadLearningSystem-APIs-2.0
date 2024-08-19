import numpy as npinv
import pandas as pd

from scipy import linalg as LA
from scipy import io as scio
from numpy import random
from sklearn import preprocessing

class GridSearchCV:
    def __init__(self) -> None:
        pass

    def run():
        return None

class BLS:
    def __init__(self, NumFeatureNodes=10, NumWindows=100, NumEnhance=1000, S=0.5, C=2**-30, is_argmax=True):
        self.FeatureNodes = NumFeatureNodes
        self.FeatureWindows = NumWindows  
        self.EnhancementNodes = NumEnhance
        self.S = S
        self.C = C
        self.is_argmax = is_argmax

    def _tansig(self, x):
        return (2/(1+np.exp(-2*x)))-1
    
    def _relu(self, x):
        return np.maximum(0, x)

    def _pinv(self, A, reg):
        return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
    
    def _pinv_cls(self, matrix):
        return np.mat(self.C * np.eye(matrix.shape[1]) + matrix.T.dot(matrix)).I.dot(matrix.T)

    def _shrinkage(self, a, b):
        z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
        return z

    def _sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = np.dot(A.T, A)
        m = A.shape[1]
        n = b.shape[1]
        wk = np.zeros([m, n], dtype='double')
        ok = np.zeros([m, n], dtype='double')
        uk = np.zeros([m, n], dtype='double')
        L1 = np.mat(AA + np.eye(m)).I
        L2 = np.dot(np.dot(L1, A.T), b)
        for i in range(itrs):
            tempc = ok - uk
            ck = L2 + np.dot(L1, tempc)
            ok = self._shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

class BLSRegressor(BLS):
    def _init_(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30, is_argmax=True):
        super()._init_()

    def fit(self, train_x, train_y):
        u = 0
        WF = list()
        for i in range(self.FeatureWindows):
            random.seed(i+u)
            WeightFea = 2*random.randn(train_x.shape[1]+1, self.FeatureNodes)-1
            WF.append(WeightFea)
        self.WeightEnhan = 2*random.randn(self.FeatureWindows*self.FeatureNodes+1, self.EnhancementNodes)-1

        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
        y = np.zeros([train_x.shape[0], self.FeatureWindows*self.FeatureNodes])
        self.WFSparse = list()
        self.distOfMaxAndMin = np.zeros(self.FeatureWindows)
        self.meanOfEachWindow = np.zeros(self.FeatureWindows)
        for i in range(self.FeatureWindows):
            WeightFea = WF[i]
            A1 = H1.dot(WeightFea)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1 = scaler1.transform(A1)
            WeightFeaSparse = self._sparse_bls(A1, H1).T
            self.WFSparse.append(WeightFeaSparse)

            T1 = H1.dot(WeightFeaSparse)
            self.meanOfEachWindow[i] = T1.mean()
            self.distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            y[:, self.FeatureNodes * i:self.FeatureNodes * (i+1)] = T1

        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
        T2 = H2.dot(self.WeightEnhan)
        T2 = self._tansig(T2)
        T3 = np.hstack([y, T2])
        self.WeightTop = self._pinv(T3, self.C).dot(train_y)

        print(self.WeightTop.shape)
        NetoutTrain = T3.dot(self.WeightTop.T)

        return NetoutTrain

    def predict(self, test_x):
        # 预测输出
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
        yy1 = np.zeros([test_x.shape[0], self.FeatureWindows * self.FeatureNodes])
        for i in range(self.FeatureWindows):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            TT1 = (TT1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            yy1[:, self.FeatureNodes * i:self.FeatureNodes*(i+1)] = TT1
        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
        TT2 = self._tansig(HH2.dot(self.WeightEnhan))
        TT3 = np.hstack([yy1, TT2])
        NetoutTest = TT3.dot(self.WeightTop.T)

        return NetoutTest

class BLSClassifier(BLS):
    def _init_(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30, is_argmax=True):
        super()._init_()
        self.is_argmax = is_argmax
    def fit(self, train_x, train_y, is_excel_label=False):
        """模型本体"""

        if is_excel_label:
            train_y = [[i] for i in train_y]
            encoder = preprocessing.OneHotEncoder()
            encoder.fit(train_y)
            train_y = encoder.transform(train_y).toarray()

        # --Train--
        train_x = preprocessing.scale(train_x, axis=1)  # 标准化处理样本
        Feature_InputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])  # 将输入矩阵进行行链接，即平铺展开整个矩阵
        Output_FeatureMappingLayer = np.zeros([train_x.shape[0], self.FeatureWindows * self.FeatureNodes])

        self.Beta1_EachWindow = []
        self.Dist_MaxAndMin = []
        self.Min_EachWindow = []
        self.ymin = 0
        self.ymax = 1

        # 特征层
        for i in range(self.FeatureWindows):
            random.seed(i + 2022)
            W_EachWindow = 2 * random.randn(train_x.shape[1] + 1, self.FeatureNodes) - 1  # 随机化特征层初始权重
            Feature_EachWindow = np.dot(Feature_InputDataWithBias, W_EachWindow)  # 计算每个特征映射中间态

            # scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(Feature_EachWindow)                      # 对上述结果归一化处理
            # Feature_EachWindowAfterPreprocess = scaler1.transform(Feature_EachWindow)                               # 进行标准化

            Feature_EachWindowAfterPreprocess = Feature_EachWindow  # 进行标准化
            Beta_EachWindow = self._sparse_bls(Feature_EachWindowAfterPreprocess, Feature_InputDataWithBias).T  # 随机化特征映射初始偏置
            self.Beta1_EachWindow.append(Beta_EachWindow)
            Output_EachWindow = np.dot(Feature_InputDataWithBias, Beta_EachWindow)  # 计算每个特征映射最终输出

            self.Dist_MaxAndMin.append(np.max(Output_EachWindow, axis=0) - np.min(Output_EachWindow, axis=0))  # 计算损失函数
            self.Min_EachWindow.append(np.min(Output_EachWindow, axis=0))
            Output_EachWindow = (Output_EachWindow - self.Min_EachWindow[i]) / self.Dist_MaxAndMin[i]
            # 计算特征层最终输出
            Output_FeatureMappingLayer[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = Output_EachWindow

        # 增强层
        train_ori_hance = np.hstack([train_x])
        train_x_enhance = preprocessing.scale(train_ori_hance, axis=1)
        Input_EnhanceLayerWithBias = np.hstack([train_x_enhance, 0.1 * np.ones((train_x_enhance.shape[0], 1))])

        self.W_EnhanceLayer = LA.orth(2 * random.randn(train_x_enhance.shape[1] + 1, self.EnhancementNodes)) - 1

        Temp_Output_EnhanceLayer = np.dot(Input_EnhanceLayerWithBias, self.W_EnhanceLayer)  # 计算增强层中间态
        self.Parameter_Shrink = self.S / np.max(Temp_Output_EnhanceLayer)
        Output_EnhanceLayer = self._relu(Temp_Output_EnhanceLayer * self.Parameter_Shrink)  # 计算增强层最终输出

        # 输出层
        Input_OutputLayer = np.hstack([Output_FeatureMappingLayer, Output_EnhanceLayer])  # 合并特征层和增强层作为输出层输入
        _pinv_Output = self._pinv_cls(Input_OutputLayer)  # 计算伪逆

        # 计算系统总权重
        self.W = np.dot(_pinv_Output, train_y)

        OutputOfTrain = np.dot(Input_OutputLayer, self.W)  # 计算预测输出

        if self.is_argmax:
            predlabel = OutputOfTrain.argmax(axis=1)
            # print(predlabel)
            # 预测标签解嵌套
            predlabel = [int(i) for j in predlabel for i in j]
        else:
            predlabel = OutputOfTrain
        
        return predlabel
    
    
    def predict(self, test_x):
        test_x = preprocessing.scale(test_x, axis=1)
        Feature_InputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        Output_FeatureMappingLayerTest = np.zeros([test_x.shape[0], self.FeatureWindows * self.FeatureNodes])

        for i in range(self.FeatureWindows):
            Output_EachWindowTest = np.dot(Feature_InputDataWithBiasTest, self.Beta1_EachWindow[i])
            Output_FeatureMappingLayerTest[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = (self.ymax - self.ymin) * (Output_EachWindowTest - self.Min_EachWindow[i]) / self.Dist_MaxAndMin[i] - self.ymin

        test_ori_hance = np.hstack([test_x])
        test_x_enhance = test_ori_hance
        Input_EnhanceLayerWithBiasTest = np.hstack([test_x_enhance, 0.1 * np.ones((test_x_enhance.shape[0], 1))])
        Temp_Output_EnhanceLayerTest = np.dot(Input_EnhanceLayerWithBiasTest, self.W_EnhanceLayer)
        Output_EnhanceLayerTest = self._relu(Temp_Output_EnhanceLayerTest * self.Parameter_Shrink)
        Input_OutputLayerTest = np.hstack([Output_FeatureMappingLayerTest, Output_EnhanceLayerTest])  # 合并特征层和增强层作为测试输出层输入

        OutputOfTest = np.dot(Input_OutputLayerTest, self.W)  # 计算预测输出

        if self.is_argmax:
            predlabel = OutputOfTest.argmax(axis=1)
            # 预测标签解嵌套
            predlabel = [int(i) for j in predlabel for i in j]
        else:
            predlabel = OutputOfTest
        
        return predlabel
