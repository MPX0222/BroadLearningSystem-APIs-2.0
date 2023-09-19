from BoradLearningSystem import *

class BLSAutoEncoder(BLS):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2 ** -30):
        super().__init__()


class BLSAEExtractor(BLSAutoEncoder):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2 ** -30):
        super().__init__()

    def fit(self, train_x):
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
            Beta_EachWindow = self.sparse_bls(Feature_EachWindowAfterPreprocess, Feature_InputDataWithBias).T  # 随机化特征映射初始偏置
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
        Output_EnhanceLayer = self.relu(Temp_Output_EnhanceLayer * self.Parameter_Shrink)  # 计算增强层最终输出

        # 输出层
        Input_OutputLayer = np.hstack([Output_FeatureMappingLayer, Output_EnhanceLayer])  # 合并特征层和增强层作为输出层输入
        Pinv_Output = self.cls_pinv(Input_OutputLayer)  # 计算伪逆

        # 计算系统总权重
        self.W = np.dot(Pinv_Output, train_x)

        OutputFeatureX = np.dot(train_x, self.W.T)  # 计算预测输出

        return OutputFeatureX
    

class StackedBLSAEExtractor(BLSAutoEncoder):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2 ** -30, NumBlock=3, is_multi_feature=False):
        super().__init__()
        self.NumBlock = NumBlock
        self.is_multi_feature = is_multi_feature

        
    def fit(self, train_x):
        self.block_list = []
        block_feature_dict = {}

        original_feature = train_x
        
        for i in range(self.NumBlock):
            block_module = BLSAEExtractor()
            block_fit_feature = block_module.fit(original_feature)
            block_feature_dict['Block' + str(i+1)] = np.array(block_fit_feature)

            original_feature = np.array(block_fit_feature)
            
            self.block_list.append(block_module)


        if self.is_multi_feature:
            return block_feature_dict
        else:
            return block_fit_feature