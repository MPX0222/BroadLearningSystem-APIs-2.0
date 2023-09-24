# BroadLearningSystemTools-2.0



> ***New Toolbox of Broad Learning System, with sklearn liked APIs. Created at 2023.08***

---

### 💬 Surveys of Broad Learning System (Updated at 2023.09.23):

* ***《A survey of current Broad learning Models》- (English Version, South China University of Technology)*** : https://ieeexplore.ieee.org/abstract/document/9380770/

* ***《宽度学习研究进展》-（中文版，华南理工大学）*** : http://www.cnki.com.cn/Article/CJFDTotal-JSYJ202108003.htm

---

### 💭 API Models (Updated at 2023.09.23):

* ***BLS（South China University of Technology, 2018）*** : Based Broad Learning System, including ***BLSRegressor*** and ***BLSClassifier***. API `fit` and `predict` is available now. `GridSearch` will online soon. Paper Link: https://ieeexplore.ieee.org/abstract/document/7987745/

* ***Stacked-BLS（South China University of Technology, 2022）*** : A Stacked Model of Broad Learning System, with several residual learning block, including ***StackedBLSClassifier***. API `fit` and `predict` is available now. Paper Link: https://ieeexplore.ieee.org/abstract/document/9308927/

* ***BLS-AutoEncoder（South China University of Technology, 2023）*** : A novel Auto-Encode structure based on BLS, which is used to extract the feacture of original data. These APIs contains ***BLSAEExtractor*** and ***StackedBLSAEExtractor***. API `fit` is available now, whose output is feature vectors, not prediction. Please check the usage of APIs in `test_demo.py` Paper Link: https://ieeexplore.ieee.org/abstract/document/9661311

* ***BLS-LRF*** : An ensemble model based on a simple convoloution module and a Broad Learning System module, which enhance the ability of feature extraction for original BLS. API of this model would be upload sonn.

* ***Deep-Broad Ensemble Model（JiNan University, 2023）*** : A novel model based on 3D-Residual Conv Module and Broad Learning System. Upload soon. Paper Link: https://www.frontiersin.org/articles/10.3389/fnins.2023.1137557/full

* ***Tree-BLS（Beijing University of Technology, 2022）*** : A novel BLS model for enhancing the efficiency of small data modeling with various dimensions, whose original mapping neurons is replaced by tree modules. Paper Link: https://ieeexplore.ieee.org/abstract/document/9938406

