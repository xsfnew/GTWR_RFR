import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from sklearn.model_selection import train_test_split


# 构建Transformer模型
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(Transformer, self).__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.transfor


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = torch.zeros(x.size(0), seq_len, self.hidden_dim)
        position = torch.arange(0, seq_len).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2) * -(math.log(10000.0) / self.hidden_dim))
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.to(x.device)
        x = x + pos_enc
        return x


def load_data(csvFile, fields, yField):
    print('导入数据：' + csvFile)
    df = pd.read_csv(csvFile, low_memory=False)

    ###################### 样本数据概况 ############################
    # print('前5行数据：\n', df.head(5))
    # # 数据描述统计
    # print('数据维度', '-' * 30, 'n', df.shape)
    # # print('数据类型', '-' * 30, '\n', df.dtypes)
    # print('数据描述', '-' * 30, '\n', df.describe())
    # print('缺失值个数', '-' * 30, '\n', df.isnull().sum())
    # print('查看字段信息：', )
    # df.info()

    # # 箱型图，查看特征分布情况
    # X_tmp = df.drop(columns=[yField])
    # X_tmp.plot.box()
    # plt.show()

    # # 柱状图，查看因变量样本分布是否均衡
    # sns.countplot(x=yField, data=df)
    # plt.xlabel(yField);
    # plt.ylabel('Number of occurrences');
    # plt.show()

    # 特征预处理
    # print("特征预处理之前 n", df)
    # from sklearn import preprocessing
    # min_max_scaler = preprocessing.MinMaxScaler()
    # for field in fields:
    #     df[field] = min_max_scaler.fit_transform(df[field].values.reshape(-1, 1))
    #     # print("特征预处理之后 n", df)

    ################# 标签与数据集 #######
    # 标签
    labels = np.array(df[yField])  # y
    # 特征标签
    dataset = df[fields].values  # X

    ################ 数据标准化 ####################
    # 数据标准化....z标准化0~1
    features = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)
    labels = labels.reshape((-1, 1))
    labels = (labels - labels.mean(axis=0)) / labels.std(axis=0)

    # features = dataset
    # labels = labels.reshape((-1, 1))

    # # 初始化特征的标准化器
    # ss = StandardScaler()
    # # 分别对数据的特征进行标准化处理
    # features = ss.fit_transform(dataset)
    # 转换为 np.array形式
    # features = np.array(dataset)

    return features, labels


def static_Result(y_test, y_predict_test):
    # 'Adjusted_R2'：1-((1-r2_score(y_test,y_predict))*(n-1))/(n-p-1)
    print("R Squared: ", round(metrics.r2_score(y_test, y_predict_test), 4))
    print("Explained Variance Score: ", round(metrics.explained_variance_score(y_test, y_predict_test), 4))
    print('Mean Absolute Error: ', round(metrics.mean_absolute_error(y_test, y_predict_test), 4))
    print('Mean Squared Error: ', round(metrics.mean_squared_error(y_test, y_predict_test), 4))
    print('Root Mean Squared Error: ', round(metrics.root_mean_squared_error(y_test, y_predict_test), 4))
    # RMSE的取值范围是0到正无穷大，数值越小表示模型的预测误差越小，模型的预测能力越强，预测效果越好。
    # 需要注意的是，RMSE对较大的偏差非常敏感，因此当数据中存在较大的异常值时，RMSE可能会受到较大的影响。
    # 计算拟合优度的另一种方法
    Regression = sum((y_predict_test - np.mean(y_test)) ** 2)  # 回归平方和
    Residual = sum((y_test - y_predict_test) ** 2)  # 残差平方和
    total = sum((y_test - np.mean(y_test)) ** 2)  # 总体平方和
    R_square = 1 - Residual / total
    print('R_square:', round(R_square[0], 4))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    print('\r', end='')

    data_dir = r'F:\Map\base'
    os.chdir(data_dir)

    yField = 'probability'
    fields = ['YU_BI_DU', 'PINGJUN_XJ',
              'N_V_GDP', 'N_V_POP', 'N_V_SLOP', 'N_V_ASPE', 'N_V_DEM', 'N_V_tcl',
              'N_V_pre', 'N_V_tmp', 'N_V_wind_',              
              'N_water_dist', 'N_road_dist'  

              # 'N_water_dist','N_road_dist','N_railways_d',
              #  'N_V_wet','N_V_dtr','N_V_pet','N_V_frs', # 'N_V_tmx','N_V_tmn',
              #  'N_V_lrad_','N_V_vap','N_V_srad_','N_V_shum_','N_V_prec_',#'N_V_temp_','N_V_pres_',
              #  'N_V_residence_density','N_V_NDVI'
              ]
    features, labels = load_data("modis_hn_Standardization.csv", fields, yField)

    # # 1.1 加载MNIST数据集,并预处理
    # data = np.load('mnist.npz')
    # xdata, ydata = data['x_train'], data['y_train']
    #
    # # 1.2 数据预处理:将图像数据reshape为(样本数, 通道数, 高度, 宽度)的形状
    # xdata = xdata.reshape(-1, 1, 28, 28)
    #
    # # 1.3 将NumPy数组转换为PyTorch张量,并指定数据类型
    # xdata = torch.tensor(xdata, dtype=torch.float32)
    # ydata = torch.tensor(ydata, dtype=torch.long)

    xdata = torch.tensor(features, dtype=torch.float32)
    ydata = torch.tensor(labels, dtype=torch.float32)

    #####################切分数据集， 训练集与测试集 #######
    # 2.1 使用train_test_split函数划分训练集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(xdata, ydata, test_size=0.2, random_state=42)

    # 2.2 创建数据加载器DataLoader,用于批量加载数据
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # # 准备训练数据
    # input_data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    # target_data = torch.tensor([[10, 20], [30, 40], [50, 60]], dtype=torch.float32)
    # dataset = TensorDataset(input_data, target_data)
    # dataloader = DataLoader(dataset, batch_size=1)

    # 定义模型参数
    input_dim = train_data.size(1)
    output_dim = train_labels.size(1)
    hidden_dim = 128
    num_layers = 2
    num_heads = 4
    # 创建模型和优化器
    model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 模型训练
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_input, batch_target in train_loader:
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # # 使用模型进行预测
    # new_input = torch.tensor([[2, 3, 4]], dtype=torch.float32)
    # predicted_output = model(new_input)
    # print("Predicted output:", predicted_output)
    test_preds = []
    test_true = []
    # 定义用于存储测试集预测值的列表
    with torch.no_grad():
        # 禁用梯度计算,以减少内存占用和加速计算
        for data, labels in test_loader:
            # 遍历测试数据加载器,获取每个批次的数据
            outputs = model(data)
            # 将数据输入模型,得到预测输出
            test_preds.extend(outputs.numpy())
            # 将预测输出转换为NumPy数组并添加到测试集预测值列表中
            test_true.extend(labels.numpy())  # 收集测试集的真实标签

    static_Result(np.array(test_true), np.array(test_preds))
