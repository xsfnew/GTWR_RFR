import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


def histogram(x):
    # Fit a normal distribution to the data:
    # mean and standard deviation
    mu, std = norm.fit(x)

    # Plot the histogram.
    plt.hist(x, bins=15, density=True, edgecolor='w', alpha=0.7, color='b')  # alpha=0.75, histtype='stepfilled'

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    plt.plot(x, p, '--', linewidth=1.5)  # label='f(x)'
    # title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title("WildFire probability")
    # plt.legend(loc='upper left')
    plt.show()


def classify(x, bins, labels, title):
    # 设置分段
    # bins=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]#
    # 设置标签
    # labels=['0.1','0.2','0.3','0.4','0.5','0.6']
    # 按分段离散化数据
    segments = pd.cut(x, bins, labels=labels)
    print('数据分段:')
    print(segments)

    plt.rcParams['font.family'] = 'simhei'
    plt.figure(figsize=(7, 7))
    plt.subplot(111)
    # 统计各分段数
    counts = pd.value_counts(segments, sort=False)
    # 绘制柱状图
    b = plt.bar(counts.index, counts)
    # 添加数据标签
    plt.bar_label(b, counts, fontsize=18)
    
    #设置标题标注和字体大小
    plt.title(title,fontsize=25)
    # plt.rcParams.update({"font.size":20})
    
    # #设置坐标标签标注和字体大小
    # plt.xlabel("step",fontsize=15)
    # plt.ylabel("rate",fontsize=15)

    #设置坐标刻度字体大小
    plt.xticks(fontsize=20)#,rotation=90
    plt.yticks(fontsize=20)
    
    # #设置图例字体大小和样式
    # plt.legend(loc="upper right",fontsize=15)
    
    plt.show()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    data_dir = r'F:\Map\base'
    os.chdir(data_dir)

    print('导入数据...')
    df = pd.read_csv(data_dir + r'\modis_hn_Standardization.csv',
                     float_precision='round_trip')  # Modis_Output.csv #Output.csv

    # x = [(out_day_by_date(dt.datetime.strptime(dn, '%Y-%m-%d')) - 1) for dn in df['road_dist'].values]
    x = df['probability'].values
    x = np.log(x)+5
    histogram(x)
    bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1]
    labels=['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
    classify(x,bins,labels,'Probability of wildfires')

    # fields = ['YU_BI_DU', 'PINGJUN_XJ', 'N_V_tcl',
              # 'N_V_GDP', 'N_V_POP', 'N_V_residence_density',
              # 'N_V_pre', 'N_V_tmp', 'N_V_wind_',
              # 'N_V_SLOP', 'N_V_ASPE', 'N_V_DEM',
              # 'N_water_dist', 'N_road_dist'  

              # # 'N_water_dist','N_road_dist','N_railways',
              # #  'N_V_wet','N_V_dtr','N_V_pet','N_V_frs', # 'N_V_tmx','N_V_tmn',
              # #  'N_V_lrad_','N_V_vap','N_V_srad_','N_V_shum_','N_V_prec_',#'N_V_temp_','N_V_pres_',
              # #  'N_V_NDVI',
              # ]

    # x = df['YU_BI_DU'].values
    # bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # labels = ['0','0.1', '0.2', '0.3', '0.4', '0.5', '0.6','0.7','0.8','0.9']
    # title="Canopy closure"
    # classify(x, bins, labels, title)

    # x = df['PINGJUN_XJ'].values
    # bins = [0,  4,  8, 12,  16,  20,  24,  28]
    # labels = ['0', '4', '8',  '12', '16', '20', '24']
    # title="Average volume of standing timber"
    # classify(x, bins, labels, title)

    # x = df['N_V_tcl'].values
    # bins = [0, 10, 20, 30, 40, 50, 60, 70]
    # labels = ['0', '10', '20', '30', '40', '50', '60']
    # title="TCL"
    # classify(x, bins, labels, title)

    # x = df['N_V_pre'].values
    # bins = [0, 50, 100, 150, 200, 250, 300, 350]
    # labels = ['0', '50', '100', '150', '200', '250', '300']
    # title="Precipitation"
    # classify(x, bins, labels, title)

    # x = df['N_V_tmp'].values
    # bins = [0,5,10, 15, 20, 25, 30, 35]
    # labels = ['0','5', '10', '15', '20', '25', '30']
    # title="Temperature"
    # classify(x, bins, labels, title)

    # x = df['N_V_wind_'].values
    # bins = [0,1, 2, 3, 4, 5, 6, 7]
    # labels = ['0','1', '2', '3', '4', '5', '6']
    # title="Wind"
    # classify(x, bins, labels, title)

    # x = df['N_V_DEM'].values   
    # bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600]
    # labels = ['0', '200', '400', '600', '800', '1000', '1200','1400']
    # title='Elevation'
    # classify(x, bins, labels, title)

    # x = df['N_V_SLOP'].values
    # bins = [0, 10, 20, 30, 40, 50, 60]
    # # labels=['平原','微斜坡','缓斜坡','斜坡','陡坡','峭坡']
    # labels = ['0', '10', '20', '30', '40', '50']
    # title='Slop'
    # classify(x, bins, labels, title)

    # x = df['N_V_ASPE'].values
    # bins = [0, 45, 90, 135, 180, 225, 270,315,360]
    # labels = ['0', '45', '90','135', '180', '225', '270','315']
    # title='Aspect'
    # classify(x, bins, labels, title)

    # x = df['N_V_water_dist'].values
    # bins = [0, 1000, 2000, 3000, 4000, 5000,6000, 7000]
    # labels = ['0', '1000', '2000', '3000', '4000', '5000', '6000']
    # title='Distance from water'
    # classify(x, bins, labels, title)

    # x = df['V_road_dist'].values
    # bins = [0,200, 400, 600, 800, 1000, 1200, 1400]
    # labels = ['0','200', '400', '600', '800', '1000', '1200']
    # title='Distance from road'
    # classify(x, bins, labels, title)

    # x = df['N_V_GDP'].values
    # bins = [0, 1000, 2000, 3000, 4000, 5000, 6000,7000]
    # labels = ['0','1000', '2000', '3000', '4000', '5000', '6000']
    # title='GDP'
    # classify(x, bins, labels, title)

    # x = df['N_V_POP'].values
    # bins = [0, 200, 400, 600, 800, 1000, 1200,1400]
    # labels = ['0','200', '400', '600', '800', '1000', '1200']
    # title='Population'
    # classify(x, bins, labels, title)

    # x = df['N_V_residence_density'].values     
    # bins = [-1,0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12,0.14]
    # labels = ['0','0.02', '0.04', '0.06', '0.08', '0.1', '0.12','0.14']
    # title='building density'
    # classify(x, bins, labels, title)