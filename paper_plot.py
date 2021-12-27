import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def single_plot(y,xticklabels,xlim,ylim,x_label,y_label,method_name,title,y_locator=0.05):
    x = np.array( [i+1 for i in range(len(xticklabels))] )
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False  

    fig, ax = plt.subplots(ncols=1,dpi=800,figsize=(24, 16))
    # ax.set_title(title,fontsize=60, fontweight='bold')

    if len(method_name)==4:
        marker = ['^','s','P','o']
        color = ['darkgreen','royalblue','goldenrod','brown']
    elif len(method_name)==5:
        marker = ['^','s','P','p','o']
        color = ['chocolate','olivedrab','cornflowerblue','lightpink','brown']

    for i in range(len(method_name)):
        ax.plot(x, y[i], marker=marker[i], color=color[i], label=method_name[i], linewidth=10,markersize=40)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=60, fontweight='bold')  # 默认字体大小为60
    ax.set_xlabel(x_label, fontsize=60, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=60, fontweight='bold')
    ax.set_xlim(xlim[0], xlim[1])  # 设置x轴的范围
    if y_locator is not None:
        y_major_locator=MultipleLocator(y_locator) # y轴间距
        ax.yaxis.set_major_locator(y_major_locator)
    ax.set_ylim(ymin=ylim[0],ymax=ylim[1])
    plt.setp(ax.get_yticklabels(),fontsize=60, fontweight='bold')

    ax.legend(loc='lower left',fontsize=30,markerscale=0.7)

    plt.savefig('./result/{}.png'.format(title), format='png',bbox_inches='tight')  #bbox_inches用来去除白边

if __name__ == "__main__":

    
    y = np.array([  [0.49409675681175036, 0.5199564281769858, 0.6558294490127149, 0.6686952066948875, 0.671700556536303],
                    [0.4967199369208644, 0.537178365663922, 0.5691707605688603, 0.5195583216723765, 0.5848225390166615],
                    [0.7221441590450085, 0.4298819775976227, 0.2559012190751585, 0.13806911006929073, 0.13586276048536167],
                    [0.4799555802332038, 0.4, 0.4, 0.4, 0.4],
                    [0.8523896976946086, 0.8625460707117908, 0.82357158721918, 0.8509534066893127, 0.6164645286734077]
                ])
    xticklabels = [20,30,40,50,60] # x轴刻度的标识
    xlim = [0.5, 5.5]
    ylim = [-0.15, 1.04]
    x_label = "个体总数"
    y_label = "ARI"
    method_name = ['KMeans(DTW)','KMeans(Euclidean)','DBSCAN(DTW)','OPTICS(DTW)','LEAD']
    title = 'ARI under different number of subject'
    single_plot(y,xticklabels,xlim,ylim,x_label,y_label,method_name,title,y_locator=None)

    y = np.array([  [0.7029186658129796, 0.7117621776335049, 0.7090471632319673, 0.7079569968402114, 0.7064328846053107],
                    [0.6931678191085557, 0.7039689361199712, 0.7059329460229062, 0.704802018137374, 0.7040994774337656],
                    [0.7489609816717127, 0.7531035236320476, 0.7524392767927226, 0.7517198468703731, 0.7532379425688699],
                    [0.9405270224119487, 0.9519744018835412, 0.9663630009836229, 0.9581657750730708, 0.9569919740377058]
                ])
    xticklabels = [20,30,40,50,60] # x轴刻度的标识
    xlim = [0.5, 5.5]
    ylim = [0.46, 1.01]
    x_label = "个体总数"
    y_label = "AUC"
    method_name = ['VAR-LiNGAM','PCMCI','DYNOTEARS','LEAD']
    title = 'AUC under different number of subject'
    single_plot(y,xticklabels,xlim,ylim,x_label,y_label,method_name,title,y_locator=0.1)



# ##########################################################################################################


    y = np.array([  [0.46313330051706353, 0.5199564281769858, 0.4607012123565597, 0.5847723809632459, 0.533226145405623],
                    [0.5312733295428936, 0.537178365663922, 0.46932220516431455, 0.48215428785289455, 0.3524037856920529],
                    [0.36236148716086675, 0.4298819775976227, 0.4986499647452571, 0.5810731925059818, 0.7347750087610425],
                    [0.38665291223615295, 0.4, 0.7, 0.7, 0.7],
                    [0.7814060714198484, 0.7856918680573538, 0.7991328956749133, 0.8707499139644043, 0.902823347177639]
                ])
    xticklabels = [40,60,80,100,120] # x轴刻度的标识
    xlim = [0.5, 5.5]
    ylim = [-0.15, 1.04]
    x_label = "时间序列长度"
    y_label = "ARI"
    method_name = ['KMeans(DTW)','KMeans(Euclidean)','DBSCAN(DTW)','OPTICS(DTW)','LEAD']
    title = 'ARI under different sample size'
    single_plot(y,xticklabels,xlim,ylim,x_label,y_label,method_name,title,y_locator=None)


    y = np.array([  [0.6594104388426849, 0.7117621776335049, 0.7554652608588655, 0.7935154149877348, 0.8227014597430153],
                    [0.640934964717537, 0.7039689361199712, 0.7455002544623504, 0.7820571736153912, 0.8059512144716485],
                    [0.7307859635301779, 0.7531035236320476, 0.7646648126355798, 0.7693766172856785, 0.7752032837218032],
                    [0.9221128468905622, 0.9498450228829911, 0.9672935077962729, 0.9569455015780527, 0.9699301122602739]
                ])
    xticklabels = [40,60,80,100,120] # x轴刻度的标识
    xlim = [0.5, 5.5]
    ylim = [0.46, 1.01]
    x_label = "时间序列长度"
    y_label = "AUC"
    method_name = ['VAR-LiNGAM','PCMCI','DYNOTEARS','LEAD']
    title = 'AUC under different sample size'
    single_plot(y,xticklabels,xlim,ylim,x_label,y_label,method_name,title,y_locator=0.1)

    

##########################################################################################################

    y = np.array([  [0.7674015622223226, 0.739081628686151, 0.4793195041260344, 0.37852460089991724, 0.40281804073932503],
                    [0.685888343790661, 0.6711725373671856, 0.3924498081805206, 0.3126531360240097, 0.34681657419074985],
                    [0.12695827716445907, 0.09766999610453617, 0.08184253848359974, 0.04122260109595616, 0.017223807199969972],
                    [0.4, 0.4688888888888889, 0.33127129902385494, 0.17799331698042106, 0.2109093975174466],
                    [0.8208678879474365, 0.6675580339538112, 0.5405626481011051, 0.26851700847280535, 0.21451023569672456]
                ])
    xticklabels = [2,3,4,5,6] # x轴刻度的标识
    xlim = [0.5, 5.5]
    ylim = [-0.15, 1.04]
    x_label = "群体个数"
    y_label = "ARI"
    method_name = ['KMeans(DTW)','KMeans(Euclidean)','DBSCAN(DTW)','OPTICS(DTW)','LEAD']
    title = 'ARI under different number of group'
    single_plot(y,xticklabels,xlim,ylim,x_label,y_label,method_name,title,y_locator=None)


    y = np.array([  [0.702216531826926, 0.7026592140085864, 0.6962076582394048, 0.6992973645756935, 0.7068581592588294],
                    [0.7008482779319951, 0.6922203330710065, 0.6886288926508828, 0.6892954132863732, 0.6939546694115108],
                    [0.7622635879992834, 0.7491553503800666, 0.7429257020837033, 0.759145890897458, 0.7534420929914178],
                    [0.9652146718293736, 0.9227802139493824, 0.8793140657121554, 0.8260951724239762, 0.8049156960881276]
                ])
    xticklabels = [2,3,4,5,6] # x轴刻度的标识
    xlim = [0.5, 5.5]
    ylim = [0.46, 1.01]
    x_label = "群体个数"
    y_label = "AUC"
    method_name = ['VAR-LiNGAM','PCMCI','DYNOTEARS','LEAD']
    title = 'AUC under different number of group'
    single_plot(y,xticklabels,xlim,ylim,x_label,y_label,method_name,title,y_locator=0.1)

    

##########################################################################################################

    y = np.array([  [0.5589255607690399, 0.9264608599779492, 0.7488297992128368, 1.0, 0.9197988686360778],
                    [0.4720811524753544, 0.7892417880312559, 0.7246344680870701, 0.9041739130434783, 0.9056066033328142],
                    [0.6082798201590951, 0.9557619299023784, 0.9121878623074796, 0.9027932960893855, 0.986652912236153],
                    [0.586652912236153, 0.8, 0.8, 0.9, 1.0],
                    [0.6507643422980605, 0.8585523329746765, 0.88082588366062, 0.9070331986929654, 0.9768745360059391]
                ])
    xticklabels = [6,8,10,12,14] # x轴刻度的标识
    xlim = [0.5, 5.5]
    ylim = [-0.15, 1.04]
    x_label = "变量个数"
    y_label = "ARI"
    method_name = ['KMeans(DTW)','KMeans(Euclidean)','DBSCAN(DTW)','OPTICS(DTW)','LEAD']
    title = 'ARI under different number of variable'
    single_plot(y,xticklabels,xlim,ylim,x_label,y_label,method_name,title,y_locator=None)


    y = np.array([  [0.6858046667369443, 0.6653218940797004, 0.6638100695595848, 0.6579921499808492, 0.639538902971698],
                    [0.6981163704627436, 0.6804735510549085, 0.6829091506811649, 0.6949591319602224, 0.686516971161947],
                    [0.7585961451722308, 0.7285522309645975, 0.7275198932420442, 0.7064445974348469, 0.6810092755942552],
                    [0.9369069023465484, 0.9502062302595207, 0.9299727151610488, 0.9139594630031105, 0.8875317245081709]
                ])
    xticklabels = [6,8,10,12,14] # x轴刻度的标识
    xlim = [0.5, 5.5]
    ylim = [0.46, 1.01]
    x_label = "变量个数"
    y_label = "AUC"
    method_name = ['VAR-LiNGAM','PCMCI','DYNOTEARS','LEAD']
    title = 'AUC under different number of variable'
    single_plot(y,xticklabels,xlim,ylim,x_label,y_label,method_name,title,y_locator=0.1)