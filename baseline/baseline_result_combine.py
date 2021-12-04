import os
import pathlib

folder_list =  os.listdir('./baseline_output')
folder_list.sort()
repeat_time = 10
n = len(folder_list)
start = 0

if not os.path.exists('./baseline_result'):
        pathlib.Path('./baseline_result').mkdir(parents=True, exist_ok=True)

while start<n:
    
    end = start + repeat_time
    
    kmeans_dtw_ARI_list = []
    kmeans_euclidean_ARI_list = []
    dbscan_dtw_ARI_list = []
    optics_dtw_ARI_list = []
    varlingam_AUC_list = []
    pcmci_AUC_list = []
    dynotears_AUC_list = []
    for filename in folder_list[start:end]:
        print(filename)
        with open(os.path.join('./baseline_output',filename,'training.log'), 'r') as fp:
            lines = fp.readlines()

            kmeans_dtw_ARI = float(lines[-9].split(':')[-1].strip())
            kmeans_dtw_ARI_list.append(kmeans_dtw_ARI)
            
            kmeans_euclidean_ARI = float(lines[-8].split(':')[-1].strip())
            kmeans_euclidean_ARI_list.append(kmeans_euclidean_ARI)

            dbscan_dtw_ARI = float(lines[-7].split(':')[-1].strip())
            dbscan_dtw_ARI_list.append(dbscan_dtw_ARI)

            optics_dtw_ARI = float(lines[-6].split(':')[-1].strip())
            optics_dtw_ARI_list.append(optics_dtw_ARI)
            
            varlingam_AUC = float(lines[-5].split(':')[-1].strip())
            varlingam_AUC_list.append(varlingam_AUC)

            pcmci_AUC = float(lines[-4].split(':')[-1].strip())
            pcmci_AUC_list.append(pcmci_AUC)

            dynotears_AUC = float(lines[-1].split(':')[-1].strip())
            dynotears_AUC_list.append(dynotears_AUC)

    with open(os.path.join('./baseline_result',filename+'_result.txt'),'w') as fp:

        fp.write(str(kmeans_dtw_ARI_list)+'\n')
        Average_ARI = sum(kmeans_dtw_ARI_list)/len(kmeans_dtw_ARI_list)
        fp.write(str(Average_ARI)+'\n')

        fp.write(str(kmeans_euclidean_ARI_list)+'\n')
        Average_ARI = sum(kmeans_euclidean_ARI_list)/len(kmeans_euclidean_ARI_list)
        fp.write(str(Average_ARI)+'\n')

        fp.write(str(dbscan_dtw_ARI_list)+'\n')
        Average_ARI = sum(dbscan_dtw_ARI_list)/len(dbscan_dtw_ARI_list)
        fp.write(str(Average_ARI)+'\n')

        fp.write(str(optics_dtw_ARI_list)+'\n')
        Average_ARI = sum(optics_dtw_ARI_list)/len(optics_dtw_ARI_list)
        fp.write(str(Average_ARI)+'\n')

        fp.write(str(varlingam_AUC_list)+'\n')
        Average_AUC = sum(varlingam_AUC_list)/len(varlingam_AUC_list)
        fp.write(str(Average_AUC)+'\n')

        fp.write(str(pcmci_AUC_list)+'\n')
        Average_AUC = sum(pcmci_AUC_list)/len(pcmci_AUC_list)
        fp.write(str(Average_AUC)+'\n')

        fp.write(str(dynotears_AUC_list)+'\n')
        Average_AUC = sum(dynotears_AUC_list)/len(dynotears_AUC_list)
        fp.write(str(Average_AUC)+'\n')
    
    start = end


