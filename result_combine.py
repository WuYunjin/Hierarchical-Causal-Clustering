import os
import pathlib

folder_list =  os.listdir('./output')
folder_list.sort()
repeat_time = 10
n = len(folder_list)
start = 0

if not os.path.exists('./result'):
        pathlib.Path('./result').mkdir(parents=True, exist_ok=True)

while start<n:
    
    end = start + repeat_time
    
    AUC_list = []
    ARI_list = []
    for filename in folder_list[start:end]:
        print(filename)
        with open(os.path.join('./output',filename,'training.log'), 'r') as fp:
            lines = fp.readlines()
            AUC = float(lines[-2].split(':')[-1].strip())
            AUC_list.append(AUC)
            ARI = float(lines[-3].split(':')[-1].strip())
            ARI_list.append(ARI)

    with open(os.path.join('./result',filename+'_result.txt'),'w') as fp:
        fp.write(str(AUC_list)+'\n')
        Average_AUC = sum(AUC_list)/len(AUC_list)
        fp.write(str(Average_AUC)+'\n')

        fp.write(str(ARI_list)+'\n')
        Average_ARI = sum(ARI_list)/len(ARI_list)
        fp.write(str(Average_ARI)+'\n')

    start = end


