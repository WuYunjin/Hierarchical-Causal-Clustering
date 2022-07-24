# Collective Causal Discovery Algorithm for Multivariate Time Series

面向多元时间序列的群体因果关系发现算法,同时也是一种基于因果结构的多元时间序列聚类算法

代码实现基于 pytorch 1.71

paper link: http://www.ecice06.com/CN/10.19678/j.issn.1000-3428.0063674


General setting: number of variables is 5; Erdos-Renyi model with parameter 0.3; lag is 1.

Run a demo of Our model :
```
python main.py
```

and the baseline methods :
```
python baseline/baseline_main.py
```

# Reproduce synthetic experiments:

Our model :

```
nohup bash run.sh &
```
the results will be stored in 'output' folder and to summarize the results with all settings :

```
python result_combine.py
```
the results under different settings will be stored in 'result' folder;


And the baseline methods :

```
nohup bash baseline/baseline_run.sh &
```
the results will be stored in 'baseline/baseline_output' folder,
and to summarize the results with all settings :

```
cd baseline
python baseline_result_combine.py
```
the results under different settings will be stored in 'baseline_result' folder.



