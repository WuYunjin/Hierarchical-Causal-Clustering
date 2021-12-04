import logging
from pytz import timezone
from datetime import datetime
import numpy as np

import os,sys
sys.path.append( os.path.join('..',os.getcwd()))

from baseline_config import save_yaml_config, get_args


from data_loader.synthetic_dataset import SyntheticDataset
# from data_loader.real_dataset import RealDataset

from helpers.log_helper import LogHelper
from helpers.torch_utils import set_seed
from helpers.dir_utils import create_dir
from helpers.analyze_utils import  plot_timeseries, plot_recovered_graph, plot_ROC_curve, AUC_score

from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import adjusted_rand_score


def synthetic(seed, num_groups, num_subjects_per_group,  num_samples, num_variables, max_lag):
    
    np.set_printoptions(precision=3)
    
    # Setup for logging
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')

    _logger = logging.getLogger(__name__)

    
    # Save the configuration for logging purpose
    save_yaml_config(args, path='{}/config.yaml'.format(output_dir))
    

    # Reproducibility
    set_seed(seed)

    # Get dataset
    _logger.info('Generating dataset...')
    dataset = SyntheticDataset(num_groups, num_subjects_per_group,  num_samples, num_variables, max_lag)
    # Save dataset
    dataset.save_dataset( output_dir=output_dir)
    _logger.info('Finished generating dataset')

    # Look at data
    _logger.info('The shape of data (num_groups ,num_subjects, Ts, num_variables): {}, {}'.format(len(dataset.X),dataset.X[0].shape))
    for q in range(num_groups):
        plot_timeseries(dataset.X[q],'Group {}'.format(q),display_mode=False,save_name=output_dir+'/group{}_timeseries_X.png'.format(q+1))

    # Shuffle the order of subjects
    data = np.concatenate(dataset.X)
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    shuffled_cluster = np.concatenate([ [i]*num_subjects_per_group for i in range(num_groups) ]) [permutation]
    _logger.info('The groudtruth clusters: {}'.format(shuffled_cluster))

    return shuffled_data, shuffled_cluster , dataset.matrix

def kmeans_dtw(X,num_groups,cluster_labels):

    _logger = logging.getLogger(__name__)

    km = TimeSeriesKMeans(n_clusters=num_groups, metric="dtw")
    km_dtw_labels = km.fit_predict(X=X)

    ari=adjusted_rand_score(cluster_labels, km_dtw_labels) 
    # _logger.info('Using k-means(DTW) and results: {}'.format(km_dtw_labels)) 
    _logger.info('Using k-means(DTW) and Adjusted Rand index，ARI: {}'.format(ari)) 

    return ari,km_dtw_labels

def kmeans_euclidean(X,num_groups,cluster_labels):

    _logger = logging.getLogger(__name__)

    km = TimeSeriesKMeans(n_clusters=num_groups, metric="euclidean")
    km_euclidean_labels = km.fit_predict(X=X)
    
    ari=adjusted_rand_score(cluster_labels, km_euclidean_labels) 
    # _logger.info('Using k-means(euclidean) and results: {}'.format(km_euclidean_labels)) 
    _logger.info('Using k-means(euclidean) and Adjusted Rand index，ARI: {}'.format(ari)) 
    

    return ari,km_euclidean_labels


def DBSCAN_dtw(X, cluster_labels):

    _logger = logging.getLogger(__name__)

    from tslearn.metrics import cdist_dtw
    X_distance = cdist_dtw(X)
    
    from sklearn.cluster import DBSCAN
    db_dtw = DBSCAN(metric='precomputed',eps=X_distance.mean() )
    DBSCAN_dtw_labels = db_dtw.fit_predict(X=X_distance)

    # _logger.info('Using DBSCAN(DTW) and results: {}'.format(DBSCAN_dtw_labels)) 
    ari=adjusted_rand_score(cluster_labels, DBSCAN_dtw_labels) 
    _logger.info('Using DBSCAN(DTW) and Adjusted Rand index，ARI: {}'.format(ari)) 
    

    return ari,DBSCAN_dtw_labels


def OPTICS_dtw(X, cluster_labels):

    _logger = logging.getLogger(__name__)

    from tslearn.metrics import cdist_dtw
    X_distance = cdist_dtw(X)
    
    from sklearn.cluster import OPTICS
    op_dtw = OPTICS(metric='precomputed')
    OPTICS_dtw_labels = op_dtw.fit_predict(X=X_distance)

    # _logger.info('Using OPTICS(DTW) and results: {}'.format(OPTICS_dtw_labels)) 
    ari=adjusted_rand_score(cluster_labels, OPTICS_dtw_labels) 
    _logger.info('Using OPTICS(DTW) and Adjusted Rand index，ARI: {}'.format(ari)) 
    

    return ari,OPTICS_dtw_labels

def VAR_LiNGAM(X, cluster_labels, matrix_labels, max_lag):

    _logger = logging.getLogger(__name__)

    # https://lingam.readthedocs.io/en/latest/tutorial/var.html
    import lingam

    n = len(cluster_labels) # n is the number of subjects
    auc_list = []
    for s in range(n):

        Xs = X[s]
        model = lingam.VARLiNGAM(lags=max_lag,random_state=0)
        model.fit(Xs)

        estimated_graph = np.abs(model.adjacency_matrices_)
        # Normalize to [0,1]
        # estimated_graph = estimated_graph/ np.max(estimated_graph)
        
        groundtruth_graph = matrix_labels [ cluster_labels[s] ]
        Score = AUC_score(estimated_graph,groundtruth_graph)
        auc_list.append(Score['AUC'])

    _logger.info('Using VAR_LiNGAM and AUC: {}'.format(sum(auc_list)/n)) 
    return sum(auc_list)/n


def DY_NOTEARS(X, cluster_labels, matrix_labels, max_lag):

    _logger = logging.getLogger(__name__)

    n = len(cluster_labels) # n is the number of subjects
    auc_list = []
    for s in range(n):

        Xs = X[s]

        # https://causalnex.readthedocs.io/en/latest/source/api_docs/causalnex.structure.dynotears.from_numpy_dynamic.html?highlight=dynotears
        from causalnex.structure.dynotears import from_pandas_dynamic
        import pandas as pd
        sm = from_pandas_dynamic( pd.DataFrame(Xs), p=max_lag)
        # from causalnex.plots import plot_structure
        # viz = plot_structure(sm)  # Default CausalNex visualisation
        # viz.draw('test.png')
        # import networkx as nx
        # estimated_matrix = nx.to_numpy_matrix(sm)

        estimated_graph = []

        num_variables = Xs.shape[1]
        # instantaneous effct
        tmp_graph = np.zeros((num_variables,num_variables))
        for i in range(num_variables):
            for j in range(num_variables):

                node1 = str(i)+'_lag0'

                node2 = str(j)+'_lag0'

                # node2 -> node1
                if sm.get_edge_data(node2,node1) is None:
                    continue
                
                # j->i
                tmp_graph[i,j] = sm.get_edge_data(node2,node1)['weight']


        estimated_graph.append(tmp_graph)
        for l in range(1,max_lag+1):
            
            tmp_graph = np.zeros((num_variables,num_variables))
            for i in range(num_variables):
                for j in range(num_variables):

                    node1 = str(i)+'_lag0'

                    node2 = str(j)+'_lag'+str(l)
                    # print(node2,'->',node1)

                    # node2 -> node1
                    if sm.get_edge_data(node2,node1) is None:
                        continue
                    
                    # j->i
                    tmp_graph[i,j] = sm.get_edge_data(node2,node1)['weight']
            
            estimated_graph.append(tmp_graph)
        
        estimated_graph = np.array(estimated_graph)
        
        groundtruth_graph = matrix_labels [ cluster_labels[s] ]
        Score = AUC_score(estimated_graph,groundtruth_graph)
        auc_list.append(Score['AUC'])

    _logger.info('Using DYNOTEARS and AUC: {}'.format(sum(auc_list)/n)) 
    return sum(auc_list)/n



def PC_MCI(X, cluster_labels, matrix_labels, max_lag):

    _logger = logging.getLogger(__name__)

    n = len(cluster_labels) # n is the number of subjects
    auc_list = []
    for s in range(n):

        Xs = X[s]

        # https://jakobrunge.github.io/tigramite/index.html#tigramite-pcmci-pcmci
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests import ParCorr
        import tigramite.data_processing as pp
        dataframe = pp.DataFrame(Xs)
        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=None)

        alpha = 0.05 # Significant parents at alpha = 0.05
        significant_matrix = results['p_matrix'] < alpha
        significant_matrix = significant_matrix.astype(np.float)


        # Note that |val_matrix[i,j]|>0 means Xi->Xj in PCMCI, see https://github.com/jakobrunge/tigramite/blob/master/tutorials/tigramite_tutorial_basics.ipynb
        # In our synthetic dataset B[i,j]=1 means Xj->Xi, so we need to transpose the val_matrix
        estimated_graph = np.array([  significant_matrix[:,:,l].T *results['val_matrix'][:,:,l].T for l in range(max_lag+1) ])
        estimated_graph = np.abs(estimated_graph)
        # Normalize to [0,1]
        # estimated_graph = estimated_graph/ np.max(estimated_graph)

        groundtruth_graph = matrix_labels [ cluster_labels[s] ]
        Score = AUC_score(estimated_graph,groundtruth_graph)
        auc_list.append(Score['AUC'])

    _logger.info('Using PCMCI and AUC: {}'.format(sum(auc_list)/n)) 
    return sum(auc_list)/n



if __name__ == "__main__":

    
    # Get arguments parsed
    args = get_args()
    
    output_dir = 'baseline/baseline_output/{}'.format(datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])

    X, cluster_labels, matrix_labels  = synthetic(seed=args.seed, num_groups=args.num_groups, num_subjects_per_group=args.num_subjects_per_group, num_samples=args.num_samples, num_variables=args.num_variables, max_lag=args.max_lag)

    # assign n_clusters with the ground-truth
    km_dtw_ARI,km_dtw_labels = kmeans_dtw(X=X, num_groups= args.num_groups, cluster_labels=cluster_labels)
            
    km_euclidean_ARI,km_euclidean_labels = kmeans_euclidean(X=X, num_groups= args.num_groups, cluster_labels=cluster_labels)

    DBSCAN_dtw_ARI,DBSCAN_dtw_labels = DBSCAN_dtw(X=X, cluster_labels=cluster_labels)

    OPTICS_dtw_ARI,OPTICS_dtw_labels = OPTICS_dtw(X=X, cluster_labels=cluster_labels)
    
    # assign cluster_labels with the ground-truth
    var_lingam_AUC = VAR_LiNGAM(X=X, cluster_labels=cluster_labels , matrix_labels=matrix_labels, max_lag = args.max_lag)

    pcmci_AUC = PC_MCI(X=X, cluster_labels=cluster_labels , matrix_labels=matrix_labels, max_lag = args.max_lag)

    dynotear_AUC = DY_NOTEARS(X=X, cluster_labels=cluster_labels , matrix_labels=matrix_labels, max_lag = args.max_lag)
