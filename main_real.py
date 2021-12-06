#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pytz import timezone
from datetime import datetime
import numpy as np
import torch

from data_loader.real_dataset import RealDataset
from models.Hierarchical_Causal_Clustering import Hierarchical_Causal_Clustering
from trainers.trainer import Trainer


from helpers.config_utils import save_yaml_config, get_args
from helpers.log_helper import LogHelper
from helpers.torch_utils import set_seed
from helpers.dir_utils import create_dir
from helpers.analyze_utils import  plot_timeseries, plot_recovered_graph


from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import adjusted_rand_score

def kmeans_dtw(X,num_groups,cluster_labels):

    _logger = logging.getLogger(__name__)

    km = TimeSeriesKMeans(n_clusters=num_groups, metric="dtw")
    km_dtw_labels = km.fit_predict(X=X)

    ari=adjusted_rand_score(cluster_labels, km_dtw_labels) 

    return ari,km_dtw_labels

def kmeans_euclidean(X,num_groups,cluster_labels):

    _logger = logging.getLogger(__name__)

    km = TimeSeriesKMeans(n_clusters=num_groups, metric="euclidean")
    km_euclidean_labels = km.fit_predict(X=X)
    
    ari=adjusted_rand_score(cluster_labels, km_euclidean_labels) 
    
    return ari,km_euclidean_labels


def DBSCAN_dtw(X, cluster_labels):

    _logger = logging.getLogger(__name__)

    from tslearn.metrics import cdist_dtw
    X_distance = cdist_dtw(X)
    
    from sklearn.cluster import DBSCAN
    db_dtw = DBSCAN(metric='precomputed',eps=X_distance.mean() )
    DBSCAN_dtw_labels = db_dtw.fit_predict(X=X_distance)

    ari=adjusted_rand_score(cluster_labels, DBSCAN_dtw_labels) 
    
    return ari,DBSCAN_dtw_labels


def OPTICS_dtw(X, cluster_labels):

    _logger = logging.getLogger(__name__)

    from tslearn.metrics import cdist_dtw
    X_distance = cdist_dtw(X)
    
    from sklearn.cluster import OPTICS
    op_dtw = OPTICS(metric='precomputed')
    OPTICS_dtw_labels = op_dtw.fit_predict(X=X_distance)

    ari=adjusted_rand_score(cluster_labels, OPTICS_dtw_labels) 
    

    return ari,OPTICS_dtw_labels

def baseline_cell(output_dir):
    
    np.set_printoptions(precision=3)
    
    _logger = logging.getLogger(__name__)
    
    # Get arguments parsed
    args = get_args()
    args.num_samples = 30
    args.num_subjects_per_group = 25
    args.num_groups = 2
    args.num_variables = 11
    
    # Reproducibility
    set_seed(args.seed)

    dataset = RealDataset().cell_dataset
   
    # Shuffle the order of subjects
    data = np.concatenate(dataset)
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    shuffled_cluster = np.concatenate([ [i]*args.num_subjects_per_group for i in range(args.num_groups) ]) [permutation]

    # Baseline methods
    ari,_ = kmeans_dtw(shuffled_data,args.num_groups,shuffled_cluster)
    _logger.info('Using k-means(DTW) and Adjusted Rand index，ARI: {}'.format(ari)) 

    ari,_ = kmeans_euclidean(shuffled_data,args.num_groups,shuffled_cluster)
    _logger.info('Using k-means(euclidean) and Adjusted Rand index，ARI: {}'.format(ari)) 

    ari,_ = DBSCAN_dtw(shuffled_data,shuffled_cluster)
    _logger.info('Using DBSCAN(DTW) and Adjusted Rand index，ARI: {}'.format(ari)) 

    ari,_ = OPTICS_dtw(shuffled_data,shuffled_cluster)
    _logger.info('Using OPTICS(DTW) and Adjusted Rand index，ARI: {}'.format(ari)) 

def hear_cell(output_dir):
    
    np.set_printoptions(precision=3)
    
    # Get arguments parsed
    args = get_args()
    args.num_samples = 30
    args.num_subjects_per_group = 25
    args.num_groups = 2
    args.num_variables = 11
    args.max_lag = 0
    args.num_iterations_clustering = 1000
    args.seed=2022

    _logger = logging.getLogger(__name__)
    
    # Save the configuration for logging purpose
    save_yaml_config(args, path='{}/config_cell.yaml'.format(output_dir))

    # Reproducibility
    set_seed(args.seed)

    dataset = RealDataset().cell_dataset
    # Look at data
    _logger.info('The shape of data (num_groups ,num_subjects, Ts, num_variables): {}, {}'.format(len(dataset),dataset[0].shape))
    # for q in range(args.num_groups):
    #     plot_timeseries(dataset[q],'Group {}'.format(q),display_mode=False,save_name=output_dir+'/group{}_timeseries_cell.png'.format(q+1))

    # Shuffle the order of subjects
    data = np.concatenate(dataset)
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    shuffled_cluster = np.concatenate([ [i]*args.num_subjects_per_group for i in range(args.num_groups) ]) [permutation]
    _logger.info('The groudtruth clusters: {}'.format(shuffled_cluster))

    # Init model
    model = Hierarchical_Causal_Clustering(args.num_samples, args.num_variables, args.max_lag, args.device, args.prior_mu, args.prior_sigma, args.prior_nu, args.prior_omega)

    trainer = Trainer(args.learning_rate, args.num_iterations_clustering, args.num_iterations_structurelearning, args.num_output, args.num_MC_sample, args.num_total_iterations)

    input_X = torch.tensor(shuffled_data,dtype=torch.float32,device=args.device)

    trainer.train_model(model=model, X = input_X, output_dir=output_dir)

    # Save result
    trainer.log_and_save_intermediate_outputs(model)

    _logger.info('Finished training model')
    _logger.info('The groudtruth clusters: {}'.format(shuffled_cluster))
    # Calculate performance
    estimated_cluster = [-1]*(args.num_groups * args.num_subjects_per_group)
    for c in model.cluster:
            for subject in c:
                estimated_cluster[subject] = model.cluster.index(c)
    _logger.info('The estimated clusters: {}'.format(estimated_cluster))
    
    from sklearn.metrics import adjusted_rand_score
    ari=adjusted_rand_score(shuffled_cluster, estimated_cluster) 
    _logger.info('Adjusted Rand index，ARI: {}'.format(ari)) 

    n_cluster = len(model.cluster)
    for c in range(n_cluster):
        parameters = model.causal_structures[c]

        estimate_B = np.abs(parameters[0][0].numpy())

        estimate_graph = estimate_B.reshape(1,args.num_variables,args.num_variables)

        group_dag =  None
        plot_recovered_graph(estimate_graph,group_dag,title='estimate_graph_{} in cellular dataset'.format(c),display_mode=False,save_name=output_dir+'/estimate_graph_{} in cellular dataset.png'.format(c))





if __name__ == '__main__':

    
    output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    
    # Setup for logging
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')

    hear_cell(output_dir)
    baseline_cell(output_dir)

