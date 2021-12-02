import logging
from typing import DefaultDict
import numpy as np
import copy
from numpy.core.fromnumeric import mean
import torch
from itertools import chain
from torch import optim
from torch import random
from torch._C import Graph
from torch.distributions import Normal,Categorical
from torch.nn.utils import clip_grad_value_
from helpers.analyze_utils import  plot_losses, AUC_score
from tqdm import tqdm

class Trainer(object):
    """
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, learning_rate, num_iterations_clustering,num_iterations_structurelearning, num_output, num_MC_sample, num_total_iterations):

        self.learning_rate = learning_rate
        self.num_iterations_clustering = num_iterations_clustering
        self.num_iterations_structurelearning = num_iterations_structurelearning
        self.num_output = num_output
        self.num_MC_sample = num_MC_sample
        self.num_total_iterations = num_total_iterations

    def train_model(self, model, X,  output_dir):
        self.output_dir = output_dir
        self.flag_clustering = True
        # X is shape of (n,Ts,m) # n is the number of total subjects and m is the number of variables.
        n,Ts,m = X.shape

        Likelihood_dict = DefaultDict(list)
        Cluster_dict = DefaultDict(list)
        Graph_dict = DefaultDict(list)
        for k in range(1,n+1):
            # Cluster_result stores the clustering result of n subjects and is a list with size k, Likelihood_result stores the average_likelihood of corresponding cluster.
            # [[1,2,3],[4,5,6]] [[0.1,0.3,0.5],[0.1,0.2,0.7]] [[G1][G2]]
            Cluster_result, Likelihood_result, Graph_result = self.Causal_Clustering(model,X,k=k,last_cluster=Cluster_dict[k-1],last_likelihood=Likelihood_dict[k-1],last_graph=Graph_dict[k-1])
            Cluster_dict[k] = Cluster_result
            Likelihood_dict[k] = Likelihood_result
            Graph_dict[k] = Graph_result
            self._logger.info("The current clusters:{}".format(Cluster_result))
        score = []
        for k in range(1,n+1):
            s = sum(sum(Likelihood_dict[k],[]))
            score.append(s)
        plot_losses(score,save_name=output_dir+'/likelihood.png')
        cluster_index = np.argmax(score)
        model.cluster = Cluster_dict[cluster_index+1]
        self._logger.info("Learning the causal structure for each clusters")
        self.flag_clustering = False
        self.Collective_Causal_Structure_learning(model,X)

    def Collective_Causal_Structure_learning(self,model,X):
        graphs = []
        for clu in model.cluster:
            g = model.prior
            self._logger.info("Learning the causal structure for clusters:{}".format(clu))
            for _ in range(self.num_total_iterations):
                for i in clu:
                    Xs = X[i]
                    g = self.Causal_Structure_learning(g,Xs,self.num_iterations_structurelearning)
            graphs.append(g)

        model.causal_structures = graphs
            

    def Causal_Clustering(self,model,X,k,last_cluster,last_likelihood,last_graph):
        n,Ts,m = X.shape
        if k==1:
            # case: all subject in one cluster
            
            i = np.random.choice(range(n))
            
            self._logger.info("Learn a Causal structure with X{} and calculate the likelihood of each subjects.".format(i))
            G_prior = self.Causal_Structure_learning(prior_structure=model.prior, Xs=X[i],VI_iteration=self.num_iterations_clustering)
            
            Likelihood_result = [[]]
            Cluster_result = [[]]
            Graph_result = [G_prior]
            for i in range(n):
                Xs = X[i]
                Xs_likelihood = self.Calculate_likelihood(causal_structure=G_prior,Xs=Xs)
                Likelihood_result[0].append(Xs_likelihood.item())
                # all subject is in Cluster[0].
                Cluster_result[0].append(i)
                
        else:
            cluster_std_likelihood = [np.std(cluster) for cluster in last_likelihood]
            # Find the cluster with the biggest cluster_std_likelihood and split it into two clusters
            while 1:
                cluster_index = np.argmax(cluster_std_likelihood)
                cluster_X_index = last_cluster[cluster_index]
                if len(cluster_X_index)>1:
                    break
                else:
                    cluster_std_likelihood[cluster_index] = float('-inf')

            self._logger.info("The subject under the biggest cluster_std_likelihood Cluster is:{}".format(cluster_X_index))
            
            i = np.argmax(last_likelihood[cluster_index])
            self._logger.info("The index of subject with biggest likelihood in cluster {} is:{}".format(cluster_X_index,cluster_X_index[i]))
            Gi_tmp = self.Causal_Structure_learning(prior_structure=model.prior,Xs=X[i],VI_iteration=self.num_iterations_clustering)

            # Find the subject with the smallest likelihood under the structure Gi_tmp
            min_likelihood_index = -1
            min_likelihood_value = float('inf')
            for j in cluster_X_index:
                if j==cluster_X_index[i]:
                    continue
                Xs_likelihood = self.Calculate_likelihood(causal_structure=Gi_tmp,Xs=X[j])   
                if Xs_likelihood < min_likelihood_value:  
                    min_likelihood_value = Xs_likelihood
                    min_likelihood_index = j
            
            self._logger.info("The index of subject with smallest likelihood  in cluster {} is:{}".format(cluster_X_index,min_likelihood_index))
            Gj_tmp = self.Causal_Structure_learning(prior_structure=model.prior,Xs=X[min_likelihood_index],VI_iteration=self.num_iterations_clustering)
            

            Cluster_i = [cluster_X_index[i]]
            Cluster_j = [min_likelihood_index]
            
            Cluster_i_likelihood = [(self.Calculate_likelihood(causal_structure=Gi_tmp,Xs=X[cluster_X_index[i]])).item() ]
            Cluster_j_likelihood = [(self.Calculate_likelihood(causal_structure=Gj_tmp,Xs=X[min_likelihood_index]) ).item() ]
            
            self._logger.info("Calculating the log_likelihood of subjects under Gi_tmp and Gj_tmp.")
            for id in cluster_X_index:
                if id==cluster_X_index[i] or id==min_likelihood_index:
                    continue
                Xs_Gi_likelihood = self.Calculate_likelihood(causal_structure=Gi_tmp,Xs=X[id])  
        
                Xs_Gj_likelihood = self.Calculate_likelihood(causal_structure=Gj_tmp,Xs=X[id])  

                if Xs_Gi_likelihood > Xs_Gj_likelihood: 
                    self._logger.info("log_likelihood of X{} under Gi_tmp:{}, log_likelihood of X{} under Gj_tmp:{},so it assigned to Cluster_i.".format(id,Xs_Gi_likelihood,id,Xs_Gj_likelihood))
                    Cluster_i.append(id)
                    Cluster_i_likelihood.append(Xs_Gi_likelihood.item())
                
                else:
                    self._logger.info("log_likelihood of X{} under Gi_tmp:{}, log_likelihood of X{} under Gj_tmp:{},so it assigned to Cluster_j.".format(id,Xs_Gi_likelihood,id,Xs_Gj_likelihood))
                    Cluster_j.append(id) 
                    Cluster_j_likelihood.append(Xs_Gj_likelihood.item())
                    
            
            self._logger.info("The cluster with smallest average likelihood in last clustering result is splited into two clusters:\n {} -> {} and {}".format(cluster_X_index,Cluster_i,Cluster_j))
            
            
            Cluster_result = []
            Likelihood_result = []
            Graph_result = []
            for idx in range(len(last_cluster)):
                if idx!= cluster_index:
                    Cluster_result.append(last_cluster[idx])
                    Likelihood_result.append(last_likelihood[idx])
                    Graph_result.append(last_graph[idx])

            Cluster_result.append(Cluster_i)
            Likelihood_result.append(Cluster_i_likelihood)
            Graph_result.append(Gi_tmp)
            Cluster_result.append(Cluster_j)
            Likelihood_result.append(Cluster_j_likelihood)
            Graph_result.append(Gj_tmp)

            
        return Cluster_result, Likelihood_result, Graph_result


    def Calculate_likelihood(self,causal_structure,Xs):
        
        
        causal_distribution_B = Normal(loc=causal_structure[0][0], scale=causal_structure[0][1])
        causal_distribution_A = Normal(loc=causal_structure[1][0], scale=causal_structure[1][1])
        
        m = causal_structure[0][0].shape[0] # num_variables
        noise_prob_ = causal_structure[2][0]
        noise_prob_last = torch.ones(size=[m],device=noise_prob_.device) - torch.sum(noise_prob_,axis=1)
        noise_prob = torch.cat((noise_prob_,noise_prob_last.reshape(m,1)),1)

        mix  = Categorical(probs=noise_prob)
        comp = Normal(loc=causal_structure[2][1] , scale=causal_structure[2][2])

        causal_distribution = [causal_distribution_B,causal_distribution_A, mix, comp]
        # Calculating L_ell
        M = self.num_MC_sample
        log_p_Xs = 0.0
        for t in range(M):
            log_p_Xs +=  self.log_likelihood_fun(distribution=causal_distribution,X=Xs) 
        log_p_Xs = log_p_Xs/M
        
        return log_p_Xs

            

    def Causal_Structure_learning(self,prior_structure,Xs,VI_iteration):
        
        train_losses = []

        
        # Init prior
        
        prior_distribution_B = Normal(loc=prior_structure[0][0], scale=prior_structure[0][1])
        prior_distribution_A = Normal(loc=prior_structure[1][0], scale=prior_structure[1][1])
        
        prior_distribution = [prior_distribution_B,prior_distribution_A]
        
        # Init posterior
        posterior_structure = copy.deepcopy(prior_structure)
        
        for para in sum(posterior_structure,[]):
            para.requires_grad = True
        
        posterior_distribution_B = Normal(loc=posterior_structure[0][0], scale=posterior_structure[0][1])
        posterior_distribution_A = Normal(loc=posterior_structure[1][0], scale=posterior_structure[1][1])
        
        m = prior_structure[0][0].shape[0] # num_variables
        noise_prob_ = posterior_structure[2][0]
        noise_prob_last = torch.ones(size=[m],device=noise_prob_.device) - torch.sum(noise_prob_,axis=1)
        noise_prob = torch.cat((noise_prob_,noise_prob_last.reshape(m,1)),1)

        mix  = Categorical(probs=noise_prob)
        comp = Normal(loc=posterior_structure[2][1] , scale=posterior_structure[2][2])

        posterior_distribution = [posterior_distribution_B,posterior_distribution_A, mix, comp]
        

            
        optimizer = optim.Adam(params= sum(posterior_structure,[]), lr=self.learning_rate)
        for iteration in range(VI_iteration):

            optimizer.zero_grad()

            loss = self.loss_fun(prior_distribution,posterior_distribution,Xs)

            if loss < 0.0: # if loss is negative, early stopping
                self._logger.info("The log_likelihood is negative, early stopping.")
                break
            
            if torch.isnan(loss): # if loss is NAN, break
                self._logger.info("!!!! Loss is NAN, check the generated data and configuration.")
                break
            loss.backward(retain_graph=True)

            # Clipping Gradient for parameters
            clip_grad_value_(sum(posterior_structure,[]),clip_value=5.0)
            with torch.no_grad():
                # In case NAN
                for para in sum(posterior_structure,[]):
                    para.grad[torch.isnan(para.grad)] = 0.0
            


            train_losses.append(loss.item())

            if(iteration% self.num_output==0):
                self._logger.info("Iteration {} , loss:{}".format(iteration,loss.item()))

            optimizer.step()
            with torch.no_grad():
                
                #Pi should >0 and <1
                posterior_structure[2][0].data = torch.clamp(posterior_structure[2][0],min=0.0,max=1.0)

                # scale should >0, and  loss will become NAN when scale is 0 so we simply set the minimal scale is 1e-5.
                posterior_structure[0][1].data = torch.clamp(posterior_structure[0][1],min=1e-5)
                posterior_structure[1][1].data = torch.clamp(posterior_structure[1][1],min=1e-5)
                posterior_structure[2][2].data = torch.clamp(posterior_structure[2][2],min=1e-5)

                
                # Set the data of the diagonal in B with 0
                for i in range(posterior_structure[0][1].shape[0]):
                    posterior_structure[0][0][i,i] = 0.0 #loc
                    posterior_structure[0][1][i,i] = 0.0 #scale

        self.train_losses = train_losses
        for para in sum(posterior_structure,[]):
            para.requires_grad = False

        return posterior_structure

    
    def loss_fun(self,prior,posterior,X):
        """
        """

        # Calculating L_kl
        from torch.distributions.kl import kl_divergence
        KL_B = kl_divergence(p=posterior[0], q=prior[0])
        # Replace the diagonal elements(NAN) with zero
        KL_B[torch.isnan(KL_B)] = 0
        
        KL_A = kl_divergence(p=posterior[1], q=prior[1])
        KL_A[torch.isnan(KL_A)] = 0
        L_kl = KL_B + KL_A


        # Calculating L_ell
        L_ell = self.log_likelihood_fun(distribution=posterior,X=X)
        # the log likelihood should be negative, but due to the approximated calculation may be positive, so we simply early stop it.
        if L_ell >0.0 :#(torch.tensor(L_ell)>0.0).sum() > 0.0:
            return torch.tensor([float('-inf')])


        ELBO =   L_ell -L_kl.sum()
        loss = -ELBO
        # loss = - L_ell
        
        # L1 constraint
        
        # B = posterior[0].rsample()
        # A = posterior[1].rsample()
        # L1_loss = torch.norm(B,p=1) + torch.norm(A,p=1)

        # def _notear_constraint(W):
        #     d = W.shape[0]
        #     # Referred from: https://github.com/xunzheng/notears/blob/master/notears/linear.py
        #     M = torch.eye(d,device=W.device, dtype=W.dtype) + W * W / d  # (Yu et al. 2019)
        #     E = torch.matrix_power(M,d-1)
        #     h = torch.sum( torch.t(E) * M) - d
        #     return h

            
        return loss #+ 2*L1_loss + 1e4*_notear_constraint(B)


    def log_likelihood_fun(self,distribution,X):
        
        from torch.distributions import MixtureSameFamily
        gmm = MixtureSameFamily(distribution[2],distribution[3])
        # See E.q. (15)
        log_P_x1 = ( gmm.log_prob(X[0])).sum()


        B = distribution[0].rsample()
        A = distribution[1].rsample()
        
        Ts = X.shape[0]
        m = X.shape[1]
        pl = A.shape[0]

        log_P_xp = []
        for t in range(2,pl+1):

            tmp_noise = torch.matmul((torch.eye(m,device=B.device)-B),X[t-1])
            for p in range(1,t): 
                tmp_noise -=  torch.matmul(A[p-1], X[t-1-p])
            
            log_P_xp.append( t*torch.log( torch.abs(torch.det( torch.eye(m,device=B.device)-B ))) +  (gmm.log_prob(tmp_noise)).sum() )

        self.log_P_xp = log_P_xp

        # log_P_xT = []
        # for t in range(pl+1,Ts+1):

        #     tmp_noise = torch.matmul((torch.eye(m,device=B.device)-B),X[t-1])
        #     for p in range(1,pl+1):
        #         tmp_noise -=  torch.matmul(A[p-1], X[t-1-p])

        #     log_P_xT.append( (pl+1)*torch.log(torch.abs(torch.det( torch.eye(m,device=B.device)-B ))) +  (gmm.log_prob(tmp_noise)).sum() )

        tmp_noise = torch.matmul( (torch.eye(m,device=B.device)-B),X[pl:Ts+1].T)
        for p in range(1,pl+1):
            tmp_noise -=  torch.matmul(A[p-1],X[pl-p:Ts-p].T)
        
        log_P_xT = ((pl+1)*torch.log(torch.abs(torch.det( torch.eye(m,device=B.device)-B ))) + (gmm.log_prob(tmp_noise.T)).sum(dim=1) )


        self.log_P_xT = log_P_xT.sum()
        if self.flag_clustering:
            return (self.log_P_xT + sum(self.log_P_xp) + log_P_x1) #/Ts 
        else:
            return (self.log_P_xT + sum(self.log_P_xp) + log_P_x1) /Ts 


            
    def log_and_save_intermediate_outputs(self,model):
        # may want to save the intermediate results
        
        k = len(model.cluster)
        for i in range(k):
            subjects = [str(s) for s  in model.cluster[i]]
            np.save(self.output_dir+'/estimated_parameters_group_{}.npy'.format("_".join(subjects)), np.array(model.causal_structures[i],dtype=object) ) 
            


