import numpy as np
import tensorflow as tf
import neuralnet
import os
import sys
import importlib
from dafi import PhysicsModel
from dafi import random_field as rf
from dafi.random_field import foam
import data_preproc as preproc
import math
def penalties():
    penalties_list = []
    nscalar_invariants = 6
    nbasis_tensors = 1
    nhlayers = 4
    nnodes = 8 
    alpha = 0
    PreProc = getattr(preproc, 'Scale')
    preprocess_data = PreProc()
    selected_rows = np.loadtxt('selected_rows')
    selected_rows = selected_rows.astype(int)
    '''
    lamda1file = '2500/tr((((1_(0.09_omega))_skew(grad(U)))&((1_(0.09_omega))_skew(grad(U)))))'
    lamda2file = '2500/tr((((((1_(0.09_omega))_skew(grad(U)))&((1_(0.09_omega))_skew(grad(U))))&((1_(0.09_omega))_symm(grad(U))))&((1_(0.09_omega))_symm(grad(U)))))'
    Rewfile = '2500/(((mag(skew(grad(U)))_yWall)_yWall)_nu)'
    pdfile = '2500/(min(kOmegaSST_G,(((c1_betaStar)_k)_omega))_((betaStar_omega)_k))'
    '''  
    foam_case = 'inputs/foam_base'
    foam_rc = None 
    cell_volumes = rf.foam.get_cell_volumes(foam_case, foam_rc=foam_rc)
    maxw = np.max(cell_volumes)
    cell_volumes = cell_volumes / maxw
    def penalty_func_1(state,REnKf_iter,isamp):
        lamda1file = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'lamda1')
        Resfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'Res')
        Rewfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'Rew')
        swfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'sw')
        rdfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'rd')
        nkfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'nk')
        
        lamda1 = rf.foam.read_scalar_field(lamda1file)
        sw = rf.foam.read_scalar_field(swfile)
        Rew = rf.foam.read_scalar_field(Rewfile)
        Res = rf.foam.read_scalar_field(Resfile)
        nk = rf.foam.read_scalar_field(nkfile)
        rd = rf.foam.read_scalar_field(rdfile)
            
        input_scalars = np.empty((len(lamda1),nscalar_invariants))
        input_scalars[:,0] = sw
        input_scalars[:,1] = Rew
        input_scalars[:,2] = Res
        input_scalars[:,3] = nk
        input_scalars[:,4] = rd
        input_scalars[:,5] = lamda1
        file0 = os.path.join('results_ensemble',f'input_preproc_stat_{0}_{REnKf_iter}')
        file1 = os.path.join('results_ensemble',f'input_preproc_stat_{1}_{REnKf_iter}')
        xmin = np.loadtxt(file0)
        xmax = np.loadtxt(file1)
        preprocess_data_stats = [xmin , xmax]
            # scale the inputs
        input_scalars_scale = preprocess_data.scale(input_scalars, preprocess_data_stats)
        input_scalars_scale = input_scalars_scale[selected_rows]
        nn = neuralnet.NN(nscalar_invariants,nbasis_tensors,nhlayers,nnodes,alpha)   
        w_shapes = neuralnet.weights_shape(nn.trainable_variables)
        w_reshape = neuralnet.reshape_weights(state,w_shapes)
        nn.set_weights(w_reshape)
        betann = nn(input_scalars_scale)
        #print(type(input_scalars_scale[1,:]))
        #print(input_scalars_scale[1,:].shape)
        #print(type(input_scalars_scale))
        print('betannshape',betann.shape)
        return betann 
    
    def penalty_gradient_1(state,REnKf_iter,isamp):
        lamda1file = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'lamda1')
        Resfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'Res')
        Rewfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'Rew')
        swfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'sw')
        rdfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'rd')
        nkfile = os.path.join(os.path.join('results_ensemble', f'sample_{isamp:d}'),str(REnKf_iter), 'nk')
        
        lamda1 = rf.foam.read_scalar_field(lamda1file)
        sw = rf.foam.read_scalar_field(swfile)
        Rew = rf.foam.read_scalar_field(Rewfile)
        Res = rf.foam.read_scalar_field(Resfile)
        nk = rf.foam.read_scalar_field(nkfile)
        rd = rf.foam.read_scalar_field(rdfile)
            
        input_scalars = np.empty((len(lamda1),nscalar_invariants))
        input_scalars[:,0] = sw
        input_scalars[:,1] = Rew
        input_scalars[:,2] = Res
        input_scalars[:,3] = nk
        input_scalars[:,4] = rd
        input_scalars[:,5] = lamda1
        file0 = os.path.join('results_ensemble',f'input_preproc_stat_{0}_{REnKf_iter}')
        file1 = os.path.join('results_ensemble',f'input_preproc_stat_{1}_{REnKf_iter}')
        xmin = np.loadtxt(file0)
        xmax = np.loadtxt(file1)
        preprocess_data_stats = [xmin , xmax]
            # scale the inputs
        input_scalars_scale = preprocess_data.scale(input_scalars, preprocess_data_stats)
        np.savetxt('in',input_scalars_scale)
        input_scalars_scale = input_scalars_scale[selected_rows]
        nn = neuralnet.NN(nscalar_invariants,nbasis_tensors,nhlayers,nnodes,alpha)   
        w_shapes = neuralnet.weights_shape(nn.trainable_variables)
        w_reshape = neuralnet.reshape_weights(state,w_shapes)
        nn.set_weights(w_reshape)
        gradient = []
        for i in range(input_scalars_scale.shape[0]):
            with tf.GradientTape() as tape:
                tape.watch(nn.trainable_variables) 
                x = input_scalars_scale[i,:] 
                x = x.reshape((1,nscalar_invariants))
                #print(x.shape)
                y = nn(x)
            grads = tape.gradient(y, nn.trainable_variables)
            grad=[]
            for i in range(len(grads)):
                add = np.array(grads[i]).flatten()                
                grad = np.concatenate([grad,add])
            gradient = np.hstack([gradient,grad])
        gradient = gradient.reshape((input_scalars_scale.shape[0],len(grad)))       
        return gradient

    #weight_matrix_1 = np.diag(cell_volumes)  
    #weight_matrix_1 = np.diag(cell_volumes) 
    weight_matrix_1 = np.eye(len(selected_rows))
    def lambda_1(iteration):
        Lamb = 0.5 * 0.001 * ( math.tanh((iteration-5)/2) + 1)
        return Lamb   
    # 创建第一个惩罚项的字典
    penalty_1 = {
        'lambda': lambda_1,
        'weight_matrix': weight_matrix_1,
        'penalty': penalty_func_1,
        'gradient': penalty_gradient_1
    }
    
    penalties_list.append(penalty_1)
    
    # 定义第二个惩罚项...
    
    return penalties_list
