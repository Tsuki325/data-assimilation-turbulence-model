# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for OpenFOAM eddy viscosity nutFoam solver. """

# standard library imports
import os
import shutil
import subprocess
import multiprocessing

# third party imports
import numpy as np
import scipy.sparse as sp
import yaml

# local imports
from dafi import PhysicsModel
from dafi import random_field as rf
from dafi.random_field import foam


import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import neuralnet
import gradient_descent as gd
import regularization as reg
import data_preproc as preproc
from get_inputs import get_inputs


import pdb

TENSORDIM = 9
TENSORSQRTDIM = 3
DEVSYMTENSORDIM = 5
DEVSYMTENSOR_INDEX = [0,1,2,4,5]
NBASISTENSORS = 10
NSCALARINVARIANTS = 5

VECTORDIM = 3

class Model(PhysicsModel):
    """ Dynamic model for OpenFoam Reynolds stress nutFoam solver.

    The eddy viscosity field (nu_t) is infered by observing the
    velocity field (U). Nut is modeled as a random field with lognormal
    distribution and median value equal to the baseline (prior) nut
    field.
    """

    def __init__(self, inputs_dafi, inputs_model):
        # get required DAFI inputs.
        self.nsamples = inputs_dafi['nsamples']
        max_iterations = inputs_dafi['max_iterations']
        self.analysis_to_obs = inputs_dafi['analysis_to_obs']

        # read input file
        self.foam_case = inputs_model['foam_case']
        iteration_nstep = inputs_model['iteration_nstep']
        self.foam_timedir = str(iteration_nstep)

        nweights = inputs_model.get('nweights', None)
        self.ncpu = inputs_model.get('ncpu', 20)
        self.rel_stddev = inputs_model.get('rel_stddev', 0.5)
        self.abs_stddev = inputs_model.get('abs_stddev', 0.5)
        self.obs_rel_std = inputs_model.get('obs_rel_std', 0.001)
        self.obs_abs_std = inputs_model.get('obs_abs_std', 0.0001)

        obs_file = inputs_model['obs_file']
        # obs_err_file = inputs_model['obs_err_file']
        # obs_mat_file = inputs_model['obs_mat_file']

        weight_baseline_file = inputs_model['weight_baseline_file']

        # required attributes
        self.name = 'NN parameterized RANS model'

        # results directory
        self.results_dir = 'results_ensemble'

        # counter
        self.daiteration = 0
        self.DAiteration = 0
        
        self.da_iteration = 0
        self.DA_iteration = 0

        self.iteration_step_length =  100 / iteration_nstep

        # control dictionary
        self.timeprecision = 6
        self.control_list = {
            'application': 'simpleFoam',
            'startFrom': 'latestTime',
            'startTime': '0',
            'stopAt': 'endTime',
            'endTime': f'{self.DA_iteration}',
            'deltaT': f'{self.iteration_step_length}',
            'writeControl': 'runTime',
            'writeInterval': '1',
            'purgeWrite': '2',
            'writeFormat': 'ascii',
            'writePrecision': f'{self.timeprecision}',
            'writeCompression': 'off',
            'timeFormat': 'fixed',
            'timePrecision': '0',
            'runTimeModifiable': 'true',
        }

        nut_base_foamfile = inputs_model['nut_base_foamfile']
        self.foam_info = foam.read_header(nut_base_foamfile)
        self.foam_info['file'] = os.path.join(
            self.foam_case,'system', 'controlDict')

        # NN architecture
        self.nscalar_invariants = inputs_model.get('nscalar_invariants', NSCALARINVARIANTS)
        self.nbasis_tensors = inputs_model.get('nbasis_tensors', NBASISTENSORS)
        nhlayers = inputs_model.get('nhlayers', 10)
        nnodes = inputs_model.get('nnodes', 10)
        alpha = inputs_model.get('alpha', 0.0)

        # initial g
        self.g_init  = np.array(inputs_model.get('g_init', [0.0]*self.nbasis_tensors))
        self.g_scale = inputs_model.get('g_scale', 1.0)

        # data pre-processing
        self.preproc_class = inputs_model.get('preproc_class', None)

        # debug
        self.fixed_inputs  = inputs_model.get('fixed_inputs', True)

        parallel = inputs_model.get('parallel', True)

        ## CREATE NN
        self.nn = neuralnet.NN(self.nscalar_invariants, self.nbasis_tensors,
            nhlayers, nnodes, alpha)
        self.nnbetann = neuralnet.NN(self.nscalar_invariants, self.nbasis_tensors,
            nhlayers, nnodes, alpha)
        self.nnPrt = neuralnet.NN(self.nscalar_invariants, self.nbasis_tensors,
            nhlayers, nnodes, alpha)
        # initial weights
        self.w_init = np.loadtxt(weight_baseline_file)
        # self.w_init = np.loadtxt('./results_dafi/t_0/xf/xf_27') # np.array([])

        self.nbasis = self.nbasis_tensors
        self.nstate = len(self.w_init)

        betann_file_case1 = os.path.join(self.foam_case, 'case1' ,'0.orig', 'betann')
        betann_data_case1 = rf.foam.read_field_file(betann_file_case1)
        betann_data_case1['file'] = os.path.join(self.foam_case, 'case1','0.orig', 'betann')
        self.betann_data_case1 = betann_data_case1
        
        betann_file_case2 = os.path.join(self.foam_case, 'case2' ,'0.orig', 'betann')
        betann_data_case2 = rf.foam.read_field_file(betann_file_case2)
        betann_data_case2['file'] = os.path.join(self.foam_case, 'case2','0.orig', 'betann')
        self.betann_data_case2 = betann_data_case2  
              
        Prt_file_case1 = os.path.join(self.foam_case, 'case1','0.orig', 'Prt')
        Prt_data_case1 = rf.foam.read_field_file(Prt_file_case1)
        Prt_data_case1['file'] = os.path.join(self.foam_case, 'case1','0.orig', 'Prt')
        self.Prt_data_case1 = Prt_data_case1
        
        Prt_file_case2 = os.path.join(self.foam_case, 'case2','0.orig', 'Prt')
        Prt_data_case2 = rf.foam.read_field_file(Prt_file_case2)
        Prt_data_case2['file'] = os.path.join(self.foam_case, 'case2','0.orig', 'Prt')
        self.Prt_data_case2 = Prt_data_case2
        

        # for iw in self.nn.trainable_variables:
        #     self.w_init = np.concatenate([self.w_init, iw.numpy().flatten()])

        self.w_shapes = neuralnet.weights_shape(self.nn.trainable_variables)

        # print NN summary
        print('\n' + '#'*80 + '\nCreated NN:' +
            f'\n  Number of scalar invariants: {self.nscalar_invariants}' +
            f'\n  Number of basis tensors: {self.nbasis_tensors}' +
            f'\n  Number of trainable parameters: {self.nn.count_params()}' +
            '\n' + '#'*80)

        # get the preprocesing class
        if self.preproc_class is not None:
            self.PreProc = getattr(preproc, self.preproc_class)

        # calculate inputs
        # initialize preprocessing instance
        if os.path.isdir(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)

        # observations
        # read observations
        norm_truth_HeatFlux = 30
        norm_truth_p = 2#3.5
        HeatFlux_exp_a = np.loadtxt(obs_file + '/exp/' + '34_hf.txt')[13:29, 1]/norm_truth_HeatFlux
        HeatFlux_exp_s = np.loadtxt(obs_file + '/exp/' + '34_hf.txt')[2:12, 1]/1
     
        p_exp = np.loadtxt(obs_file + '/exp/' + '24_p.txt')[:, 1]/norm_truth_p
        #p_exp         = np.loadtxt(obs_file + '/exp/' + 'EXPToMesh_p')[:, 1]/norm_truth_p
        #cf_exp = np.loadtxt(obs_file + '/exp/' + 'EXPToMesh_cf')[:, 1]/norm_truth_cf
        # pdb.set_trace()
        #self.obs = cf_exp
        theta_exp = 0.0199*100
        self.obs = np.concatenate((HeatFlux_exp_a, HeatFlux_exp_s,p_exp, np.tile(theta_exp, 5)), axis=0)
        self.obs_error = np.diag(self.obs_rel_std * abs(self.obs) + self.obs_abs_std)
        self.nstate_obs = len(self.obs)
        # create sample directories
        sample_dirs = []
        for isample in range(self.nsamples):
            sample_dir = self._sample_dir(isample)
            sample_dirs.append(sample_dir)
            # TODO foam.copyfoam(, post='') - copies system, constant, 0
            shutil.copytree(self.foam_case, sample_dir)
            
        self.sample_dirs = sample_dirs
        # pdb.set_trace()

    def __str__(self):
        return 'Dynamic model for nutFoam eddy viscosity solver.'

    # required methods
    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Creates the OpenFOAM case directories for each sample, creates
        samples of eddy viscosity (nut) based on samples of the KL modes
        coefficients (state) and writes nut field files. Returns the
        coefficients of KL modes for each sample.
        """

        # update X (nut)
        w = np.zeros([self.nstate, self.nsamples])
        for i in range(self.nstate):
            w[i, :] = self.w_init[i] + np.random.normal(0,
                abs(self.w_init[i] * self.rel_stddev + self.abs_stddev)
                , self.nsamples)
        return w

    def state_to_observation(self, state_vec):
        """ Map the states to observation space (from X to HX).

        Modifies the OpenFOAM cases to use nu_t reconstructed from the
        specified coeffiecients. Runs OpenFOAM, and returns the
        velocities at the observation locations.
        """
        self.daiteration += 1
        self.da_iteration = 100 * self.daiteration 
        self.DAiteration += 1
        self.DA_iteration = self.DAiteration * 100
        self.control_list = {
            'application': 'simpleFoam',
            'startFrom': 'latestTime',
            'startTime': '0',
            'stopAt': 'endTime',
            'endTime': f'{self.DA_iteration}',
            'deltaT': f'{self.iteration_step_length}',
            'writeControl': 'runTime',
            'writeInterval': '100',
            'purgeWrite': '2',
            'writeFormat': 'ascii',
            'writePrecision': f'{self.timeprecision}',
            'writeCompression': 'off',
            'timeFormat': 'fixed',
            'timePrecision': '0',
            'runTimeModifiable': 'true',
        }
        
        for isample in range(self.nsamples):
            sample_dir = self._sample_dir(isample)
            foam.write_controlDict(
                self.control_list, self.foam_info['foam_version'],
                self.foam_info['website'], ofcase=sample_dir+'/case1')
            foam.write_controlDict(
                self.control_list, self.foam_info['foam_version'],
                self.foam_info['website'], ofcase=sample_dir+'/case2')
        # set weights
        w = state_vec.copy()
        #if self.da_iteration:
            
        time_dir = f'{self.da_iteration:d}'
        Prtsamps1 = []
        betannsamps1 = []
       
        Prtsamps2 = []
        betannsamps2 = []
        self.preprocess_data = self.PreProc()
        ts = time.time()
        for isamp in range(self.nsamples):
            # true input
            '''
            lamda1file = '2500/tr((((1_(0.09_omega))_skew(grad(U)))&((1_(0.09_omega))_skew(grad(U)))))'
            lamda2file = '2500/tr((((((1_(0.09_omega))_skew(grad(U)))&((1_(0.09_omega))_skew(grad(U))))&((1_(0.09_omega))_symm(grad(U))))&((1_(0.09_omega))_symm(grad(U)))))'
            Rewfile = '2500/(((mag(skew(grad(U)))_yWall)_yWall)_nu)'
            pdfile = '2500/(min(kOmegaSST_G,(((c1_betaStar)_k)_omega))_((betaStar_omega)_k))'
            # iter
            '''

            Fsfile = os.path.join(self._sample_dir(isamp),'case1',str(self.da_iteration-100), 'Fs')
            nkfile = os.path.join(self._sample_dir(isamp),'case1',str(self.da_iteration-100), 'nk')
            Mgfile = os.path.join(self._sample_dir(isamp),'case1',str(self.da_iteration-100), 'Mg')
            TuMfile = os.path.join(self._sample_dir(isamp),'case1',str(self.da_iteration-100), 'TuM')
            PDfile = os.path.join(self._sample_dir(isamp),'case1',str(self.da_iteration-100), 'PD')
  

            #'''

            Fs = rf.foam.read_scalar_field(Fsfile)
            nk = rf.foam.read_scalar_field(nkfile)
            Mg = rf.foam.read_scalar_field(Mgfile)
            TuM = rf.foam.read_scalar_field(TuMfile)
            PD = rf.foam.read_scalar_field(PDfile)
            #Q = rf.foam.read_scalar_field(Qfile)
            
            input_scalars = np.empty((len(nk),self.nscalar_invariants))
            input_scalars_ = np.empty((len(nk),self.nscalar_invariants))
            '''
            input_scalars[:,0] = lamda1
            input_scalars[:,1] = Q
            input_scalars[:,2] = sw
            input_scalars[:,3] = Rew
            '''
            Ma=2.8
            input_scalars[:,0] = Fs
            input_scalars[:,1] = nk
            input_scalars[:, 2] = np.full(input_scalars.shape[0], Ma) 
            input_scalars[:,3] = PD
            input_scalars[:,4] = Mg
            input_scalars[:,5] = TuM


            input_scalars_[:,0] = Fs
            input_scalars_[:,1] = nk
            input_scalars_[:,2] = np.full(input_scalars.shape[0], Ma) 
            input_scalars_[:,3] = PD
            input_scalars_[:,4] = Mg
            input_scalars_[:,5] = TuM
            

            
            wset = w[:, isamp]
            #print(len(wset))
            wset_betann = wset[:int(len(wset)/2)]
            #print(len(wset_betann))
            wset_Prt =    wset[int(len(wset)/2):]
            #print(len(wset_Prt))
            #w6248#
            for i in range(8):
                wset_betann[i+48]=0
                wset_betann[i+120]=0
                wset_betann[i+192]=0
                wset_betann[i+264]=0
            wset_betann[280]=0
           
            for i in range(8):
                wset_Prt[i+48]=0
                wset_Prt[i+120]=0
                wset_Prt[i+192]=0
                wset_Prt[i+264]=0
            wset_Prt[280]=0
 
            w_reshape_betann = neuralnet.reshape_weights(wset_betann, self.w_shapes)
            w_reshape_Prt = neuralnet.reshape_weights(wset_Prt, self.w_shapes)
            self.nnbetann.set_weights(w_reshape_betann)
            self.nnPrt.set_weights(w_reshape_Prt)
            # evaluate NN: cost and gradient
            #with tf.GradientTape(persistent=True) as tape:
            betann = np.array( self.nnbetann(input_scalars) * self.g_scale + self.g_init )
            Prt = np.array( self.nnPrt(input_scalars_) * self.g_scale*0.2 + self.g_init )
            for i in range(len(Prt)):
                 if  Prt[i,0] < 0.25: Prt[i,0] = 0.25
                 if  Prt[i,0] > 2: Prt[i,0] = 2
            #print(Prt)
            Prtsamps1.append(Prt)
            betannsamps1.append(betann)
        print(f'      TensorFlow ... {time.time()-ts:.2f}s')
        
        for isamp in range(self.nsamples):
            # true input
            '''
            lamda1file = '2500/tr((((1_(0.09_omega))_skew(grad(U)))&((1_(0.09_omega))_skew(grad(U)))))'
            lamda2file = '2500/tr((((((1_(0.09_omega))_skew(grad(U)))&((1_(0.09_omega))_skew(grad(U))))&((1_(0.09_omega))_symm(grad(U))))&((1_(0.09_omega))_symm(grad(U)))))'
            Rewfile = '2500/(((mag(skew(grad(U)))_yWall)_yWall)_nu)'
            pdfile = '2500/(min(kOmegaSST_G,(((c1_betaStar)_k)_omega))_((betaStar_omega)_k))'
            # iter
            '''

            Fsfile = os.path.join(self._sample_dir(isamp),'case2',str(self.da_iteration-100), 'Fs')
            nkfile = os.path.join(self._sample_dir(isamp),'case2',str(self.da_iteration-100), 'nk')
            Mgfile = os.path.join(self._sample_dir(isamp),'case2',str(self.da_iteration-100), 'Mg')
            TuMfile = os.path.join(self._sample_dir(isamp),'case2',str(self.da_iteration-100), 'TuM')
            PDfile = os.path.join(self._sample_dir(isamp),'case2',str(self.da_iteration-100), 'PD')


            #'''
            Fs = rf.foam.read_scalar_field(Fsfile)
            nk = rf.foam.read_scalar_field(nkfile)
            Mg = rf.foam.read_scalar_field(Mgfile)
            TuM = rf.foam.read_scalar_field(TuMfile)
            PD = rf.foam.read_scalar_field(PDfile)
            #Q = rf.foam.read_scalar_field(Qfile)
            
            input_scalars = np.empty((len(nk),self.nscalar_invariants))
            input_scalars_ = np.empty((len(nk),self.nscalar_invariants))
            '''
            input_scalars[:,0] = lamda1
            input_scalars[:,1] = Q
            input_scalars[:,2] = sw
            input_scalars[:,3] = Rew
            '''
            Ma=9
            #input_scalars[:,0] = lamda1
            #input_scalars[:,1] = -lamda2
            input_scalars[:,0] = Fs
            input_scalars[:,1] = nk
            input_scalars[:, 2] = np.full(input_scalars.shape[0], Ma) 
            input_scalars[:,3] = PD
            input_scalars[:,4] = Mg
            input_scalars[:,5] = TuM
            
            input_scalars_[:,0] = Fs
            input_scalars_[:,1] = nk
            input_scalars_[:,2] = np.full(input_scalars.shape[0], Ma) 
            input_scalars_[:,3] = PD
            input_scalars_[:,4] = Mg
            input_scalars_[:,5] = TuM

            
            wset = w[:, isamp]
            #print(len(wset))
            wset_betann = wset[:int(len(wset)/2)]
            #print(len(wset_betann))
            wset_Prt =    wset[int(len(wset)/2):]
            #print(len(wset_Prt))
            #w6248#
            for i in range(8):
                wset_betann[i+48]=0
                wset_betann[i+120]=0
                wset_betann[i+192]=0
                wset_betann[i+264]=0
            wset_betann[280]=0
           
            for i in range(8):
                wset_Prt[i+48]=0
                wset_Prt[i+120]=0
                wset_Prt[i+192]=0
                wset_Prt[i+264]=0
            wset_Prt[280]=0

            w_reshape_betann = neuralnet.reshape_weights(wset_betann, self.w_shapes)
            w_reshape_Prt = neuralnet.reshape_weights(wset_Prt, self.w_shapes)
            self.nnbetann.set_weights(w_reshape_betann)
            self.nnPrt.set_weights(w_reshape_Prt)
            # evaluate NN: cost and gradient
            #with tf.GradientTape(persistent=True) as tape:
            betann = np.array( self.nnbetann(input_scalars) * self.g_scale + self.g_init )
            Prt = np.array( self.nnPrt(input_scalars_) * self.g_scale*0.2 + self.g_init )
            for i in range(len(Prt)):
                 if  Prt[i,0] < 0.25: Prt[i,0] = 0.25
                 if  Prt[i,0] > 2: Prt[i,0] = 2
            #print(Prt)
            Prtsamps2.append(Prt)
            betannsamps2.append(betann)
        print(f'      TensorFlow ... {time.time()-ts:.2f}s')
        
        
        # write sample
        for i in range(self.nsamples):
            iPrt1 = Prtsamps1[i]
            iPrt2 = Prtsamps2[i] 
            self._modify_foam_case1_Prt(iPrt1, self.da_iteration-100, foam_dir=self._sample_dir(i)+'/case1')
            self._modify_foam_case2_Prt(iPrt2, self.da_iteration-100, foam_dir=self._sample_dir(i)+'/case2')
            
            ibetann1 = betannsamps1[i] 
            ibetann2 = betannsamps2[i]
            self._modify_foam_case1_betann(ibetann1, self.da_iteration-100, foam_dir=self._sample_dir(i)+'/case1')
            self._modify_foam_case2_betann(ibetann2, self.da_iteration-100, foam_dir=self._sample_dir(i)+'/case2')
        
        inputs = []
        parallel = multiprocessing.Pool(self.ncpu)
        for i in range(self.nsamples):
            inputs.append((self._sample_dir(i) + '/case1', self.da_iteration, self.timeprecision))
            inputs.append((self._sample_dir(i) + '/case2', self.da_iteration, self.timeprecision))
        _ = parallel.starmap(_run_foam, inputs)
        parallel.close() 

        for i in range(self.nsamples):
            
            _ = writeInput(self._sample_dir(i)+'/case1', self.da_iteration, self.timeprecision)
            _ = writeInput(self._sample_dir(i)+'/case2', self.da_iteration, self.timeprecision)
        # get HX
        norm_truth_HeatFlux = 30
        norm_truth_p = 2
        #print('hxnorm'norm_truth)
        '''
        for isample in range(self.nsamples):
            file = os.path.join(self._sample_dir(isample), time_dir, 'U')
            U = foam.read_vector_field(file)
            Ux = U[:, 0] # * 0.5
            Uy = U[:, 1]  #500
            Uz = U[:, 2] * 1000 #500
            state_in_obs[:, isample] = np.concatenate([Ux, Uy]) / norm_truth
        '''
        
        state_in_obs = np.empty([self.nstate_obs, self.nsamples])
        for isample in range(self.nsamples):

            file = os.path.join(self._sample_dir(isample), 'case1' ,'postProcessing', 'sampleDict1', time_dir)
            file1 = os.path.join(self._sample_dir(isample), 'case1' ,'postProcessing', 'sampleDict', time_dir)
            file_ = os.path.join(self._sample_dir(isample), 'case2' ,'postProcessing', 'sampleDict1', time_dir)
            
            p_cfd = np.loadtxt(file + '/p_walls_constant.raw',skiprows=2) 
            u_cfd = np.loadtxt(file1 + '/line_x0_U.xy')
            HeatFlux_cfd = np.loadtxt(file_ + '/wallHeatFlux_walls_constant.raw',skiprows=2)
            xhfexp = np.loadtxt('inputs/data/exp/' + '34_hf.txt')[:, 0]/1.2
            xpexp = np.loadtxt('inputs/data/exp/' + '24_p.txt')[:, 0]
            
            from scipy import interpolate
            interp_func1 = interpolate.interp1d(HeatFlux_cfd[:,0],-HeatFlux_cfd[:,3]/61000)
            exp_interplot_hf = interp_func1(xhfexp)
            HeatFlux_cfd_a =  exp_interplot_hf[13:29]/norm_truth_HeatFlux
            
            HeatFlux_cfd_s =  exp_interplot_hf[2:12]/1
            

            interp_func3 = interpolate.interp1d(u_cfd[:,1],u_cfd[:,0]) #  边界层速度  插值到U*0.99
            theta_cfd = 100*interp_func3(570*0.99)
            
            interp_func2 = interpolate.interp1d(p_cfd[:,0]/0.023,p_cfd[:,3]/23526)
            exp_interplot_p = interp_func2(xpexp)
            p_cfd =  exp_interplot_p/norm_truth_p
            
            #p_cfd         = np.loadtxt(file + '/p_walls_constant.raw',skiprows=2)[100:200, 3]/2509/norm_truth_p         
            #p_cfd         = np.loadtxt(file + '/p_walls_constant.raw',skiprows=2)[100:200, 3]/2509/norm_truth_p         
            #cf_cfd = np.loadtxt(file + '/wallShearStress_walls_constant.raw',skiprows=2)[157:230, 3]/0.7/605/605/0.5 / norm_truth_cf
            #v1 = np.loadtxt(file + '/line_x1_U.xy')[:, 2] #'/UyFullField') * 3
            #v2 = np.loadtxt(file + '/line_x3_U.xy')[:, 2] #'/UyFullField') * 3
            #v3 = np.loadtxt(file + '/line_x5_U.xy')[:, 2] #'/UyFullField') * 3
            #v4 = np.loadtxt(file + '/line_x7_U.xy')[:, 2] #'/UyFullField') * 3
            #Uy = np.hstack([v1, v2, v3, v4]) 
            #state_in_obs[:, isample] =  -cf_cfd
            #state_in_obs[:, isample] = p_cfd
            #state_in_obs[:, isample] = np.concatenate(( HeatFlux_cfd,p_cfd,[theta_cfd,theta_cfd,theta_cfd,theta_cfd,theta_cfd,theta_cfd,theta_cfd,theta_cfd,theta_cfd,theta_cfd],[HeatFlux_plate,HeatFlux_plate,HeatFlux_plate,HeatFlux_plate,HeatFlux_plate,HeatFlux_plate,HeatFlux_plate,HeatFlux_plate,HeatFlux_plate]), axis=0)
            state_in_obs[:, isample] = np.concatenate((HeatFlux_cfd_a,HeatFlux_cfd_s, p_cfd, np.tile(theta_cfd, 5)), axis=0)
        
        return state_in_obs

    def get_obs(self, time):
        """ Return the observation and error matrix.
        """
        return self.obs, self.obs_error

    def clean(self, loop):
        if loop == 'iter' and self.analysis_to_obs:
            for isamp in range(self.nsamples):
                dir = os.path.join(self._sample_dir(isamp),
                                   f'{self.da_iteration + 1:d}')
                shutil.rmtree(dir)

    # internal methods
    def _sample_dir(self, isample):
        "Return name of the sample's directory. "
        return os.path.join(self.results_dir, f'sample_{isample:d}')

    def _modify_foam_case1_Prt(self, Prt, step, foam_dir=None):
        Prt_data = self.Prt_data_case1
        Prt_data['internal_field']['value'] = Prt
        if foam_dir is not None:
            Prt_data['file'] = os.path.join(foam_dir, str(step), 'Prt')
        _ = rf.foam.write_field_file(**Prt_data)

    def _modify_foam_case2_Prt(self, Prt, step, foam_dir=None):
        Prt_data = self.Prt_data_case2
        Prt_data['internal_field']['value'] = Prt
        if foam_dir is not None:
            Prt_data['file'] = os.path.join(foam_dir, str(step), 'Prt')
        _ = rf.foam.write_field_file(**Prt_data)
        
        
    def _modify_foam_case1_betann(self, betann, step, foam_dir=None):
        betann_data = self.betann_data_case1
        betann_data['internal_field']['value'] = betann
        if foam_dir is not None:
            betann_data['file'] = os.path.join(foam_dir, str(step), 'betann')
        _ = rf.foam.write_field_file(**betann_data)

    def _modify_foam_case2_betann(self, betann, step, foam_dir=None):
        betann_data = self.betann_data_case2
        betann_data['internal_field']['value'] = betann
        if foam_dir is not None:
            betann_data['file'] = os.path.join(foam_dir, str(step), 'betann')
        _ = rf.foam.write_field_file(**betann_data)

def _get_dadg(tensors, tke):
    tke = np.expand_dims(np.squeeze(tke), axis=(1, 2))
    return 2.0*tke*tensors

def _clean_foam(foam_dir):
    bash_command = './clean > /dev/null'
    bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
    return subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)

# def _run_foam_init(foam_dir, iteration, timeprecision):
#     bash_command = './run > /dev/null'
#     bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
#     return subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)

def _run_foam(foam_dir, iteration, timeprecision):

    # run foam
    solver = 'hisa'
    logfile = os.path.join(foam_dir, solver + '.log')
    bash_command = f'{solver} -case {foam_dir} > {logfile}'
    subprocess.call(bash_command, shell=True)

    logfile = os.path.join(foam_dir, 'yplus.log')
    bash_command = f"hisa -postProcess -func yPlus -case {foam_dir}" + \
         f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)

    logfile = os.path.join(foam_dir, 'wallHeatFlux.log')
    bash_command = f"hisa -postProcess -func wallHeatFlux -case {foam_dir}" + \
         f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)
    
    
    logfile = os.path.join(foam_dir, 'log.R')
    bash_command = f"{solver} -postProcess -func 'turbulenceFields(R)' -case {foam_dir}" + \
         f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)

    # bash_command = './run > /dev/null'
    # bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
    # process = subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)
    if iteration > 100:
        delf = os.path.join(foam_dir, f'{iteration-100:d}')
        shutil.rmtree(delf)
    
    # move results from i to i-1 directory
    logfile = os.path.join(foam_dir, 'sample.log')
    bash_command = f"postProcess -func 'sampleDict1' -case {foam_dir}" + \
         f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)
    
    logfile = os.path.join(foam_dir, 'sample.log')
    bash_command = f"postProcess -func 'sampleDict' -case {foam_dir}" + \
         f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)
    
def writeInput(foam_dir, iteration, timeprecision):

    # run foam

    logfile = os.path.join(foam_dir, 'log.R')
    bash_command = f"writeFieldsMLr4 -case {foam_dir} > {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)

