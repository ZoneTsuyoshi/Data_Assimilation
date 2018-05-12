'''
4-D Variational Method (Ajoint Method)
- Fortran code なら TAMC で自動で update_function 作ってくれる
- 4D-var は関数ごと，最適化したい変数ごとにプログラムが異なってくるため，
　汎用的なクラス作成は難しそうである
'''

import math

import numpy as np
import numpy.random as rd


class Adjoint_Method (object) :
    '''
    <Input Variables>
    y [time, n_dim_obs] {numpy-array, float}
    	: observation data, 観測値
    n_epoch {int}: number of epochs for gradient descent, エポック数
    alpha {float} : learning rate of gradient method, 勾配法の学習率
    x_init [n_dim_sys] {float} : initial system x (adjusting parameter), 初期状態変数
    F_init [time] {numpy-array, float} : parameter for outforce F, 外力の初期値
    update_param [n_dim_param] : parameter for update function,
    	 モデル更新関数用パラメータセット
    ad_eval_bg [n_dim_bg, each_bg_dim]
        : parameter for adjoint, evaluation function,
        アジョイント更新関数，評価関数用背景値セット
    ad_param [n_dim_param]
    	: parameter for adjoint update function,
    	アジョイント更新関数用パラメータセット
    eval_param [n_dim_param]
    	: parameter for evaluation function, 評価関数用パラメータセット
    evaluation_function {function}
    	: model evaluation function, 評価関数
    model_update_function {function}
    	: model update function of system variables, 時間方向更新関数
    adjoint_update_function {function}
    	: adjoint update function of adjoint variables lambda, アジョイント更新関数
    
    <Variable>
    n_dim_sys {int} : dimension of system x, 状態変数の次元
    x [time, n_dim_sys] {numpy-array, float} : system state of each time, システム変数
    F_update [time] {numpy-array, float} : parameter for outforce F, 外力パラメータF
    eval_value [n_epoch] {numpy-array, float} :
    	evaluation function value,
    	評価関数の値
    grad_abs [n_epoch] {numpy-array, float} : gradient absolute value, 勾配の大きさ
    ad [n_dim + 1] {float} : adjoint variable lambda, アジョイント変数
    '''
    def __init__(self, y = None, n_epoch = 20, alpha = 0.01, x_init = None,
    			 F_init = None, update_param = None, ad_eval_bg = None,
    			 ad_param = None, eval_param = None,
                 evaluation_function = None,
                 model_update_function = None,
                 adjoint_update_function = None) :
        self.y = y
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.x_init = x_init
        self.n_dim_sys = self.x_init.shape[0]
        self.F_update = F_init
        self.update_param = update_param
        self.ad_eval_bg = ad_eval_bg
        self.ad_param = ad_param
        self.eval_param = eval_param
        self.evaluation_function = evaluation_function
        self.model_update_function = model_update_function
        self.adjoint_update_function = adjoint_update_function
    
    def simulate(self) :
        '''
        <Variables>
        T {int} : length for time (ステップ数)
        g [time + n_dim_sys] {float} : gradient of evaluation function (勾配)
        d {float} : search vector (勾配法の探索方向)
        '''
        T = len(self.y)
        if self.n_dim_sys == 1:
            self.x = np.zeros(self.n_dim_sys)
        else :
            self.x = np.zeros((T, self.n_dim_sys))
        
        # 初期値の代入
        self.x[0] = self.x_init
        
        # update function の都合上
        self.x[1] = self.x_init
        
        # 評価関数値，勾配の絶対値保存用配列の用意
        self.eval_value = np.zeros(self.n_epoch)
        self.grad_abs = np.zeros(self.n_epoch)
        
        # 0 ~ T-1 次元目：外力パラメータ勾配, T ~ L+T-1 次元目 : 初期値勾配
        g = np.zeros(T + self.n_dim_sys)
        
        # アジョイント変数保存用配列
        self.ad = np.zeros((T, self.n_dim_sys))
        
        for i in range(self.n_epoch) :
            # print epoch number
            print('\r epoch :', i + 1, '/' , self.n_epoch, end = '')
            
            # model forward
            # x[1]が伝播しているのかは多少心配
            self.x[1] = self.model_update_function(
            	self.x[0], self.x[0], self.F_update[1], self.update_param, 0
            	)
            for t in range(1, T - 1) :
                self.x[t + 1] = self.model_update_function(
                	self.x[t], self.x[t-1], self.F_update[t+1], self.update_param, t
                	)
            
            # adjoint backward
            # ad[T-2]が伝播しているのかは多少心配
            self.ad[T - 2], g[T - 2] = self.adjoint_update_function(
            	self.y, self.x, self.ad[T-1], self.ad[T - 1],
                self.F_update[T-2], self.ad_eval_bg, self.ad_param, 0
                )
            for t in list(reversed(range(2, T))) :
                self.ad[t - 2], g[t - 2] = self.adjoint_update_function(
                	self.y, self.x, self.ad[t], self.ad[t-1],
                    self.F_update[t-2], self.ad_eval_bg, self.ad_param, t
                    )
            
            # 初期値勾配
            g[T:] = self.ad[0]
            
            # 勾配の大きさ，評価関数
            self.grad_abs[i] = np.linalg.norm(g)
            self.eval_value[i] = self.evaluation_function(
            	self.x, self.y, self.F_update, self.ad_eval_bg, self.eval_param
            	)
            
            # 取り敢えず勾配降下法
            # TO DO : 共役勾配法，準ニュートン法
            d = - g
            
            # 挿入変数の更新
            self.F_update = self.F_update + self.alpha * d[:T]
            self.x[0] = self.x[0] + self.alpha * d[T:]
        
        # model forward (prediction)
        self.x[1] = self.model_update_function(
        	self.x[0], self.x[0], self.F_update[1], self.update_param, 0
        	)
        for t in range(1, L - 1) :
            self.x[t + 1] = self.model_update_function(
            	self.x[t], self.x[t-1], self.F_update[t+1], self.update_param, t
            	)
    
    # get adjoint value (状態値が得られる)
    def get_adjoint_value (self) :
        return self.x
    
    # get force value (外力が得られる)
    def get_force_value (self) :
        return self.F_update
    
    # get evaluation value (評価関数が得られる)
    def get_evaluation_value (self) :
        return self.eval_value
    
    # get gradient absotute value (勾配の大きさが得られる)
    def get_gradient_absolute_value (self) :
        return self.grad_abs
