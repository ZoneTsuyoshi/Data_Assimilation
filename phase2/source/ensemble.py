# install packages
import math
import numpy as np
import numpy.random as rd
import pandas as pd
from utils import array1d, array2d, check_random_state, get_params, preprocess_arguments, check_random_state

class Ensemble_Kalman_Filter(object):
	'''
	Ensemble Kalman Filter のクラス

	<Input Variables>
    y, observation [n_time, n_dim_obs] {numpy-array, float}
        : observation y
        観測値 [時間軸,観測変数軸]
    initial_mean [n_dim_sys] {float} 
        : initial state mean
        初期状態分布の期待値 [状態変数軸]
    f, transition_functions [n_time] {function}
        : transition function from x_{t-1} to x_t
        システムモデルの遷移関数 [時間軸] or []
    H, observation_matrices [n_time, n_dim_sys, n_dim_obs] {function}
        : observation matrices from x_t to y_t
        観測行列 [時間軸，状態変数軸，観測変数軸] or [状態変数軸，観測変数軸]
    q, transition_noise [n_time - 1] {(method, parameters)}
        : transition noise for v_t
        システムノイズの発生方法とパラメータ [時間軸]
    R, observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
        : covariance of observation normal noise
        観測正規ノイズの共分散行列 [時間軸，観測変数軸，観測変数軸]
    n_particle {int} : number of particles (粒子数)
    n_dim_sys {int} : dimension of system variable （システム変数の次元）
    n_dim_obs {int} : dimension of observation variable （観測変数の次元）
    seed {int} : random seed (ランダムシード)


	'''

	def __init__(self, observation = None, transition_functions = None,
				observation_functions = None, initial_mean = None,
				transition_noise = None, observation_covariance = None,
				n_particle = 100, n_dim_sys = None, n_dim_obs = None, seed = 71) :

		# 次元数をチェック，欠測値のマスク処理
        self.y = self._parse_observations(observation)

		# 次元決定
		self.n_dim_sys = self._determine_dimensionality(
            [(initial_mean, array1d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = self._determine_dimensionality(
            [(observation_covariance, array2d, -2)],
            n_dim_obs
        )

        # transition_functions
        # None -> system + noise
        if transition_functions is None:
        	self.f = lambda x, v: x + v
        else:
        	self.f = transition_functions
        
        # observation_matrices
        # None -> np.eye
        if observation_matrices is None:
        	self.H = np.eye(self.n_dim_obs, self.n_dim_sys)
        else:
        	self.H = observation_matrices

    	# transition_noise
    	# None -> standard normal distribution
    	if transition_noise is None:
    		self.q = (rd.normal, [0, 1])
    	else:
    		self.q = transition_noise

        # observation_covariance
        # None -> np.eye
		if observation_covariance is None:
			self.R = np.eye(self.n_dim_obs)
		else:
			self.R = observation_covariance

    	# initial_mean None -> np.zeros
    	if initial_mean is None:
    		self.initial_mean = np.zeros(self.n_dim_sys)
		else:
			self.initial_mean = initial_mean

		self.n_particle = n_particle
		self.seed = seed


	# filtering step
	def filter(self, y = self.y, n_particle = self.n_particle):
		'''
        y [n_time, n_dim_obs]: observation, 観測 y 
        T {int} : length of data y （時系列の長さ）
        x_pred [n_time, n_particle, n_dim_sys] {numpy-array, float}
            : hidden state at time t given observations from times [0...t-1] for each particle
            時刻 t における状態変数の予測期待値 [時間軸，粒子軸，状態変数軸]
        x_pred_mean [n_time, n_dim_sys] {numpy-array, float}
        	: mean of x_pred regarding to particles at time t
        	時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
    	x_pred_center [n_particle, n_dim_sys] {numpy-array, float}
    		: centering of x_pred
    		時刻 t における x_pred の中心化 [粒子軸，状態変数軸]
        V_pred [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            : covariance of hidden state at time t given observations from times [0...t-1]
            時刻 t における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
        w_ensemble [n_time, n_particle, n_dim_obs] {numpy-array, float}
        	: observation noise ensemble
        	各時刻における観測ノイズのアンサンブル [時間軸，粒子軸，観測変数軸]
    	w_ensemble_mean [n_time, n_dim_obs] {numpy-array, float}
    		: mean of w_ensemble regarding to particles
    		粒子に関する w_ensemble のアンサンブル平均 [時間軸，観測変数軸]
		w_ensemble_center [n_time, n_particle, n_dim_obs] {numpy-array, float}
			: centering of w_ensemble
			w_ensemble の中心化 [時間軸，粒子軸，観測変数軸]
        x_filt [n_time, n_particle, n_dim_sys] {numpy-array, float}
            : hidden state at time t given observations from times [0...t] for each particle
            時刻 t における状態変数のフィルタ期待値 [時間軸，粒子軸，状態変数軸]
        K [n_time, n_dim_sys, n_dim_obs] {numpy-array, float}
            : Kalman gain matrix for time t [時間軸，状態変数軸，観測変数軸]
            各時刻のカルマンゲイン
        '''

        # マスク処理，次元確認
        y = self._parse_observations(y)
        T = y.shape[0]

        # 配列定義
        self.x_pred = np.zeros((T, n_particle, self.n_dim_sys))
        self.x_pred_mean = np.zeros((T, self.n_dim_sys))
        self.x_pred_center = np.zeros((n_particle, self.n_dim_sys))
        self.V_pred = np.zeros((T, self.n_dim_sys, self.n_dim_sys))
        self.w_ensemble = np.zeros((T, n_particle, self.n_dim_obs))
        self.w_ensemble_mean = np.zeros((T, self.n_dim_obs))
        self.w_ensemble_center = np.zeros((T, n_particle, self.n_dim_obs))
        self.x_filt = np.zeros((T, n_particle, self.n_dim_sys))
        self.K = np.zeros((T, self.n_dim_sys, self.n_dim_obs))

        # 各時刻で予測・フィルタ計算
        for t in range(T):
        	# 計算している時間を可視化
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            # 0 の時は initial_mean
            if t==0:
            	self.x_pred[0] = self.initial_mean
            	self.x_pred_mean[0] = self.initial_mean
            	self.x_filt[0] = self.initial_mean

            for i in range(n_particle):






	# parse observations (観測変数の次元チェック，マスク処理)
    def _parse_observations(self, obs):
        """Safely convert observations to their expected format"""
        obs = np.ma.atleast_2d(obs)

        # 2軸目の方が大きい場合は，第1軸と第2軸を交換
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T

        # 欠測値をマスク処理
        obs = np.ma.array(obs, mask = np.isnan(obs))
        return obs


	# determine dimensionality function (次元決定関数)
	def _determine_dimensionality(self, variables, default = None):
	    """Derive the dimensionality of the state space
	    Parameters (入力変数)
	    ----------
	    variables : list of ({None, array}, conversion function, index)
	        variables, functions to convert them to arrays, and indices in those
	        arrays to derive dimensionality from.
	        (配列，時間軸を除いた軸数，対応する次元のインデックス)を入れる
	        望ましい軸数より1多い場合，最初の軸が時間軸であることがわかる

	    default : {None, int}
	        default dimensionality to return if variables is empty
	        デフォルト次元が設定されていたら int 値，そうでなければ None
	    
	    Returns
	    -------
	    dim : int
	        dimensionality of state space as derived from variables or default.
	        状態空間モデルの次元数を返す
	    """

	    # gather possible values based on the variables
	    # 各変数の候補次元を集める
	    candidates = []
	    for (v, converter, idx) in variables:
	        if v is not None:
	            v = converter(v)
	            candidates.append(v.shape[idx])

	    # also use the manually specified default
	    # 人為的にデフォルト値が定まっていればそれも候補次元とする
	    if default is not None:
	        candidates.append(default)

	    # ensure consistency of all derived values
	    # 各処理の次元数の一致確認
	    if len(candidates) == 0:
	        return 1
	    else:
	        # 候補次元が一致しなければ ValueError を raise する
	        if not np.all(np.array(candidates) == candidates[0]):
	            print(candidates)
	            raise ValueError(
	                "The shape of all " +
	                "parameters is not consistent.  " +
	                "Please re-check their values."
	            )
	        return candidates[0]


    # last dim (各時刻におけるパラメータを決定する関数)
    def _last_dims(self, X, t, ndims = 2):
        """Extract the final dimensions of `X`
        Extract the final `ndim` dimensions at index `t` if `X` has >= `ndim` + 1
        dimensions, otherwise return `X`.
        Parameters
        ----------
        X : array with at least dimension `ndims`
        t : int
            index to use for the `ndims` + 1th dimension
        ndims : int, optional
            number of dimensions in the array desired

        Returns
        -------
        Y : array with dimension `ndims`
            the final `ndims` dimensions indexed by `t`
        """
        X = np.asarray(X)
        if len(X.shape) == ndims + 1:
            return X[t]
        elif len(X.shape) == ndims:
            return X
        else:
            raise ValueError(("X only has %d dimensions when %d" +
                    " or more are required") % (len(X.shape), ndims))
