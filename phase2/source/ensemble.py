# ensemble kalman filter
'''
18.03.22
- change evensen formulation
18.03.26
- add truncated svd
	- if you use truncated svd,
	n_dim_obs must be > n_particle
18.03.27
- add pypropack svdp
	- if you use this,
	you must install pypropack (https://github.com/jakevdp/pypropack)
	- before install pypropack, you must install gfortran or gcc,
	because pypropack wrap fortran source.
'''


# install packages
import math

import numpy as np
import numpy.random as rd
import pandas as pd

from scipy import linalg

from utils import array1d, array2d, check_random_state, get_params, \
	preprocess_arguments, check_random_state
from utils_filter import _parse_observations, _last_dims, \
    _determine_dimensionality


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
		サイズは指定できる形式
	R, observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
		: covariance of observation normal noise
		観測正規ノイズの共分散行列 [時間軸，観測変数軸，観測変数軸]
	n_particle {int} : number of particles (粒子数)
	n_dim_sys {int} : dimension of system variable （システム変数の次元）
	n_dim_obs {int} : dimension of observation variable （観測変数の次元）
	n_iter {int} : iteration number of randomized truncated SVD
		 (TruncatedSVD のイテレーション数)
	dtype {np.dtype} : numpy dtype (numpy のデータ型)
	seed {int} : random seed (ランダムシード)
	'''

	def __init__(self, observation = None, transition_functions = None,
				observation_matrices = None, initial_mean = None,
				transition_noise = None, observation_covariance = None,
				n_particle = 100, n_dim_sys = None, n_dim_obs = None,
				dtype = np.float32, seed = 10) :

		# 次元数をチェック，欠測値のマスク処理
		self.y = _parse_observations(observation)

		# 次元決定
		self.n_dim_sys = _determine_dimensionality(
			[(initial_mean, array1d, -1)],
			n_dim_sys
		)

		self.n_dim_obs = _determine_dimensionality(
			[(observation_covariance, array2d, -2)],
			n_dim_obs
		)

		# transition_functions
		# None -> system + noise
		if transition_functions is None:
			self.f = [lambda x, v: x + v]
		else:
			self.f = transition_functions

		# observation_matrices
		# None -> np.eye
		if observation_matrices is None:
			self.H = np.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
		else:
			self.H = observation_matrices.astype(dtype)

		# transition_noise
		# None -> standard normal distribution
		if transition_noise is None:
			self.q = (rd.multivariate_normal,
				[np.zeros(self.n_dim_sys, dtype = dtype),
				np.eye(self.n_dim_sys, dtype = dtype)])
		else:
			self.q = transition_noise

		# observation_covariance
		# None -> np.eye
		if observation_covariance is None:
			self.R = np.eye(self.n_dim_obs, dtype = dtype)
		else:
			self.R = observation_covariance.astype(dtype)

		# initial_mean None -> np.zeros
		if initial_mean is None:
			self.initial_mean = np.zeros(self.n_dim_sys, dtype = dtype)
		else:
			self.initial_mean = initial_mean.astype(dtype)

		self.n_particle = n_particle
		np.random.seed(seed)
		self.seed = seed
		self.dtype = dtype


	# filtering step
	def filter(self):
		'''
		T {int} : length of data y （時系列の長さ）
		x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_pred regarding to particles at time t
			時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
		V_pred [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
			: covariance of hidden state at time t given observations
			 from times [0...t-1]
			時刻 t における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
		x_filt [n_time+1, n_particle, n_dim_sys] {numpy-array, float}
			: hidden state at time t given observations for each particle
			状態変数のフィルタアンサンブル [時間軸，粒子軸，状態変数軸]
		x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_filt regarding to particles
			時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]
		X5 [n_time, n_dim_sys, n_dim_obs] {numpy-array, float}
			: right operator for filter, smooth calulation
			filter, smoothing 計算で用いる各時刻の右作用行列

		x_pred [n_particle, n_dim_sys] {numpy-array, float}
			: hidden state at time t given observations for each particle
			状態変数の予測アンサンブル [粒子軸，状態変数軸]
		x_pred_center [n_particle, n_dim_sys] {numpy-array, float}
			: centering of x_pred
			x_pred の中心化 [粒子軸，状態変数軸]
		w_ensemble [n_particle, n_dim_obs] {numpy-array, float}
			: observation noise ensemble
			観測ノイズのアンサンブル [粒子軸，観測変数軸]
		Inovation [n_dim_obs, n_particle] {numpy-array, float}
			: Inovation from observation [観測変数軸，粒子軸]
			観測と予測のイノベーション
		'''

		# 時系列の長さ, lenght of time-series
		T = self.y.shape[0]

		## 配列定義, definition of array
		# 時刻0における予測・フィルタリングは初期値, initial setting
		self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
		self.x_filt = np.zeros((T + 1, self.n_particle, self.n_dim_sys),
			 dtype = self.dtype)
		self.x_filt[0, :] = self.initial_mean
		self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
		self.X5 = np.zeros((T + 1, self.n_particle, self.n_particle),
			 dtype = self.dtype)
		self.X5[:] = np.eye(self.n_particle, dtype = self.dtype)

		# 初期値のセッティング, initial setting
		self.x_pred_mean[0] = self.initial_mean
		self.x_filt_mean[0] = self.initial_mean

		# initial setting
		x_pred = np.zeros((self.n_particle, self.n_dim_sys), dtype = self.dtype)
		x_pred_center = np.zeros((T + 1, self.n_particle, self.n_dim_sys),
			 dtype = self.dtype)
		w_ensemble = np.zeros((self.n_particle, self.n_dim_obs), dtype = self.dtype)

		# イノベーション, observation inovation
		Inovation = np.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)

		# 各時刻で予測・フィルタ計算, prediction and filtering
		for t in range(T):
			# 計算している時間を可視化, visualization for calculation
			print("\r filter calculating... t={}".format(t+1) + "/" + str(T), end="")

			## filter update
			# 一期先予測, prediction
			f = _last_dims(self.f, t, 1)[0]

			# システムノイズをパラメトリックに発生, raise parametric system noise
			v = self.q[0](*self.q[1], size = self.n_particle)

			# アンサンブル予測, ensemble prediction
			x_pred = f(*np.transpose([self.x_filt[t], v], (0, 2, 1))).T


			# x_pred_mean を計算, calculate x_pred_mean
			self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 0)

			# 欠測値の対処, treat missing values
			if np.any(np.ma.getmask(self.y[t])):
				self.x_filt[t + 1] = x_pred
			else:
				# x_pred_center を計算, calculate x_pred_center
				x_pred_center = x_pred - self.x_pred_mean[t + 1]

				# 観測ノイズのアンサンブルを発生, raise observation noise ensemble
				R = _last_dims(self.R, t)
				w_ensemble = rd.multivariate_normal(np.zeros(self.n_dim_obs), R,
					size = self.n_particle)

				# イノベーションの計算
				H = _last_dims(self.H, t)
				Inovation.T[:] = self.y[t]
				Inovation += w_ensemble.T - H @ x_pred.T

				# 特異値分解
				U, s, _ = linalg.svd(H @ x_pred_center.T + w_ensemble.T, False)

				# 右作用行列の計算
				X1 = np.diag(1 / (s * s)) @ U.T
				X2 = X1 @ Inovation
				X3 = U @ X2
				X4 = (H @ x_pred_center.T).T @ X3
				self.X5[t + 1] = np.eye(self.n_particle, dtype = self.dtype) + X4

				# フィルタ分布のアンサンブルメンバーの計算
				self.x_filt[t + 1] = self.X5[t + 1].T @ x_pred

			# フィルタ分布のアンサンブル平均の計算
			self.x_filt_mean[t + 1] = np.mean(self.x_filt[t + 1], axis = 0)


	# get predicted value (一期先予測値を返す関数, Filter 関数後に値を得たい時)
	def get_predicted_value(self, dim = None) :
		# filter されてなければ実行
		try :
			self.x_pred_mean[0]
		except :
			self.filter()

		if dim is None:
			return self.x_pred_mean[1:]
		elif dim <= self.x_pred_mean.shape[1]:
			return self.x_pred_mean[1:, int(dim)]
		else:
			raise ValueError('The dim must be less than '
				 + self.x_pred_mean.shape[1] + '.')


	# get filtered value (フィルタ値を返す関数，Filter 関数後に値を得たい時)
	def get_filtered_value(self, dim = None) :
		# filter されてなければ実行
		try :
			self.x_filt_mean[0]
		except :
			self.filter()

		if dim is None:
			return self.x_filt_mean[1:]
		elif dim <= self.x_filt_mean.shape[1]:
			return self.x_filt_mean[1:, int(dim)]
		else:
			raise ValueError('The dim must be less than ' 
				+ self.x_filt_mean.shape[1] + '.')


	# fixed lag smooth with filter
	def smooth(self, lag = 10):
		'''
		lag {int} : smoothing lag
		x_smooth [n_time+1, n_particle, n_dim_sys] {numpy-array, float}
			: hidden state at time s given observations
			 from times [0...t] for each particle
			時刻 s における状態変数の平滑化値 [時間軸，粒子軸，状態変数軸]
		x_smooth_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_smooth
			時刻 s における平滑化 x_smooth の平均値
		V_smooth [n_dim_sys, n_dim_sys] {numpy-array, float}
			: covariance of hidden state at times 
			given observations from times [0...t]
			時刻 s における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
		'''

		# 時系列の長さ
		T = self.y.shape[0]

		# 配列定義
		x_smooth = np.zeros((self.n_particle, self.n_dim_sys), dtype = self.dtype)
		self.x_smooth_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)

		# 初期値
		x_smooth[0] = self.initial_mean
		self.x_smooth_mean[0] = self.initial_mean

		for t in range(T + 1):
			# 計算している時間を可視化
			print("\r smooth calculating... t={}".format(t+1) + "/" + str(T), end="")
			x_smooth = self.x_filt[t]

			# 平滑化レンジの決定
			if t > T - lag:
				s_range = range(t + 1, T + 1)
			else:
				s_range = range(t + 1, t + lag + 1)

			for s in s_range:
				x_smooth = self.X5[s] @ x_smooth
			
			# smooth_mean の計算
			self.x_smooth_mean[t] = np.mean(x_smooth, axis = 0)


	# get smoothed value
	def get_smoothed_value(self, dim = None) :
		# smooth されてなければ実行
		try :
			self.x_smooth_mean[0]
		except :
			self.smooth()

		if dim is None:
			return self.x_smooth_mean[1:]
		elif dim <= self.x_smooth_mean.shape[1]:
			return self.x_smooth_mean[1:, int(dim)]
		else:
			raise ValueError('The dim must be less than '
			 + self.x_smooth_mean.shape[1] + '.')
