# local ensemble transform kalman filter
'''
18.04.02
- ローカルへの変換は隣接行列として実装
'''


# install packages
import math
import itertools

import numpy as np
import numpy.random as rd
import pandas as pd

from scipy import linalg

from utils import array1d, array2d, check_random_state, get_params, \
	preprocess_arguments, check_random_state


class Local_Ensemble_Transform_Kalman_Filter(object):
	'''
	Local Ensemble Transform Kalman Filter のクラス

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
	h, observation_functions [n_time] {function}
		: observation function from x_t to y_t
		観測関数 [時間軸] or []
	q, transition_noise [n_time - 1] {(method, parameters)}
		: transition noise for v_t
		システムノイズの発生方法とパラメータ [時間軸]
		サイズは指定できる形式
	R, observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
		: covariance of observation normal noise
		観測正規ノイズの共分散行列 [時間軸，観測変数軸，観測変数軸]
	A_sys, system_adjacency_matrix [n_dim_sys, n_dim_sys] {numpy-array, int}
		: adjacency matrix of system variables
		システム変数の隣接行列 [状態変数軸，状態変数軸]
	A_obs, observation_adjacency_matrix [n_dim_obs, n_dim_obs] {numpy-array, int}
		: adjacency matrix of system variables
		観測変数の隣接行列 [観測変数軸，観測変数軸]
	rho {float} : multipliative covariance inflating factor
	n_particle {int} : number of particles (粒子数)
	n_dim_sys {int} : dimension of system variable （システム変数の次元）
	n_dim_obs {int} : dimension of observation variable （観測変数の次元）
	dtype {np.dtype} : numpy dtype (numpy のデータ型)
	seed {int} : random seed (ランダムシード)
	'''

	def __init__(self, observation = None, transition_functions = None,
				observation_functions = None, initial_mean = None,
				transition_noise = None, observation_covariance = None,
				system_adjacency_matrix = None, observation_adjacency_matrix = None,
				rho = 1,
				n_particle = 100, n_dim_sys = None, n_dim_obs = None,
				dtype = np.float32, seed = 10, cpu_number = 'all') :

		# 次元数をチェック，欠測値のマスク処理
		self.y = self._parse_observations(observation)

		# 次元決定
		self.n_dim_sys = self._determine_dimensionality(
			[(initial_mean, array1d, -1)],
			n_dim_sys
		)

		self.n_dim_obs = self._determine_dimensionality(
			[(observation, array1d, -1)],
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
		if observation_functions is None:
			self.h = [lambda x : x]
		else:
			self.h = observation_functions

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

		# system_adjacency_matrix None -> np.eye
		if system_adjacency_matrix is None:
			self.A_sys = np.eye(self.n_dim_sys).astype(bool)
		else:
			self.A_sys = system_adjacency_matrix.astype(bool)

		# observation_adjacency_matrix is None -> np.eye
		if observation_adjacency_matrix is None:
			self.A_obs = np.eye(self.n_dim_obs).astype(bool)
		else:
			self.A_obs = observation_adjacency_matrix.astype(bool)

		self.rho = rho
		self.n_particle = n_particle
		np.random.seed(seed)
		self.seed = seed
		self.dtype = dtype
		if cpu_number == 'all':
			self.cpu_number = multi.cpu_count()
		else:
			self.cpu_number = cpu_number


	# filtering step
	def filter(self):
		'''
		T {int} : length of data y （時系列の長さ）
		x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_pred regarding to particles at time t
			時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
		V_pred [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
			: covariance of hidden state at time t given observations from times [0...t-1]
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
		self.x_filt = np.zeros((T + 1, self.n_dim_sys, self.n_particle), dtype = self.dtype)
		self.x_filt[0].T[:] = self.initial_mean
		self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)

		# 初期値のセッティング, initial setting
		self.x_pred_mean[0] = self.initial_mean
		self.x_filt_mean[0] = self.initial_mean

		# イノベーション, observation inovation
		Inovation = np.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)

		# 各時刻で予測・フィルタ計算, prediction and filtering
		for t in range(T):
			# 計算している時間を可視化, visualization for calculation
			print('\r filter calculating... t={}'.format(t+1) + '/' + str(T), end='')

			## filter update
			# 一期先予測, prediction
			f = self._last_dims(self.f, t, 1)[0]

			# システムノイズをパラメトリックに発生, raise parametric system noise
			# sys x particle
			v = self.q[0](*self.q[1], size = self.n_particle).T

			# アンサンブル予測, ensemble prediction
			# sys x particle
			x_pred = f(*[self.x_filt[t], v])

			# x_pred_mean を計算, calculate x_pred_mean
			# time x sys
			self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 1)

			# 欠測値の対処, treat missing values
			if np.any(np.ma.getmask(self.y[t])):
				# time x sys x particle
				self.x_filt[t + 1] = x_pred
			else:
				## Step1 : model space -> observation space
				h = self._last_dims(self.h, t, 1)[0]
				R = self._last_dims(self.R, t)

				# y_background : obs x particle
				y_background = h(x_pred)

				# y_background mean : obs
				y_background_mean = np.mean(y_background, axis = 1)

				# y_background center : obs x particle
				y_background_center = (y_background.T - y_background_mean).T


				## Step2 : calculate for model space
				# x_pred_center : sys x particle
				x_pred_center = (x_pred.T - self.x_pred_mean[t + 1]).T


				# 先ずは素朴に各 grid point に関して行う方法でコードを書く
				for i in range(self.n_dim_sys):
					## Step3 : select data for grid point
					# now, we select surrounding points for each data
					# local_sys
					x_pred_mean_local = self.x_pred_mean[t, self.A_sys[i]]

					# local_sys x particle
					x_pred_center_local = x_pred_center[self.A_sys[i]]

					# local_obs
					y_background_mean_local = y_background_mean[self.A_obs[i]]

					# local_obs x particle
					y_background_center_local = y_background_center[self.A_obs[i]]

					# local_obs
					y_local = self.y[t, self.A_obs[i]]

					# local_obs x local_obs
					R_local = R[self.A_obs[i]][:, self.A_obs[i]]


					## Step4 : calculate matrix C
					# R はここでしか使わないので，線型方程式 R C^T=Y を解く方が速いかもしれない
					# R が時不変なら毎回逆行列計算するコスト抑制をしても良い
					# particle x local_obs
					C = y_background_center_local.T @ linalg.pinv(R_local)


					## Step5 : calculate analysis error covariance in ensemble space
					# particle x particle
					analysis_error_covariance = linalg.pinv(
						(self.n_particle - 1) / self.rho * np.eye(self.n_particle) \
						+ C @ y_background_center_local
						)


					## Step6 : calculate analysis weight matrix in ensemble space
					# particle x particle
					analysis_weight_matrix = linalg.sqrtm(
						(self.n_particle - 1) * analysis_error_covariance
						)


					# Step7 : calculate analysis weight ensemble
					# particle
					analysis_weight_mean = analysis_error_covariance @ C @ (
						(y_local - y_background_center_local.T).T
						)

					# analysis_weight_matrix が対称なら転置とる必要がなくなる
					# particle x particle
					analysis_weight_ensemble = (analysis_weight_matrix.T + analysis_weight_mean).T


					## Step8 : calculate analysis system variable in model space
					# 転置が多くて少し気持ち悪い
					# local_sys x particle
					analysis_system = (x_pred_mean_local + (
						x_pred_center_local @ analysis_weight_ensemble
						).T).T


					## Step9 : move analysis result to global analysis
					# time x sys x particle
					self.x_filt[t + 1, i] = analysis_system[len(np.where(self.A_sys[i, :i])[0])]


			# フィルタ分布のアンサンブル平均の計算
			self.x_filt_mean[t + 1] = np.mean(self.x_filt[t + 1], axis = 1)


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
			raise ValueError('The dim must be less than ' + self.x_pred_mean.shape[1] + '.')


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
			raise ValueError('The dim must be less than ' + self.x_filt_mean.shape[1] + '.')


	# parse observations (観測変数の次元チェック，マスク処理)
	def _parse_observations(self, obs):
		'''Safely convert observations to their expected format'''
		obs = np.ma.atleast_2d(obs)

		# 2軸目の方が大きい場合は，第1軸と第2軸を交換
		if obs.shape[0] == 1 and obs.shape[1] > 1:
			obs = obs.T

		# 欠測値をマスク処理
		obs = np.ma.array(obs, mask = np.isnan(obs))
		return obs


	# determine dimensionality function (次元決定関数)
	def _determine_dimensionality(self, variables, default = None):
		'''Derive the dimensionality of the state space
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
		'''

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
					'The shape of all ' +
					'parameters is not consistent.  ' +
					'Please re-check their values.'
				)
			return candidates[0]


	# last dim (各時刻におけるパラメータを決定する関数)
	def _last_dims(self, X, t, ndims = 2):
		'''Extract the final dimensions of `X`
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
		'''
		X = np.asarray(X)
		if len(X.shape) == ndims + 1:
			return X[t]
		elif len(X.shape) == ndims:
			return X
		else:
			raise ValueError(('X only has %d dimensions when %d (time-invariant)' +
					' or %d (time-variant) are required') % (len(X.shape), ndims, ndims + 1))
