# install packages
import math
import numpy as np
import numpy.random as rd
import pandas as pd
from scipy import linalg
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
				observation_matrices = None, initial_mean = None,
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
			self.f = [lambda x, v: x + v]
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
			self.q = (rd.multivariate_normal, [np.zeros(self.n_dim_sys), np.eye(self.n_dim_sys)])
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
	def filter(self, y = None, n_particle = None):
		'''
		y [n_time, n_dim_obs]: observation, 観測 y 
		T {int} : length of data y （時系列の長さ）
		x_pred [n_time+1, n_particle, n_dim_sys] {numpy-array, float}
			: hidden state at time t given observations from times [0...t-1] for each particle
			時刻 t における状態変数の予測期待値 [時間軸，粒子軸，状態変数軸]
		x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_pred regarding to particles at time t
			時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
		x_pred_center [n_time+1, n_particle, n_dim_sys] {numpy-array, float}
			: centering of x_pred
			時刻 t における x_pred の中心化 [時間軸，粒子軸，状態変数軸]
		V_pred [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
			: covariance of hidden state at time t given observations from times [0...t-1]
			時刻 t における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
		w_ensemble [n_time+1, n_particle, n_dim_obs] {numpy-array, float}
			: observation noise ensemble
			各時刻における観測ノイズのアンサンブル [時間軸，粒子軸，観測変数軸]
		w_ensemble_mean [n_time+1, n_dim_obs] {numpy-array, float}
			: mean of w_ensemble regarding to particles
			粒子に関する w_ensemble のアンサンブル平均 [時間軸，観測変数軸]
		w_ensemble_center [n_time+1, n_particle, n_dim_obs] {numpy-array, float}
			: centering of w_ensemble
			w_ensemble の中心化 [時間軸，粒子軸，観測変数軸]
		x_filt [n_time+1, n_particle, n_dim_sys] {numpy-array, float}
			: hidden state at time t given observations from times [0...t] for each particle
			時刻 t における状態変数のフィルタアンサンブル [時間軸，粒子軸，状態変数軸]
		x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_filt regarding to particles
			時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]
		K [n_time+1, n_dim_sys, n_dim_obs] {numpy-array, float}
			: Kalman gain matrix for time t [時間軸，状態変数軸，観測変数軸]
			各時刻のカルマンゲイン
		'''

		# y, n_particle
		if y is None:
			y = self.y
		if n_particle is None:
			n_particle = self.n_particle

		# マスク処理，次元確認
		y = self._parse_observations(y)
		T = y.shape[0]

		# 配列定義
		self._filter_definition(T, n_particle)

		# 各時刻で予測・フィルタ計算
		for t in range(T):
			# 計算している時間を可視化
			print("\r filter calculating... t={}".format(t+1) + "/" + str(T), end="")

			# filter update
			self._filter_update(t, n_particle, y)


	# filter definition
	# smooth でも使うため，関数化しておく
	def _filter_definition(self, T, n_particle):
		# 時刻0における予測・フィルタリングは初期値
		self.x_pred = np.zeros((T + 1, n_particle, self.n_dim_sys))
		self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys))
		self.x_pred_center = np.zeros((T + 1, n_particle, self.n_dim_sys))
		self.x_pred_center_outer = np.zeros((n_particle, self.n_dim_sys, self.n_dim_sys))
		#self.V_pred = np.zeros((T + 1, self.n_dim_sys, self.n_dim_sys))
		self.w_ensemble = np.zeros((T + 1, n_particle, self.n_dim_obs))

		# n_particle が十分大きければ ensemble_mean, center を計算する必要性はない
		#self.w_ensemble_mean = np.zeros((T + 1, self.n_dim_obs))
		#self.w_ensemble_center = np.zeros((T + 1, n_particle, self.n_dim_obs))

		self.x_filt = np.zeros((T + 1, n_particle, self.n_dim_sys))
		self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys))
		#self.K = np.zeros((T + 1, self.n_dim_sys, self.n_dim_obs))
		self.Z = np.zeros((T + 1, n_particle, n_particle))


		# 初期値のセッティング
		self.x_pred[0] = self.initial_mean
		self.x_pred_mean[0] = self.initial_mean
		self.x_filt[0] = self.initial_mean
		self.x_filt_mean[0] = self.initial_mean


	# filter update
	# smooth でも使うため，関数化しておく
	def _filter_update(self, t, n_particle, y):
		# 一期先予測
		f = self._last_dims(self.f, t, 1)[0]

		# システムノイズをパラメトリックに発生
		#v = np.zeros((n_particle, self.n_dim_sys))
		for i in range(n_particle):
			v = self.q[0](*self.q[1])

			# アンサンブル予測
			self.x_pred[t + 1, i] = f(self.x_filt[t, i], v)

		# x_pred_mean を計算
		self.x_pred_mean[t + 1] = np.mean(self.x_pred[t + 1], axis = 0)

		# 欠測値の対処
		if np.any(np.ma.getmask(y[t])):
			self.x_filt[t + 1] = self.x_pred[t + 1]
		else:
			# x_pred_center を計算
			self.x_pred_center[t + 1] = self.x_pred[t + 1] - self.x_pred_mean[t + 1]

			# 保留(使わない) : V_pred を計算
			'''
			for i in range(n_particle):
				self.x_pred_center_outer[i] = np.outer(self.x_pred_center[t + 1, i], self.x_pred_center[t + 1, i])

			self.V_pred[t + 1] = np.mean(self.x_pred_center_outer, axis = 0) * (n_particle / (n_particle - 1))
			'''

			# 観測ノイズのアンサンブルを発生
			R = self._last_dims(self.R, t)
			for i in range(n_particle):
				self.w_ensemble[t + 1, i] = rd.multivariate_normal(np.zeros(self.n_dim_obs), R)

			# カルマンゲインの計算
			H = self._last_dims(self.H, t)
			#HVH_R = np.dot(H, np.dot(self.V_pred[t + 1], H.T)) + R
			HVH_R = np.dot(H, np.dot(self.x_pred_center[t + 1].T,
				np.dot(self.x_pred_center[t + 1], H.T))) + R
			'''保留 : 以前のコード
			self.K[t + 1] = np.dot(self.V_pred[t + 1], np.dot(H.T, linalg.pinv(HVH_R)))
			'''
			Inovation = np.zeros((self.n_dim_obs, n_particle))
			Inovation.T[:] = y[t]
			Inovation += self.w_ensemble[t + 1].T - np.dot(H, self.x_pred[t + 1].T)
			self.Z[t + 1] = np.eye(n_particle) + np.dot(
				self.x_pred_center[t + 1],
				np.dot(H.T, np.dot(linalg.pinv(HVH_R), Inovation))
				)

			# フィルタ分布のアンサンブルメンバーの計算
			self.x_filt[t + 1] = np.dot(self.Z[t + 1].T, self.x_pred[t + 1])
			#self.x_filt[t + 1] = np.dot(self.x_pred[t + 1].T, self.Z[t + 1]).T
			'''保留 : 以前のコード
			for i in range(n_particle):
				self.x_filt[t + 1, i] = self.x_pred[t + 1, i] + np.dot(
					self.K[t + 1], y[t] + self.w_ensemble[t + 1, i] - np.dot(H, self.x_pred[t + 1, i])
					)'''
					

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


	# fixed lag smooth with filter
	def smooth(self, y = None, n_particle = None, lag = 10):
		'''
		lag {int} : smoothing lag
		Z [n_time, n_dim_sys, n_dim_obs] {numpy-array, float}
			: kalman gain for covariance
			smoothing 用のカルマンゲインっぽい積要素
		x_smooth [n_time+1, n_particle, n_dim_sys] {numpy-array, float}
			: hidden state at time s given observations from times [0...t] for each particle
			時刻 s における状態変数の平滑化値 [時間軸，粒子軸，状態変数軸]
		x_smooth_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_smooth
			時刻 s における平滑化 x_smooth の平均値
		V_smooth [n_dim_sys, n_dim_sys] {numpy-array, float}
			: covariance of hidden state at time s given observations from times [0...t]
			時刻 s における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
		'''

		# y, n_particle
		if y is None:
			y = self.y
		if n_particle is None:
			n_particle = self.n_particle

		# マスク処理，次元確認
		y = self._parse_observations(y)
		T = y.shape[0]

		# filter と同様の配列定義
		self._filter_definition(T, n_particle)
		#Z = np.zeros(self.n_dim_sys)
		self.x_smooth = np.zeros((T + 1, n_particle, self.n_dim_sys))
		self.x_smooth_mean = np.zeros((T + 1, self.n_dim_sys))
		#x_smooth_center = np.zeros((T + 1, n_particle, self.n_dim_sys))
		#x_smooth_center_outer = np.zeros((n_particle, self.n_dim_sys, self.n_dim_sys))
		#V_smooth = np.zeros((self.n_dim_sys, self.n_dim_sys))

		# 初期値
		self.x_smooth[0] = self.initial_mean
		self.x_smooth_mean[0] = self.initial_mean

		for t in range(T):
			# 計算している時間を可視化
			print("\r smooth calculating... t={}".format(t+1) + "/" + str(T), end="")

			# filter
			self._filter_update(t, n_particle, y)

			# t+1 の初期 smooth 値
			self.x_smooth[t + 1] = self.x_filt[t + 1]

			''' 保留 : 以前のコード
			R = self._last_dims(self.R, t)
			H = self._last_dims(self.H, t)
			HVH_R = np.dot(H, np.dot(self.V_pred[t + 1], H.T)) + R
			'''

			# 平滑化レンジの決定
			if t < lag + 1:
				s_range = range(1, t + 1)
			else:
				s_range = range(t-lag, t + 1)

			# 平滑化
			if not np.any(np.ma.getmask(y[t])):
				for s in s_range:
					# 観測が得られたら，s_range 区間を t における観測で慣らす
					self.x_smooth[s] = np.dot(self.Z[t + 1].T, self.x_smooth[s])
					''' 保留 : 以前のコード
					self.x_smooth_mean[s] = np.mean(self.x_smooth[s], axis = 0)
					x_smooth_center[s] = self.x_smooth[s] - self.x_smooth_mean[s]

					# particle ごとに外積を取るために for 分回しているが，できれば回したくない
					for i in range(n_particle):
						x_smooth_center_outer[i] = np.outer(x_smooth_center[s, i], self.x_pred_center[t + 1, i])
					V_smooth = np.mean(x_smooth_center_outer, axis = 0) * (n_particle / (n_particle - 1))

					for i in range(n_particle):
						Z = np.dot(
							V_smooth, np.dot(
								H.T, np.dot(
									linalg.pinv(HVH_R),
									y[t] + self.w_ensemble[t + 1, i] - np.dot(H, self.x_pred[t + 1, i]))
								)
							)
						self.x_smooth[s, i] += Z
						'''

		# smooth_mean の計算
		self.x_smooth_mean = np.mean(self.x_smooth, axis = 1)


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
			raise ValueError('The dim must be less than ' + self.x_smooth_mean.shape[1] + '.')


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
			raise ValueError(("X only has %d dimensions when %d (time-invariant)" +
					" or %d (time-variant) are required") % (len(X.shape), ndims, ndims + 1))
