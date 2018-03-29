'''
particle filter のクラス
18.03.22
- 観測ノイズは正規分布を想定した尤度計算となっている
'''

import numpy as np
import numpy.random as rd
from scipy import linalg

from utils import array1d, array2d, check_random_state, get_params, \
	preprocess_arguments, check_random_state

class Particle_Filter(object):
	'''
	Particle Filter のクラス

	<Input Variables>
	y, observation [n_time, n_dim_obs] {numpy-array, float}
		: observation y
		観測値 [時間軸,観測変数軸]
	initial_mean [n_dim_sys] {float} 
		: initial state mean
		初期状態分布の期待値 [状態変数軸]
	initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
        : initial state covariance
        初期状態分布の共分散行列[状態変数軸，状態変数軸]
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
	dtype {np.dtype} : numpy dtype (numpy のデータ型)
	seed {int} : random seed (ランダムシード)
	'''
	def __init__(self, observation = None, transition_functions = None,
				observation_matrices = None, initial_mean = None,
				initial_covariance = None,
				transition_noise = None, observation_covariance = None,
				n_particle = 100, n_dim_sys = None, n_dim_obs = None,
				dtype = np.float32, seed = 10) :

		# 次元数をチェック，欠測値のマスク処理
		self.y = self._parse_observations(observation)

		# 次元決定
		self.n_dim_sys = self._determine_dimensionality(
			[(initial_mean, array1d, -1),
			(initial_covariance, array2d, -2)],
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

		# initial_covariance None -> np.eye
		if initial_covariance is None:
			self.initial_covariance = np.eye(self.n_dim_sys, dtype = dtype)
		else:
			self.initial_covariance = initial_covariance.astype(dtype)

		self.n_particle = n_particle
		np.random.seed(seed)
		self.dtype = dtype
		self.log_likelihood = - np.inf
	

	# likelihood for normal distribution
	# 正規分布の尤度 (カーネルのみ)
	def _norm_likelihood(self, y, mean, covariance):
		'''
		y [n_dim_obs] {numpy-array, float} 
			: observation
			観測 [観測変数軸]
		mean [n_particle, n_dim_obs] {numpy-array, float}
			: mean of normal distribution
			各粒子に対する正規分布の平均 [粒子軸，観測変数軸]
		covariance [n_dim_obs, n_dim_obs] {numpy-array, float}
			: covariance of normal distribution
			正規分布の共分散 [観測変数軸]
		'''
		Y = np.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)
		Y.T[:] = y
		return np.exp((- 0.5 * (Y - mean).T @ linalg.pinv(covariance) @ (Y - mean))[:, 0])
	

	# log likelihood for normal distribution
	# 正規分布の対数尤度 (カーネルのみ)
	def _log_norm_likelihood(self, y, mean, covariance) :
		'''
		y [n_dim_obs] {numpy-array, float} 
			: observation
			観測 [観測変数軸]
		mean [n_particle, n_dim_obs] {numpy-array, float}
			: mean of normal distribution
			各粒子に対する正規分布の平均 [粒子軸，観測変数軸]
		covariance [n_dim_obs, n_dim_obs] {numpy-array, float}
			: covariance of normal distribution
			正規分布の共分散 [観測変数軸]
		'''
		Y = np.zeros((self.n_dim_obs, self.n_particle), dtype = self.dtype)
		Y.T[:] = y
		return (- 0.5 * (Y - mean).T @ linalg.pinv(covariance) @ (Y - mean)).diagonal()


	# 経験分布の逆写像
	def _emperical_cummulative_inv(self, w_cumsum, idx, u):
		if np.any(w_cumsum < u) == False:
			return 0
		k = np.max(idx[w_cumsum < u])
		return k + 1
		

	# 通常のリサンプリング
	def _resampling(self, weights):
		'''
		通常のリサンプリング (standard resampling method)
		'''
		w_cumsum = np.cumsum(weights)

		# labelの生成
		idx = np.asanyarray(range(self.n_particle))

		# サンプリングしたkのリスト格納場所
		k_list = np.zeros(self.n_particle, dtype = np.int32)
		
		# 一様分布から重みに応じてリサンプリングする添え字を取得
		for i, u in enumerate(rd.uniform(0, 1, size = self.n_particle)):
			k = self._emperical_cummulative_inv(w_cumsum, idx, u)
			k_list[i] = k
		return k_list


	# 層化リサンプリング
	def _stratified_resampling(self, weights):
		"""
		層化リサンプリング (stratified resampling method)
		"""
		idx = np.asanyarray(range(self.n_particle))
		u0 = rd.uniform(0, 1 / self.n_particle)
		u = [1 / self.n_particle*i + u0 for i in range(self.n_particle)]
		w_cumsum = np.cumsum(weights)
		k = np.asanyarray([self._emperical_cummulative_inv(w_cumsum, idx, val) for val in u])
		return k
	

	# filtering
	def filter(self):
		'''
		T {int} : 時系列の長さ，length of y
		x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_pred regarding to particles at time t
			時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
		x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_filt regarding to particles
			時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]

		x_pred [n_dim_sys, n_particle]
			: hidden state at time t given observations for each particle
			状態変数の予測アンサンブル [状態変数軸，粒子軸]
		x_filt [n_particle, n_dim_sys] {numpy-array, float}
			: hidden state at time t given observations for each particle
			状態変数のフィルタアンサンブル [状態変数軸，粒子軸]

		w [n_particle] {numpy-array, float}
			: weight lambda of each particle
			各粒子の尤度 [粒子軸]
		v [n_dim_sys, n_particle] {numpy-array, float}
			: system noise particles
			各時刻の状態ノイズ [状態変数軸，粒子軸]
		k [n_particle] {numpy-array, float}
			: index number for resampling
			各時刻のリサンプリングインデックス [粒子軸]
		'''

		# 時系列の長さ, number of time-series data
		T = len(self.y)
		
		# initial filter, prediction
		self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
		self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
		x_pred = np.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)

		# initial distribution
		x_filt = np.multivariate_normal(self.initial_mean, self.initial_covariance, 
			size = self.n_particle).T
		
		# initial setting
		self.x_pred_mean[0] = self.initial_mean
		self.x_filt_mean[0] = self.initial_mean

		for t in range(T):
			print("\r filter calculating... t={}".format(t), end="")
			
			## filter update
			# 一期先予測, prediction
			f = self._last_dims(self.f, t, 1)[0]

			# システムノイズをパラメトリックに発生, raise parametric system noise
			v = self.q[0](*self.q[1], size = self.n_particle).T

			# アンサンブル予測, ensemble prediction
			x_pred = f(*[x_filt, v])

			# mean
			self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 1)
			
			# 欠測値の対処, treat missing values
			if np.any(np.ma.getmask(self.y[t])):
				x_filt = x_pred
			else:
				# log likelihood for each particle for y[t]
				# 対数尤度の計算
				H = self._last_dims(self.H, t)
				R = self._last_dims(self.R, t)
				w = self._log_norm_likelihood(self.y[t], H @ x_pred, R)

				# normalization
				# 重みの正規化
				w = np.exp(w - np.max(w))
				w = w / np.sum(w)

				# resampling
				k = self._stratified_resampling(w)
				x_filt = x_pred[:, k]
			
			# mean
			self.x_filt_mean[t + 1] = np.mean(x_filt, axis = 1)

		
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


	# メモリ節約のため，filter のオーバーラップ
	def smooth(self, lag = 10):
		'''
		lag {int} : ラグ，lag of smoothing
		T {int} : 時系列の長さ，length of y

		x_pred_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_pred regarding to particles at time t
			時刻 t における x_pred の粒子平均 [時間軸，状態変数軸]
		x_filt_mean [n_time+1, n_dim_sys] {numpy-array, float}
			: mean of x_filt regarding to particles at time t
			時刻 t における状態変数のフィルタ平均 [時間軸，状態変数軸]
		x_smooth_mean [n_time, n_dim_sys] {numpy-array, float}
			: mean of x_smooth regarding to particles at time t
			時刻 t における状態変数の平滑化平均 [時間軸，状態変数軸]

		x_pred [n_dim_sys, n_particle]
			: hidden state at time t given observations[:t-1] for each particle
			状態変数の予測アンサンブル [状態変数軸，粒子軸]
		x_filt [n_particle, n_dim_sys] {numpy-array, float}
			: hidden state at time t given observations[:t] for each particle
			状態変数のフィルタアンサンブル [状態変数軸，粒子軸]
		x_smooth [n_time, n_dim_sys, n_particle] {numpy-array, float}
			: hidden state at time t given observations[:t+lag] for each particle
			状態変数の平滑化アンサンブル [時間軸，状態変数軸，粒子軸]

		w [n_particle] {numpy-array, float}
			: weight lambda of each particle
			各粒子の尤度 [粒子軸]
		v [n_dim_sys, n_particle] {numpy-array, float}
			: system noise particles
			各時刻の状態ノイズ [状態変数軸，粒子軸]
		k [n_particle] {numpy-array, float}
			: index number for resampling
			各時刻のリサンプリングインデックス [粒子軸]
		'''

		# 時系列の長さ, number of time-series data
		T = len(self.y)
		
		# initial filter, prediction
		self.x_pred_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
		self.x_filt_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
		self.x_smooth_mean = np.zeros((T + 1, self.n_dim_sys), dtype = self.dtype)
		x_pred = np.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)
		x_filt = np.zeros((self.n_dim_sys, self.n_particle), dtype = self.dtype)
		x_smooth = np.zeros((T + 1, self.n_dim_sys, self.n_particle), dtype = self.dtype)
		
		# initial setting
		self.x_pred_mean[0] = self.initial_mean
		self.x_filt_mean[0] = self.initial_mean
		self.x_smooth_mean[0] = self.initial_mean
		x_filt.T[:] = self.initial_mean

		for t in range(T):
			print("\r filter and smooth calculating... t={}".format(t), end="")
			
			## filter update
			# 一期先予測, prediction
			f = self._last_dims(self.f, t, 1)[0]

			# システムノイズをパラメトリックに発生, raise parametric system noise
			v = self.q[0](*self.q[1], size = self.n_particle).T

			# アンサンブル予測, ensemble prediction
			x_pred = f(*[x_filt, v])

			# mean
			self.x_pred_mean[t + 1] = np.mean(x_pred, axis = 1)
			
			# 欠測値の対処, treat missing values
			if np.any(np.ma.getmask(self.y[t])):
				x_filt = x_pred
			else:
				# log likelihood for each particle for y[t]
				# 対数尤度の計算
				H = self._last_dims(self.H, t)
				R = self._last_dims(self.R, t)
				w = self._log_norm_likelihood(self.y[t], H @ x_pred, R)

				# normalization
				# 重みの正規化
				w = np.exp(w - np.max(w))
				w = w / np.sum(w)

				# resampling
				k = self._stratified_resampling(w)
				x_filt = x_pred[:, k]
			
			# initial smooth value
			x_smooth[t + 1] = x_filt
			
			# mean
			self.x_filt_mean[t + 1] = np.mean(x_filt, axis = 1)

			# smoothing
			if (t > lag - 1) :
				x_smooth[t - lag:t + 1] = x_smooth[t - lag:t + 1, :, k]
			else :
				x_smooth[:t + 1] = x_smooth[:t + 1, :, k]

		# x_smooth_mean
		self.x_smooth_mean = np.mean(x_smooth, axis = 2)


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