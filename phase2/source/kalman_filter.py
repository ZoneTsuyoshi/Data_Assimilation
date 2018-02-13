'''
Kalman Filter のクラス
pykalman を参考にして作成する
pykalman では欠測値に対応できていないので，欠測にも対応できるクラス作成がメインテーマ
-> と思っていたけど，マスク処理で対応している
-> 欠測値(NaN)の場合は自動でマスク処理するようなコードを追加すれば拡張の意義がある
'''


# パッケージのインストール
import math
import numpy as np

# pykalman
import warnings
from scipy import linalg
from utils import array1d, array2d, check_random_state, \
    get_params, log_multivariate_normal_density, preprocess_arguments


class Kalman_Filter(object) :
    '''
    コード上では，pred, filt が0:Tであり，tはtに対応している
    一方，smooth は 0:T-1であり，tはt-1に対応している

    <Input Variables>
    observation [n_time, n_dim_obs] {numpy-array, float}
        : observation y
        観測値[時間軸,観測変数軸]
    initial_mean [n_dim_sys] {float} 
        : initial state mean
        初期状態分布の期待値[状態変数軸]
    initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
        : initial state covariance （初期状態分布の共分散行列[状態変数軸，状態変数軸]）
    transition_matrices [n_time - 1, n_dim_sys, n_dim_sys]  or [n_dim_sys, n_dim_sys]{numpy-array, float}
        : transition matrix from x_{t-1} to x_t
        システムモデルの変換行列[時間軸，状態変数軸，状態変数軸] or [状態変数軸,状態変数軸] (時不変な場合)
    transition_noise_matrices [n_time - 1, n_dim_sys, n_dim_noise] or [n_dim_sys, n_dim_noise] {numpy-array, float}
        : transition noise matrix
        ノイズ変換行列[時間軸，状態変数軸，ノイズ変数軸] or [状態変数軸，ノイズ変数軸]
    observation_matrices [n_time, n_dim_sys, n_dim_obs] or [n_dim_sys, n_dim_obs] {numpy-array, float}
        : observation matrix
        観測行列[時間軸，状態変数軸，観測変数軸] or [状態変数軸，観測変数軸]
    transition_covariance [n_time - 1, n_dim_noise, n_dim_noise] or [n_dim_sys, n_dim_noise] {numpy-array, float}
        : covariance of system noise
        システムノイズの共分散行列[時間軸，ノイズ変数軸，ノイズ変数軸]
    observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
        : covariance of observation noise
        観測ノイズの共分散行列[時間軸，観測変数軸，観測変数軸]
    transition_offsets [n_time - 1, n_dim_sys] or [n_dim_sys] {numpy-array, float} 
        : offsets of system transition model
        システムモデルの切片（バイアス，オフセット）[時間軸，状態変数軸] or [状態変数軸]
    observation_offsets [n_time, n_dim_obs] or [n_dim_obs] {numpy-array, float}
        : offsets of observation model
        観測モデルの切片[時間軸，観測変数軸] or [観測変数軸]
    n_dim_sys {int} : dimension of system variable （システム変数の次元）
    n_dim_obs {int} : dimension of observation variable （観測変数の次元）

    <Variables>
    y : observation
    F : transition_matrices
    G : transition_noise_matrices
    Q : transition_covariance
    b : transition_offsets
    H : observation_matrices
    R : observation_covariance
    d : observation_offsets
    x_pred [n_time+1, n_dim_sys] {numpy-array, float} 
        : mean of prediction distribution
        予測分布の平均 [時間軸，状態変数軸]
    V_pred [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
        : covariance of prediction distribution
        予測分布の共分散行列 [時間軸，状態変数軸，状態変数軸]
    x_filt [n_time+1, n_dim_sys] {numpy-array, float}
        : mean of filtering distribution
        フィルタ分布の平均 [時間軸，状態変数軸]
    V_filt [n_time+1, n_dim_sys, n_dim_sys] {numpy-array, float}
        : covariance of filtering distribution
        フィルタ分布の共分散行列 [時間軸，状態変数軸，状態変数軸]
    x_RTS [n_time, n_dim_sys] {numpy-array, float}
        : mean of RTS smoothing distribution
        固定区間平滑化分布の平均 [時間軸，状態変数軸]
    V_RTS [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
        : covariance of RTS smoothing distribution
        固定区間平滑化の共分散行列 [時間軸，状態変数軸，状態変数軸]

    
    state space model(状態方程式)
        x[t+1] = F[t]x[t] + G[t]v[t]
        y[t+1] = H[t]x[t] + w[t]
        v[t] ~ N(0, Q[t])
        w[t] ~ N(0, R[t])
    '''

    def __init__(self, observation = None, initial_mean = None, initial_covariance = None,
                transition_matrices = None, transition_noise_matrices = None, observation_matrices = None,
                transition_covariance = None, observation_covariance = None,
                transition_offsets = None, observation_offsets = None,
                n_dim_sys = None, n_dim_obs = None) :
        # 次元決定
        if transition_offsets is None :
            self.n_dim_sys = self._determine_dimensionality(
                [(transition_matrices, array2d, -2),
                 (transition_noise_matrices, array2d, -2),
                 (initial_mean, array1d, -1),
                 (initial_covariance, array2d, -2),
                 (observation_matrices, array2d, -1)],
                n_dim_sys
                )
        else :
            self.n_dim_sys = self._determine_dimensionality(
                [(transition_matrices, array2d, -2),
                 (transition_offsets, array1d, -1),
                 (transition_noise_matrices, array2d, -2),
                 (initial_mean, array1d, -1),
                 (initial_covariance, array2d, -2),
                 (observation_matrices, array2d, -1)],
                n_dim_sys
            )

        if observation_offsets is None:
            self.n_dim_obs = self._determine_dimensionality(
                [(observation_matrices, array2d, -2),
                 (observation_covariance, array2d, -2)],
                n_dim_obs
            )
        else :
            self.n_dim_obs = self._determine_dimensionality(
                [(observation_matrices, array2d, -2),
                 (observation_offsets, array1d, -1),
                 (observation_covariance, array2d, -2)],
                n_dim_obs
            )

        self.n_dim_noise = self._determine_dimensionality(
                [(transition_noise_matrices, array2d, -1),
                 (transition_covariance, array2d, -2)]
            )

        # 次元数をチェック，欠測値のマスク処理
        self.y = self._parse_observations(observation)

        self.initial_mean = initial_mean
        self.initial_covariance = initial_covariance
        self.F = transition_matrices
        self.G = transition_noise_matrices
        self.Q = transition_covariance

        # オフセットが未入力であれば，0にしておく
        if transition_offsets is None :
            self.b = np.zeros(self.n_dim_sys)
        else :
            self.b = transition_offsets

        self.H = observation_matrices
        self.R = observation_covariance

        # オフセットが未入力であれば，0にしておく
        if observation_offsets is None :
            self.d = np.zeros(self.n_dim_obs)
        else :
            self.d = observation_offsets


    # filter function (フィルタ値を計算する関数)
    def filter(self) :
        '''
        T {int} : length of data y （時系列の長さ）
        x_pred [n_time, n_dim_sys] {numpy-array, float}
            : mean of hidden state at time t given observations from times [0...t-1]
            時刻 t における状態変数の予測期待値 [時間軸，状態変数軸]
        V_pred [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            : covariance of hidden state at time t given observations from times [0...t-1]
            時刻 t における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
        x_filt [n_time, n_dim_sys] {numpy-array, float}
            : mean of hidden state at time t given observations from times [0...t]
            時刻 t における状態変数のフィルタ期待値 [時間軸，状態変数軸]
        V_filt [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            : covariance of hidden state at time t given observations from times [0...t]
            時刻 t における状態変数のフィルタ共分散 [時間軸，状態変数軸，状態変数軸]
        K [n_time, n_dim_sys, n_dim_obs] {numpy-array, float}
            : Kalman gain matrix for time t [時間軸，状態変数軸，観測変数軸]
            各時刻のカルマンゲイン
        '''
        T = self.y.shape[0]
        self.x_pred = np.zeros((T, self.n_dim_sys))
        self.V_pred = np.zeros((T, self.n_dim_sys, self.n_dim_sys))
        self.x_filt = np.zeros((T, self.n_dim_sys))
        self.V_filt = np.zeros((T, self.n_dim_sys, self.n_dim_sys))
        self.K = np.zeros((T, self.n_dim_sys, self.n_dim_obs))

        # 各時刻で予測・フィルタ計算
        for t in range(T) :
            # 計算している時間を可視化
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting (初期分布)
                self.x_pred[0] = self.initial_mean
                self.V_pred[0] = self.initial_covariance
            else:
                # extract t-1 parameters (時刻t-1のパラメータ取り出す)
                F = self._last_dims(self.F, t - 1, 2)
                G = self._last_dims(self.G, t - 1, 2)
                Q = self._last_dims(self.Q, t - 1, 2)
                b = self._last_dims(self.b, t - 1, 1)

                # predict t distribution (時刻tの予測分布の計算)
                self.x_pred[t] = np.dot(F, self.x_filt[t-1]) + b
                self.V_pred[t] = np.dot(F, np.dot(self.V_filt[t-1], F.T)) + np.dot(G, np.dot(Q, G.T))

            
            # y[t] の何れかがマスク処理されていれば，フィルタリングはカットする
            if np.any(np.ma.getmask(self.y[t])) :
                self.x_filt[t] = self.x_pred[t]
                self.V_filt[t] = self.V_pred[t]
            else :
                # extract t parameters (時刻tのパラメータを取り出す)
                H = self._last_dims(self.H, t, 2)
                R = self._last_dims(self.R, t, 2)
                d = self._last_dims(self.d, t, 1)

                # filtering (フィルタ分布の計算)
                self.K[t] = np.dot(self.V_pred[t], np.dot(H.T, linalg.pinv(np.dot(H, np.dot(self.V_pred[t], H.T)) + R)))
                self.x_filt[t] = self.x_pred[t] + np.dot(self.K[t], self.y[t] - (np.dot(H, self.x_pred[t]) + d))
                self.V_filt[t] = self.V_pred[t] - np.dot(self.K[t], np.dot(H, self.V_pred[t]))
                

    # get predicted value (一期先予測値を返す関数, Filter 関数後に値を得たい時)
    def get_predicted_value(self, dim = None) :
        # filter されてなければ実行
        try :
            self.x_pred[0]
        except :
            self.filter()

        if dim is None:
            return self.x_pred
        elif dim <= self.x_pred.shape[1]:
            return self.x_pred[:, int(dim)]
        else:
            raise ValueError('The dim must be less than ' + self.x_pred.shape[1] + '.')


    # get filtered value (フィルタ値を返す関数，Filter 関数後に値を得たい時)
    def get_filtered_value(self, dim = None) :
        # filter されてなければ実行
        try :
            self.x_filt[0]
        except :
            self.filter()

        if dim is None:
            return self.x_filt
        elif dim <= self.x_filt.shape[1]:
            return self.x_filt[:, int(dim)]
        else:
            raise ValueError('The dim must be less than ' + self.x_filt.shape[1] + '.')


    # RTS smooth function (RTSスムーシングを計算する関数)
    def RTS_smooth(self) :
        '''
        T : length of data y (時系列の長さ)
        x_RTS [n_time, n_dim_sys] {numpy-array, float}
            : mean of hidden state distributions for times [0...n_timesteps-1] given all observations
            時刻 t における状態変数の平滑化期待値 [時間軸，状態変数軸]
        V_RTS [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            : covariances of hidden state distributions for times [0...n_timesteps-1] given all observations
            時刻 t における状態変数の平滑化共分散 [時間軸，状態変数軸，状態変数軸]
        A [n_time - 1, n_dim_sys, n_dim_sys] {numpy-array, float}
            : fixed interval smoothed gain
            各時刻の固定区間平滑化ゲイン [時間軸，状態変数軸，状態変数軸]
        '''

        # filter が実行されていない場合は実行
        try :
            self.x_pred[0]
        except :
            self.filter()

        T = self.y.shape[0]
        self.x_RTS = np.zeros((T, self.n_dim_sys))
        self.V_RTS = np.zeros((T, self.n_dim_sys, self.n_dim_sys))
        self.A = np.zeros((T-1, self.n_dim_sys, self.n_dim_sys))

        self.x_RTS[-1] = self.x_filt[-1]
        self.V_RTS[-1] = self.V_filt[-1]

        # t in [0, T-2] (tが1~Tの逆順であることに注意)
        for t in reversed(range(T - 1)) :
            # 時間を可視化
            print("\r smooth calculating... t={}".format(T - t) + "/" + str(T), end="")

            # 時刻 t のパラメータを取り出す
            F = self._last_dims(self.F, t, 2)

            # 固定区間平滑ゲインの計算
            self.A[t] = np.dot(self.V_filt[t], np.dot(F.T, linalg.pinv(self.V_pred[t + 1])))
            
            # 固定区間平滑化
            self.x_RTS[t] = self.x_filt[t] + np.dot(self.A[t], self.x_RTS[t + 1] - self.x_pred[t + 1])
            self.V_RTS[t] = self.V_filt[t] + np.dot(self.A[t], np.dot(self.V_RTS[t + 1] - self.V_pred[t + 1], self.A[t].T))

            
    # get RTS smoothed value (RTS スムーシング値を返す関数，RTS_Smooth 後に)
    def get_RTS_smoothed_value(self, dim = None) :
        # filter されてなければ実行
        try :
            self.x_RTS[0]
        except :
            self.RTS_smooth()

        if dim is None:
            return self.x_RTS
        elif dim <= self.x_RTS.shape[1]:
            return self.x_RTS[:, int(dim)]
        else:
            raise ValueError('The dim must be less than ' + self.x_RTS.shape[1] + '.')


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



