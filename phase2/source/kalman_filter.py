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


# Dimensionality of each Kalman Filter parameter for a single time step
# EM algorithm で使用する可能性のあるパラメータ群は DIM で指定しておく
DIM = {
    'transition_matrices': 2,
    'transition_offsets': 1,
    'observation_matrices': 2,
    'observation_offsets': 1,
    'transition_covariance': 2,
    'observation_covariance': 2,
    'initial_mean': 1,
    'initial_covariance': 2,
}


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
    Q : transition_covariance, transition_noise_matrices
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
                transition_matrices = None, observation_matrices = None,
                transition_covariance = None, observation_covariance = None,
                transition_noise_matrices = None,
                transition_offsets = None, observation_offsets = None,
                em_vars=['transition_covariance', 'observation_covariance',
                    'initial_mean', 'initial_covariance'],
                n_dim_sys = None, n_dim_obs = None) :
        # 次元決定
        if transition_offsets is None :
            if transition_noise_matrices is None :
                self.n_dim_sys = self._determine_dimensionality(
                    [(transition_matrices, array2d, -2),
                     (initial_mean, array1d, -1),
                     (initial_covariance, array2d, -2),
                     (observation_matrices, array2d, -1)],
                    n_dim_sys
                    )
            else :
                self.n_dim_sys = self._determine_dimensionality(
                    [(transition_matrices, array2d, -2),
                     (transition_noise_matrices, array2d, -2),
                     (initial_mean, array1d, -1),
                     (initial_covariance, array2d, -2),
                     (observation_matrices, array2d, -1)],
                    n_dim_sys
                    )
        elif transition_noise_matrices is None :
            self.n_dim_sys = self._determine_dimensionality(
                [(transition_matrices, array2d, -2),
                 (transition_offsets, array1d, -1),
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

        # transition_noise_matrices を設定していない場合は，system と次元を一致させる
        if transition_noise_matrices is None :
            self.n_dim_noise = self._determine_dimensionality(
                    [(transition_covariance, array2d, -2),
                    self.n_dim_sys]
                )
            transition_noise_matrices = np.eye(self.n_dim_sys)
        else :
            self.n_dim_noise = self._determine_dimensionality(
                    [(transition_noise_matrices, array2d, -1),
                     (transition_covariance, array2d, -2)]
                )

        # 次元数をチェック，欠測値のマスク処理
        self.y = self._parse_observations(observation)

        self.initial_mean = initial_mean
        self.initial_covariance = initial_covariance
        self.F = transition_matrices
        self.Q = self._calc_transition_covariance(transition_noise_matrices, transition_covariance)

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

        # EM algorithm で最適化するパラメータ群
        self.em_vars = em_vars


    # filter function (フィルタ値を計算する関数)
    def filter(self, y = None) :
        '''
        y [n_time, n_dim_obs]: observation, 観測 y 
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

        # y が未入力であればクラス作成時に入力された y を用いる
        if y is None:
            y = self.y

        # マスク処理，次元確認
        y = self._parse_observations(y)

        T = y.shape[0]
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
                Q = self._last_dims(self.Q, t - 1, 2)
                b = self._last_dims(self.b, t - 1, 1)

                # predict t distribution (時刻tの予測分布の計算)
                self.x_pred[t] = np.dot(F, self.x_filt[t-1]) + b
                self.V_pred[t] = np.dot(F, np.dot(self.V_filt[t-1], F.T)) + Q

            
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
                self.x_filt[t] = self.x_pred[t] + np.dot(self.K[t], y[t] - (np.dot(H, self.x_pred[t]) + d))
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
    def RTS_smooth(self, y = None) :
        '''
        y [n_time, n_dim_obs] : observation, 観測 y 
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

        # y が入力されていなければクラス作成時に入力した y を用いる
        if y is None:
            y = self.y

        # マスク処理，次元確認
        y = self._parse_observations(y)

        # filter が実行されていない場合は実行
        try :
            self.x_pred[0]
        except :
            self.filter()

        T = y.shape[0]
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


    # em algorithm
    def em(self, y = None, n_iter = 10, em_vars = None):
        """Apply the EM algorithm
        Apply the EM algorithm to estimate all parameters specified by `em_vars`.
        em_vars に入れられているパラメータ集合について EM algorithm を用いて最適化する．
        ただし，各遷移パラメータは時不変であるとする．

        Parameters
        ----------
        y : [n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `y[t]`'s components is masked, then
            `y[t]` will be treated as a missing observation.
        n_iter : int, optional
            number of EM iterations to perform
            EM algorithm におけるイテレーション回数
        em_vars : iterable of strings or 'all'
            variables to perform EM over.  Any variable not appearing here is
            left untouched.
            EM algorithm で最適化するパラメータ群
        """

        # y が未入力であればクラス作成時に入力した観測yをEMで回す
        if y is None:
            y = self.y

        # 欠測値処理，次元確認
        y = self._parse_observations(y)

        # Create dictionary of variables not to perform EM on
        # em_vars が入力されなかったらクラス作成時に入力した em_vars を使用
        if em_vars is None:
            em_vars = self.em_vars

        # em_vars を setting
        if em_vars == 'all':
            # all だったら既存値が何も与えられていない
            given = {}
        else:
            given = {
                'transition_matrices': self.F,
                'observation_matrices': self.H,
                'transition_offsets': self.b,
                'observation_offsets': self.d,
                'transition_covariance': self.Q,
                'observation_covariance': self.R,
                'initial_mean': self.initial_mean,
                'initial_covariance': self.initial_covariance
            }
            # em_vars に要素がある場合，given dictionary から削除
            em_vars = set(em_vars)
            for k in list(given.keys()):
                if k in em_vars:
                    given.pop(k)

        # If a parameter is time varying, print a warning
        # DIM に含まれているパラメータだけ get_params で取得して考える
        # get_params で取得するとき，__init__ 関数の入力値を考える
        # given に含まれていないパラメータが時不変でなければ警告を出す
        for (k, v) in get_params(self).items():
            if k in DIM and (not k in given) and len(v.shape) != DIM[k]:
                warn_str = (
                    '{0} has {1} dimensions now; after fitting, '
                    + 'it will have dimension {2}'
                ).format(k, len(v.shape), DIM[k])
                warnings.warn(warn_str)

        # Actual EM iterations
        # EM algorithm の計算
        for i in range(n_iter):
            # E step
            self.filter(y)
            self.RTS_smooth(y)
            
            # sigma pair smooth
            # 時刻 t,t-1 のシステムの共分散遷移
            self.V_pair(y)

            # M step
            self._calc_em(y, given = given)
        return self


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


    # calculate transition covariance (Q_new = GQG^T の計算をしておく)
    def _calc_transition_covariance(self, G, Q) :
        if G.ndim == 2:
            GT = G.T
        elif G.ndim == 3:
            GT = G.transpose(0,2,1)
        else:
            raise ValueError('The ndim of transition_noise_matrices should be 2 or 3, but your input is ' + str(G.ndim) + '.')
        if Q.ndim == 2 or Q.ndim == 3:
            return np.matmul(G, np.matmul(Q, GT))
        else:
            raise ValueError('The ndim of transition_covariance should be 2 or 3, but your input is ' + str(Q.ndim) + '.')
    

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


    # sigma pair smooth 計算
    def _sigma_pair_smooth(self, y = None):
        '''
        y [n_time, n_dim_obs] : observation, 観測 y
        T {int} : length of y (時系列の長さ) 
        V_pair [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            : Covariance between hidden states at times t and t-1 for t = [1...n_timesteps-1].  Time 0 is ignored.
            時刻t,t-1間の状態の共分散．0は無視する
        '''

        # y が入力されていなければクラス作成時に入力した y を用いる
        if y is None:
            y = self.y

        # マスク処理，次元確認
        y = self._parse_observations(y)

        # 時系列の長さ
        T = y.shape[0]

        # pairwise covariance
        self.V_pair = np.zeros((T, self.n_dim_sys, self.n_dim_sys))

        # t in [1, T-1]
        # 文献の計算と甚だ違うが，pykalman ではこうなっている
        for t in range(1, T):
            self.V_pair = np.dot(self.V_RTS[t], self.K[t-1].T)


    # calculate parameters by EM algorithm
    # EM algorithm を用いたパラメータ計算
    def _calc_em(self, y, given = {}):
        '''
        y [n_time, n_dim_obs] {masked-numpy-array, float} : observation y
        T {int} : length of observation y
        '''

        # length of y
        T = y.shape[0]

        # observation_matrices を最初に更新
        if 'observation_matrices' not in given:
            '''math
            y_t : observation, d_t : observation_offsets
            x_t : system, H : observation_matrices

            H &= ( \sum_{t=0}^{T-1} (y_t - d_t) \mathbb{E}[x_t]^T )
             ( \sum_{t=0}^{T-1} \mathbb{E}[x_t x_t^T] )^-1
            '''
            res1 = np.zeros((self.n_dim_obs, self.n_dim_sys))
            res2 = np.zeros((self.n_dim_sys, self.n_dim_sys))

            for t in range(T):
                # 欠測がない y_t に関して
                if not np.any(np.ma.getmask(observations[t])):
                    d = self._last_dims(self.d, t, 1)
                    # それぞれの要素毎の積を取りたいので，outer(外積)を使う
                    res1 += np.outer(y[t] - d, self.x_RTS[t])
                    res2 += self.V_RTS[t] + np.outer(self.x_RTS[t], self.x_RTS[t])

            # observation_matrices (H) を更新
            self.H = np.dot(res1, res2)

        # 次に observation_covariance を更新
        if 'observation_covariance' not in given:
            '''math
            R : observation_covariance, H_t : observation_matrices,
            x_t : system, d_t : observation_offsets, y_t : observation

            R &= \frac{1}{T} \sum_{t=0}^{T-1}
                [y_t - H_t \mathbb{E}[x_t] - d_t]
                    [y_t - H_t \mathbb{E}[x_t] - d_t]^T
                + H_t Var(x_t) H_t^T
            '''

            # 計算補助
            res = np.zeros((self.n_dim_obs, self.n_dim_obs))
            n_obs = 0

            for t in range(T):
                if not np.any(np.ma.getmask(y[t])):
                    H = self._last_dims(self.H, t)
                    d = self._last_dims(self.d, t, 1)
                    err = y[t] - np.dot(H, self.x_RTS[t]) - d
                    res += np.outer(err, err) + np.dot(H, np.dot(self.x_RTS[t], H.T))
                    n_obs += 1
            
            # 観測が1回でも確認できた場合
            if n_obs > 0:
                self.R = (1.0 / n_obs) * res
            else:
                self.R = res

        # 次に transition_matrices の更新
        if 'transition_matrices' not in given:
            '''math
            F : transition_matrices, x_t : system,
            b_t : transition_offsets

            F &= ( \sum_{t=1}^{T-1} \mathbb{E}[x_t x_{t-1}^{T}]
                - b_{t-1} \mathbb{E}[x_{t-1}]^T )
             ( \sum_{t=1}^{T-1} \mathbb{E}[x_{t-1} x_{t-1}^T] )^{-1}
             '''
            #計算補助
            res1 = np.zeros((self.n_dim_sys, n_dim_sys))
            res2 = np.zeros((self.n_dim_sys, n_dim_sys))
            for t in range(1, T):
                b = self._last_dims(self.b, t - 1, 1)
                res1 += self.V_pair[t] + np.outer(self.x_RTS[t], self.x_RTS[t - 1])
                res1 -= np.outer(b, self.x_RTS[t - 1])            
                res2 += self.V_RTS[t - 1] + np.outer(self.x_RTS[t - 1], self.x_RTS[t - 1])

            self.F = np.dot(res1, linalg.pinv(res2))


        # 次に transition_covariance の更新
        if 'transition_covariance' not in given:
            '''math
            Q : transition_covariance, x_t : system, 
            b_t : transition_offsets, F_t : transition_matrices

            Q &= \frac{1}{T-1} \sum_{t=0}^{T-2}
                (\mathbb{E}[x_{t+1}] - A_t \mathbb{E}[x_t] - b_t)
                    (\mathbb{E}[x_{t+1}] - A_t \mathbb{E}[x_t] - b_t)^T
                + F_t Var(x_t) F_t^T + Var(x_{t+1})
                - Cov(x_{t+1}, x_t) F_t^T - F_t Cov(x_t, x_{t+1})
            '''
            # 計算補助
            res = np.zeros((self.n_dim_sys, self.n_dim_sys))

            # 少し回りくどい計算をしているように思える
            for t in range(T - 1):
                F = self._last_dims(self.F, t)
                b = self._last_dims(self.b, t, 1)
                err = self.x_RTS[t + 1] - np.dot(F, self.x_RTS[t]) - b
                Vt1t_A = np.dot(self.V_pair[t + 1], F.T)
                res += (
                    np.outer(err, err)
                    + np.dot(F, np.dot(self.x_RTS[t], F.T))
                    + self.x_RTS[t + 1]
                    - Vt1t_A - Vt1t_A.T
                )

            return (1.0 / (T - 1)) * res


        # 次に initial_mean の更新
        if 'initial_mean' not in  given:


        # 次に initial_covariance の更新
        if 'initial_covariance' not in given:


        # 次に transition_offsets の更新
        if 'transition_offsets' not in given:


        # 最後に observation_offsets の更新
        if 'observation_offsets' not in given:






