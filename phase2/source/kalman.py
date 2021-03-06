'''
Kalman Filter のクラス
pykalman を参考にして作成する
pykalman では欠測値に対応できていないので，欠測にも対応できるクラス作成がメインテーマ
-> と思っていたけど，マスク処理で対応している
-> 欠測値(NaN)の場合は自動でマスク処理するようなコードを追加すれば拡張の意義がある

18.03.13
行列の特定の要素を最適化できる EM Algorithm に改変

18.03.17
メモリ効率化を行う
- 不要なメモリを保存しないようにする
- pykalman もカルマンゲイン等でメモリが喰われているため，それらも節約する
- デフォルトで np.float32 を使用してメモリ節約
    - 速くなったかどうかは測定していない
- 辞書はハッシュテーブル用意するからメモリ喰う
    - 辞書とリストをなるべく減らしたいけどどうするべきか

18.05.12
EMアルゴリズム様式を変える
- em_dics で要素を指定していたが，共分散構造に応じて実行するように変更
    - all : 全要素最適化
    - triD1 : 対角要素σ, 三重対角要素ρ以外は全て0の構造
    - triD2 : 2次元の格子過程を考えたときに隣接要素との共分散ρ，非隣接要素との共分散0
        となる構造．空間の縦横スケール vertical_length x horizontal_length も
        入力する必要がある．transition_vh_length, observation_vh_length で指定．
        vertical x vertical ブロックが horizontal x horizontal 個ある感じ
'''


# パッケージのインストール
import math
import numpy as np

# pykalman
import warnings
from scipy import linalg
from utils import array1d, array2d, check_random_state, \
    get_params, log_multivariate_normal_density, preprocess_arguments
from utils_filter import _parse_observations, _last_dims, \
    _determine_dimensionality


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
    transition_matrices [n_time - 1, n_dim_sys, n_dim_sys] 
        or [n_dim_sys, n_dim_sys]{numpy-array, float}
        : transition matrix from x_{t-1} to x_t
        システムモデルの変換行列[時間軸，状態変数軸，状態変数軸]
         or [状態変数軸,状態変数軸] (時不変な場合)
    transition_noise_matrices [n_time - 1, n_dim_sys, n_dim_noise]
        or [n_dim_sys, n_dim_noise] {numpy-array, float}
        : transition noise matrix
        ノイズ変換行列[時間軸，状態変数軸，ノイズ変数軸] or [状態変数軸，ノイズ変数軸]
    observation_matrices [n_time, n_dim_sys, n_dim_obs] or [n_dim_sys, n_dim_obs]
         {numpy-array, float}
        : observation matrix
        観測行列[時間軸，状態変数軸，観測変数軸] or [状態変数軸，観測変数軸]
    transition_covariance [n_time - 1, n_dim_noise, n_dim_noise]
         or [n_dim_sys, n_dim_noise]
        {numpy-array, float}
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
    transition_observation_covariance [n_time, n_dim_obs, n_dim_sys]
        or [n_dim_obs, n_dim_sys], {numpy-array, float}
        : covariance between transition noise and observation noise
        状態ノイズと観測ノイズ間の共分散 [時間軸，観測変数軸，状態変数軸]
         or [観測変数軸，状態変数軸]
    em_vars {list, string} : variable name list for EM algorithm
        (EMアルゴリズムで最適化する変数リスト)
    transition_covariance_structure, transition_cs {str} : 
        covariance structure for transition
        状態遷移分布の共分散構造
    observation_covariance_structure, observation_cs {str} :
        covariance structure for observation
        観測分布の共分散構造
    transition_vh_length, transition_v {list or numpy-array, int} :
        if think 2d space, this shows vertical dimension and horizontal length
        for transition space
        2次元空間の遷移を考えている場合，状態変数の各空間の長さ
    observation_vh_length, observation_v {list or numpy-array, int} :
        if think 2d space, this shows vertical dimension and horizontal length
        for observation space
        2次元空間の遷移を考えている場合，観測変数の各空間の長さ
    n_dim_sys {int} : dimension of system variable （システム変数の次元）
    n_dim_obs {int} : dimension of observation variable （観測変数の次元）
    dtype {type} : numpy-array type (numpy のデータ形式)

    <Variables>
    y : observation
    F : transition_matrices
    Q : transition_covariance, transition_noise_matrices
    b : transition_offsets
    H : observation_matrices
    R : observation_covariance
    d : observation_offsets
    S : transition_observation_covariance
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
    x_smooth [n_time, n_dim_sys] {numpy-array, float}
        : mean of RTS smoothing distribution
        固定区間平滑化分布の平均 [時間軸，状態変数軸]
    V_smooth [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
        : covariance of RTS smoothing distribution
        固定区間平滑化の共分散行列 [時間軸，状態変数軸，状態変数軸]
    filter_update {function}
        : update function from x_t to x_{t+1}
        フィルター更新関数

    
    state space model(状態方程式)
        x[t+1] = F[t]x[t] + G[t]v[t]
        y[t+1] = H[t]x[t] + w[t]
        v[t] ~ N(0, Q[t])
        w[t] ~ N(0, R[t])
    '''

    def __init__(self, observation = None,
                initial_mean = None, initial_covariance = None,
                transition_matrices = None, observation_matrices = None,
                transition_covariance = None, observation_covariance = None,
                transition_noise_matrices = None,
                transition_offsets = None, observation_offsets = None,
                transition_observation_covariance = None,
                em_vars = ['transition_covariance', 'observation_covariance',
                    'initial_mean', 'initial_covariance'],
                transition_covariance_structure = 'all',
                observation_covariance_structure = 'all',
                transition_vh_length = None,
                observation_vh_length = None, 
                n_dim_sys = None, n_dim_obs = None, dtype = np.float32) :
        
        # 次元決定
        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrices, array2d, -2),
             (transition_offsets, array1d, -1),
             (transition_noise_matrices, array2d, -2),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrices, array2d, -1),
             (transition_observation_covariance, array2d, -2)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrices, array2d, -2),
             (observation_offsets, array1d, -1),
             (observation_covariance, array2d, -2),
             (transition_observation_covariance, array2d, -1)],
            n_dim_obs
        )

        # transition_noise_matrices を設定していない場合は，system と次元を一致させる
        if transition_noise_matrices is None :
            self.n_dim_noise = _determine_dimensionality(
                    [(transition_covariance, array2d, -2)],
                    self.n_dim_sys
                )
            transition_noise_matrices = np.eye(self.n_dim_noise, dtype = dtype)
        else :
            self.n_dim_noise = _determine_dimensionality(
                    [(transition_noise_matrices, array2d, -1),
                     (transition_covariance, array2d, -2)]
                )


        # 次元数をチェック，欠測値のマスク処理
        self.y = _parse_observations(observation)

        # initial_mean が未入力ならば零ベクトル
        if initial_mean is None:
            self.initial_mean = np.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = initial_mean.astype(dtype)
        
        # initial_covariance が未入力ならば単位行列
        if initial_covariance is None:
            self.initial_covariance = np.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = initial_covariance.astype(dtype)

        # transition_matrices が未入力ならば単位行列
        if transition_matrices is None:
            self.F = np.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = transition_matrices.astype(dtype)

        # transition_covariance が未入力ならば単位行列
        if transition_covariance is not None:
            if transition_noise_matrices is not None:
                self.Q = self._calc_transition_covariance(
                    transition_noise_matrices,
                    transition_covariance
                    ).astype(dtype)
            else:
                self.Q = transition_covariance.astype(dtype)
        else:
            self.Q = np.eye(self.n_dim_sys, dtype = dtype)

        # transition_offsets が未入力であれば，零ベクトル
        if transition_offsets is None :
            self.b = np.zeros(self.n_dim_sys, dtype = dtype)
        else :
            self.b = transition_offsets.astype(dtype)

        # observation_matrices が未入力であれば，単位行列
        if observation_matrices is None:
            self.H = np.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = observation_matrices.astype(dtype)
        
        # observation_covariance が未入力であれば，単位行列
        if observation_covariance is None:
            self.R = np.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = observation_covariance.astype(dtype)

        # observation_offsets が未入力であれば，零ベクトル
        if observation_offsets is None :
            self.d = np.zeros(self.n_dim_obs, dtype = dtype)
        else :
            self.d = observation_offsets.astype(dtype)

        # transition_observation_covariance が未入力ならば，零行列
        if transition_observation_covariance is None:
            self.predict_update = self._predict_update_no_noise
        else:
            self.S = transition_observation_covariance
            self.predict_update = self._predict_update_noise

        ## EM algorithm で最適化するパラメータ群
        self.em_vars = em_vars

        if transition_covariance_structure == 'triD2':
            if transition_vh_length is None:
                raise ValueError('you should input transition_vh_length.')
            elif transition_vh_length[0] * transition_vh_length[1] != self.n_dim_sys:
                raise ValueError('you should confirm transition_vh_length.')
            else:
                self.transition_v = transition_vh_length[0]
                self.transition_cs = transition_covariance_structure
        elif transition_covariance_structure in ['all', 'triD1']:
            self.transition_cs = transition_covariance_structure
        else:
            raise ValueError('you should confirm transition_covariance_structure.')

        if observation_covariance_structure == 'triD2':
            if observation_vh_length is None:
                raise ValueError('you should input observation_vh_length.')
            elif observation_vh_length[0]*observation_vh_length[1] != self.n_dim_obs:
                raise ValueError('you should confirm observation_vh_length.')
            else:
                self.observation_v = observation_vh_length[0]
                self.observation_cs = observation_covariance_structure
        elif observation_covariance_structure in ['all', 'triD1']:
            self.observation_cs = observation_covariance_structure
        else:
            raise ValueError('you should confirm observation_covariance_structure.')

        # dtype
        self.dtype = dtype


    # filter function (フィルタ値を計算する関数)
    def filter(self) :
        '''
        T {int} : length of data y （時系列の長さ）
        x_pred [n_time, n_dim_sys] {numpy-array, float}
            : mean of hidden state at time t given observations
             from times [0...t-1]
            時刻 t における状態変数の予測期待値 [時間軸，状態変数軸]
        V_pred [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            : covariance of hidden state at time t given observations
             from times [0...t-1]
            時刻 t における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
        x_filt [n_time, n_dim_sys] {numpy-array, float}
            : mean of hidden state at time t given observations from times [0...t]
            時刻 t における状態変数のフィルタ期待値 [時間軸，状態変数軸]
        V_filt [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            : covariance of hidden state at time t given observations
             from times [0...t]
            時刻 t における状態変数のフィルタ共分散 [時間軸，状態変数軸，状態変数軸]
        K [n_dim_sys, n_dim_obs] {numpy-array, float}
            : Kalman gain matrix for time t [状態変数軸，観測変数軸]
            カルマンゲイン
        '''

        T = self.y.shape[0]
        self.x_pred = np.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_pred = np.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        self.x_filt = np.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_filt = np.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        K = np.zeros((self.n_dim_sys, self.n_dim_obs), dtype = self.dtype)

        # 各時刻で予測・フィルタ計算
        for t in range(T) :
            # 計算している時間を可視化
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting (初期分布)
                self.x_pred[0] = self.initial_mean
                self.V_pred[0] = self.initial_covariance
            else:
                self.predict_update(t)
            
            # y[t] の何れかがマスク処理されていれば，フィルタリングはカットする
            if np.any(np.ma.getmask(self.y[t])) :
                self.x_filt[t] = self.x_pred[t]
                self.V_filt[t] = self.V_pred[t]
            else :
                # extract t parameters (時刻tのパラメータを取り出す)
                H = _last_dims(self.H, t, 2)
                R = _last_dims(self.R, t, 2)
                d = _last_dims(self.d, t, 1)

                # filtering (フィルタ分布の計算)
                K = self.V_pred[t] @ (
                    H.T @ linalg.pinv(H @ (self.V_pred[t] @ H.T + R))
                    )
                self.x_filt[t] = self.x_pred[t] + K @ (
                    self.y[t] - (H @ self.x_pred[t] + d)
                    )
                self.V_filt[t] = self.V_pred[t] - K @ (H @ self.V_pred[t])
                

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
            raise ValueError('The dim must be less than '
                 + self.x_pred.shape[1] + '.')


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
            raise ValueError('The dim must be less than '
                 + self.x_filt.shape[1] + '.')


    # RTS smooth function (RTSスムーシングを計算する関数)
    def smooth(self) :
        '''
        T : length of data y (時系列の長さ)
        x_smooth [n_time, n_dim_sys] {numpy-array, float}
            : mean of hidden state distributions for times
             [0...n_timesteps-1] given all observations
            時刻 t における状態変数の平滑化期待値 [時間軸，状態変数軸]
        V_smooth [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            : covariances of hidden state distributions for times
             [0...n_timesteps-1] given all observations
            時刻 t における状態変数の平滑化共分散 [時間軸，状態変数軸，状態変数軸]
        A [n_dim_sys, n_dim_sys] {numpy-array, float}
            : fixed interval smoothed gain
            固定区間平滑化ゲイン [時間軸，状態変数軸，状態変数軸]
        '''

        # filter が実行されていない場合は実行
        try :
            self.x_pred[0]
        except :
            self.filter()

        T = self.y.shape[0]
        self.x_smooth = np.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_smooth = np.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        A = np.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

        self.x_smooth[-1] = self.x_filt[-1]
        self.V_smooth[-1] = self.V_filt[-1]

        # t in [0, T-2] (tが1~Tの逆順であることに注意)
        for t in reversed(range(T - 1)) :
            # 時間を可視化
            print("\r smooth calculating... t={}".format(T - t)
                 + "/" + str(T), end="")

            # 時刻 t のパラメータを取り出す
            F = _last_dims(self.F, t, 2)

            # 固定区間平滑ゲインの計算
            A = np.dot(self.V_filt[t], np.dot(F.T, linalg.pinv(self.V_pred[t + 1])))
            
            # 固定区間平滑化
            self.x_smooth[t] = self.x_filt[t] \
                + np.dot(A, self.x_smooth[t + 1] - self.x_pred[t + 1])
            self.V_smooth[t] = self.V_filt[t] \
                + np.dot(A, np.dot(self.V_smooth[t + 1] - self.V_pred[t + 1], A.T))

            
    # get RTS smoothed value (RTS スムーシング値を返す関数，smooth 後に)
    def get_smoothed_value(self, dim = None) :
        # filter されてなければ実行
        try :
            self.x_smooth[0]
        except :
            self.smooth()

        if dim is None:
            return self.x_smooth
        elif dim <= self.x_smooth.shape[1]:
            return self.x_smooth[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_smooth.shape[1] + '.')


    # em algorithm
    def em(self, n_iter = 10, em_vars = None):
        """Apply the EM algorithm
        Apply the EM algorithm to estimate all parameters specified by `em_vars`.
        em_vars に入れられているパラメータ集合について EM algorithm を用いて最適化する．
        ただし，各遷移パラメータは時不変であるとする．

        Parameters
        ----------
        n_iter : int, optional
            number of EM iterations to perform
            EM algorithm におけるイテレーション回数
        em_vars : iterable of strings or 'all'
            variables to perform EM over.  Any variable not appearing here is
            left untouched.
            EM algorithm で最適化するパラメータ群
        """

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
        '''
        for (k, v) in get_params(self).items():
            if k in DIM and (not k in given) and len(v.shape) != DIM[k]:
                warn_str = (
                    '{0} has {1} dimensions now; after fitting, '
                    + 'it will have dimension {2}'
                ).format(k, len(v.shape), DIM[k])
                warnings.warn(warn_str)
                '''

        # Actual EM iterations
        # EM algorithm の計算
        for i in range(n_iter):
            print("EM calculating... i={}".format(i+1) + "/" + str(n_iter), end="")

            # E step
            self.filter()
            
            # sigma pair smooth
            # 時刻 t,t-1 のシステムの共分散遷移
            self._sigma_pair_smooth()

            # M step
            self._calc_em(given = given)
        return self


    # calculate transition covariance (Q_new = GQG^T の計算をしておく)
    def _calc_transition_covariance(self, G, Q) :
        if G.ndim == 2:
            GT = G.T
        elif G.ndim == 3:
            GT = G.transpose(0,2,1)
        else:
            raise ValueError('The ndim of transition_noise_matrices'
                + ' should be 2 or 3,' + ' but your input is ' + str(G.ndim) + '.')
        if Q.ndim == 2 or Q.ndim == 3:
            return np.matmul(G, np.matmul(Q, GT))
        else:
            raise ValueError('The ndim of transition_covariance should be 2 or 3,'
                + ' but your input is ' + str(Q.ndim) + '.')


    # ノイズなしの予報アップデート関数
    def _predict_update_no_noise(self, t):
        # extract t-1 parameters (時刻t-1のパラメータ取り出す)
        F = _last_dims(self.F, t - 1, 2)
        Q = _last_dims(self.Q, t - 1, 2)
        b = _last_dims(self.b, t - 1, 1)

        # predict t distribution (時刻tの予測分布の計算)
        self.x_pred[t] = F @ self.x_filt[t-1] + b
        self.V_pred[t] = F @ self.V_filt[t-1] @ F.T + Q


    # ノイズありの予報アップデート関数
    def _predict_update_noise(self, t):
        if np.any(np.ma.getmask(self.y[t-1])) :
            self._predict_update_no_noise(t)
        else:
            # extract t-1 parameters (時刻t-1のパラメータ取り出す)
            F = _last_dims(self.F, t - 1, 2)
            Q = _last_dims(self.Q, t - 1, 2)
            b = _last_dims(self.b, t - 1, 1)
            H = _last_dims(self.H, t - 1, 2)
            d = _last_dims(self.d, t - 1, 1)
            S = _last_dims(self.S, t - 1, 2)
            R = _last_dims(self.R, t - 1, 2)

            # predict t distribution (時刻tの予測分布の計算)
            SR = S @ linalg.pinv(R)
            F_SRH = F - SR @ H
            self.x_pred[t] = F_SRH @ self.x_filt[t-1] + b + SR @ (self.y[t-1] - d)
            self.V_pred[t] = F_SRH @ self.V_filt[t-1] @ F_SRH.T + Q - SR @ S.T


    # sigma pair smooth 計算
    # EM のメモリセーブのために平滑化も中に組み込む
    def _sigma_pair_smooth(self):
        '''
        T {int} : length of y (時系列の長さ) 
        V_pair [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
            : Covariance between hidden states at times t and t-1
             for t = [1...n_timesteps-1].  Time 0 is ignored.
            時刻t,t-1間の状態の共分散．0は無視する
        '''

        # 時系列の長さ
        T = self.y.shape[0]
        self.x_smooth = np.zeros((T, self.n_dim_sys), dtype = self.dtype)
        self.V_smooth = np.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)

        # pairwise covariance
        self.V_pair = np.zeros((T, self.n_dim_sys, self.n_dim_sys),
             dtype = self.dtype)
        A = np.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

        self.x_smooth[-1] = self.x_filt[-1]
        self.V_smooth[-1] = self.V_filt[-1]

        # t in [0, T-2] (tが1~Tの逆順であることに注意)
        for t in reversed(range(T - 1)) :
            # 時間を可視化
            print("\r expectation step calculating... t={}".format(T - t)
                 + "/" + str(T), end="")

            # 時刻 t のパラメータを取り出す
            F = _last_dims(self.F, t, 2)

            # 固定区間平滑ゲインの計算
            A = np.dot(self.V_filt[t], np.dot(F.T, linalg.pinv(self.V_pred[t + 1])))
            
            # 固定区間平滑化
            self.x_smooth[t] = self.x_filt[t] \
                + np.dot(A, self.x_smooth[t + 1] - self.x_pred[t + 1])
            self.V_smooth[t] = self.V_filt[t] \
                + np.dot(A, np.dot(self.V_smooth[t + 1] - self.V_pred[t + 1], A.T))

            # 時点間共分散行列
            self.V_pair[t + 1] = np.dot(self.V_smooth[t], A.T)


    # calculate parameters by EM algorithm
    # EM algorithm を用いたパラメータ計算
    def _calc_em(self, given = {}):
        '''
        T {int} : length of observation y
        '''

        # length of y
        T = self.y.shape[0]

        # observation_matrices を最初に更新
        if 'observation_matrices' not in given:
            '''math
            y_t : observation, d_t : observation_offsets
            x_t : system, H : observation_matrices

            H &= ( \sum_{t=0}^{T-1} (y_t - d_t) \mathbb{E}[x_t]^T )
             ( \sum_{t=0}^{T-1} \mathbb{E}[x_t x_t^T] )^-1
            '''
            res1 = np.zeros((self.n_dim_obs, self.n_dim_sys), dtype = self.dtype)
            res2 = np.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

            for t in range(T):
                # 欠測がない y_t に関して
                if not np.any(np.ma.getmask(self.y[t])):
                    d = _last_dims(self.d, t, 1)
                    # それぞれの要素毎の積を取りたいので，outer(外積)を使う
                    res1 += np.outer(self.y[t] - d, self.x_smooth[t])
                    res2 += self.V_smooth[t] \
                        + np.outer(self.x_smooth[t], self.x_smooth[t])

            # observation_matrices (H) を更新
            self.H = np.dot(res1, linalg.pinv(res2))


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
            res1 = np.zeros((self.n_dim_obs, self.n_dim_obs), dtype = self.dtype)
            n_obs = 0

            for t in range(T):
                if not np.any(np.ma.getmask(self.y[t])):
                    H = _last_dims(self.H, t)
                    d = _last_dims(self.d, t, 1)
                    err = self.y[t] - np.dot(H, self.x_smooth[t]) - d
                    res1 += np.outer(err, err) \
                         + np.dot(H, np.dot(self.V_smooth[t], H.T))
                    n_obs += 1
            
            # temporary
            # tmp = self.R

            # 観測が1回でも確認できた場合
            if n_obs > 0:
                self.R = (1.0 / n_obs) * res1
            else:
                self.R = res1

            # covariance_structure によって場合分け
            if self.observation_cs == 'triD1':
                # 新しい R を定義しておく
                new_R = np.zeros_like(self.R, dtype=self.dtype)

                # 対角成分に関して平均を取る
                np.fill_diagonal(new_R, self.R.diagonal().mean())

                # 三重対角成分に関して平均を取る
                rho = (self.R.diagonal(1).mean() + self.R.diagonal(-1).mean()) / 2

                # 結果の統合
                self.R = new_R + np.diag(rho * np.ones(self.n_dim_obs - 1), 1) \
                     + np.diag(rho * np.ones(self.n_dim_obs - 1), -1)
            elif self.observation_cs == 'triD2':
                # 新しい R を定義しておく
                new_R = np.zeros_like(self.R, dtype=self.dtype)

                # 対角成分に関して平均を取る
                np.fill_diagonal(new_R, self.R.diagonal().mean())

                # 三重対角成分, 隣接成分に関して平均を取る
                td = np.ones(self.n_dim_obs - 1)
                td[self.observation_v-1::self.observation_v-1] = 0
                condition = np.diag(td, 1) + np.diag(td, -1) \
                    + np.diag(
                        np.ones(self.n_dim_obs - self.observation_v),
                        self.observation_v
                        ) \
                    + np.diag(
                        np.ones(self.n_dim_obs - self.observation_v),
                        self.observation_v
                        )
                rho = self.R[condition.astype(bool)].mean()

                # 結果の統合
                self.R = new_R + rho * condition.astype(self.dtype)


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
            res1 = np.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
            res2 = np.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)
            for t in range(1, T):
                b = _last_dims(self.b, t - 1, 1)
                res1 += self.V_pair[t] + np.outer(
                    self.x_smooth[t], self.x_smooth[t - 1]
                    )
                res1 -= np.outer(b, self.x_smooth[t - 1])            
                res2 += self.V_smooth[t - 1] \
                    + np.outer(self.x_smooth[t - 1], self.x_smooth[t - 1])

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
            res1 = np.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

            # 全てを最適化するわけではないので，素朴な計算になっている
            for t in range(T - 1):
                F = _last_dims(self.F, t)
                b = _last_dims(self.b, t, 1)
                err = self.x_smooth[t + 1] - np.dot(F, self.x_smooth[t]) - b
                Vt1t_F = np.dot(self.V_pair[t + 1], F.T)
                res1 += (
                    np.outer(err, err)
                    + np.dot(F, np.dot(self.V_smooth[t], F.T))
                    + self.V_smooth[t + 1]
                    - Vt1t_F - Vt1t_F.T
                )

            self.Q = (1.0 / (T - 1)) * res1

            # covariance_structure によって場合分け
            if self.transition_cs == 'triD1':
                # 新しい R を定義しておく
                new_Q = np.zeros_like(self.Q, dtype=self.dtype)

                # 対角成分に関して平均を取る
                np.fill_diagonal(new_Q, self.Q.diagonal().mean())

                # 三重対角成分に関して平均を取る
                rho = (self.Q.diagonal(1).mean() + self.Q.diagonal(-1).mean()) / 2

                # 結果の統合
                self.Q = new_Q + np.diag(rho * np.ones(self.n_dim_sys - 1), 1)\
                     + np.diag(rho * np.ones(self.n_dim_sys - 1), -1)
            elif self.transition_cs == 'triD2':
                # 新しい R を定義しておく
                new_Q = np.zeros_like(self.Q, dtype=self.dtype)

                # 対角成分に関して平均を取る
                np.fill_diagonal(new_Q, self.Q.diagonal().mean())

                # 三重対角成分, 隣接成分に関して平均を取る
                td = np.ones(self.n_dim_sys - 1)
                td[self.transition_v-1::self.transition_v-1] = 0
                condition = np.diag(td, 1) + np.diag(td, -1) \
                    + np.diag(
                        np.ones(self.n_dim_sys - self.transition_v),
                        self.transition_v
                        ) \
                    + np.diag(
                        np.ones(self.n_dim_sys - self.transition_v),
                        self.transition_v
                        )
                rho = self.Q[condition.astype(bool)].mean()

                # 結果の統合
                self.Q = new_Q + rho * condition.astype(self.dtype)

        # 次に initial_mean の更新
        if 'initial_mean' not in  given:
            '''math
            x_0 : system of t=0
                \mu_0 = \mathbb{E}[x_0]
            '''
            tmp = self.initial_mean
            self.initial_mean = self.x_smooth[0]


        # 次に initial_covariance の更新
        if 'initial_covariance' not in given:
            '''math
            mu_0 : system of t=0
                \Sigma_0 = \mathbb{E}[x_0, x_0^T] - \mu_0 \mu_0^T
            '''
            x0 = self.x_smooth[0]
            x0_x0 = self.V_smooth[0] + np.outer(x0, x0)

            self.initial_covariance = x0_x0 - np.outer(self.initial_mean, x0)
            self.initial_covariance += - np.outer(x0, self.initial_mean)\
                 + np.outer(self.initial_mean, self.initial_mean)


        # 次に transition_offsets の更新
        if 'transition_offsets' not in given:
            '''math
            b : transition_offsets, x_t : system
            F_t : transition_matrices
                b = \frac{1}{T-1} \sum_{t=1}^{T-1}
                        \mathbb{E}[x_t] - F_{t-1} \mathbb{E}[x_{t-1}]
            '''
            self.b = np.zeros(self.n_dim_sys, dtype = self.dtype)

            # 最低でも3点での値が必要
            if T > 1:
                for t in range(1, T):
                    F = _last_dims(self.F, t - 1)
                    self.b += self.x_smooth[t] - np.dot(F, self.x_smooth[t - 1])
                self.b *= (1.0 / (T - 1))


        # 最後に observation_offsets の更新
        if 'observation_offsets' not in given:
            '''math
            d : observation_offsets, y_t : observation
            H_t : observation_matrices, x_t : system
                d = \frac{1}{T} \sum_{t=0}^{T-1} y_t - H_{t} \mathbb{E}[x_{t}]
            '''
            self.d = np.zeros(self.n_dim_obs, dtype = self.dtype)
            n_obs = 0
            for t in range(T):
                if not np.any(np.ma.getmask(self.y[t])):
                    H = _last_dims(self.H, t)
                    self.d += self.y[t] - np.dot(H, self.x_smooth[t])
                    n_obs += 1
            if n_obs > 0:
                self.d *= (1.0 / n_obs)

