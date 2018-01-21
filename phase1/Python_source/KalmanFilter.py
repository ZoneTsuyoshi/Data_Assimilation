class Kalman_Filter(object) :
    '''
    all numpy variable (numpy 変数で定義したものを入れる)
    コード上では，pred, filt が0:Tであり，tはtに対応している
    一方，smooth は 0:T-1であり，tはt-1に対応している

    <Input Variables>
    observation [time, n_dim_obs] {float} : observation y （観測値）
    initial_mean [time, n_dim_sys] {float} : initial state mean (初期フィルタ分布の平均)
    initial_covariance [n_dim_sys, n_dim_sys] {float} : initial state covariance （初期フィルタ分布の共分散行列）
    transition_matrix [n_dim_sys, n_dim_sys] {float} : transition matrix from x_{t-1} to x_t （システムモデルの変換行列）
    transition_noise_matrix [n_dim_noise, n_dim_sys] {float} : transition noise matrix (外力行列，ノイズ変換行列)
    observation_matrix [n_dim_sys, n_dim_obs] {float} : observation matrix （観測行列）
    transition_covariance [n_dim_sys, n_dim_sys] {float} : covariance of system noise （システムノイズの共分散行列）
    observation_covariance [n_dim_obs, n_dim_obs] {float} : covariance of observation noise （観測ノイズの共分散行列）
    transition_offsets [n_dim_sys] {float} : offsets of system transition model （システムモデルの切片 ＝ バイアス = オフセット）
    observation_offsets [n_dim_obs] {float} : offsets of observation model （観測モデルの切片 = バイアス = オフセット）
    n_dim_sys {int} : dimension of system variable （システム変数の次元）
    n_dim_obs {int} : dimension of observation variable （観測変数の次元）

    <Variables>
    y [time, n_dim_obs] {float} : observation y （観測値）
    F [n_dim_sys, n_dim_sys] {float} : transition matrix from x_{t-1} to x_t （システムモデルの変換行列）
    G [n_dim_noise, n_dim_sys] {float} : transition noise matrix (外力行列，ノイズ変換行列)
    Q [n_dim_sys, n_dim_sys] {float} : covariance of system noise （システムノイズの共分散行列）
    b [time, n_dim_sys] {float} : offsets of system transition model （システムモデルの切片 ＝ バイアス = オフセット）
    H [n_dim_sys, n_dim_obs] {float} : observation matrix （観測行列）
    R [n_dim_obs, n_dim_obs] {float} : covariance of observation noise （観測ノイズの共分散行列）
    d [time, n_dim_obs] {float} : offsets of observation model （観測モデルの切片 = バイアス = オフセット）
    x_pred [time, n_dim_sys] {float} :  mean of prediction distribution （予測分布の平均）
    V_pred [time, n_dim_sys, n_dim_sys] {float} : covariance of prediction distribution (予測分布の共分散行列)
    x_filt [time, n_dim_sys] {float} : mean of filtering distribution (フィルタ分布の平均)
    V_filt [time, n_dim_sys, n_dim_sys] {float} : covariance of filtering distribution (フィルタ分布の共分散行列)
    x_RTS [time, n_dim_sys] {float} : mean of RTS smoothing distribution (固定区間平滑化分布の平均)
    V_RTS [time, n_dim_sys, n_dim_sys] {float} : covariance of RTS smoothing distribution (固定区間平滑化の共分散行列)
    q_RTS [time, n_dim_force] {float} : mean of RTS smoothing forcing noise (外力ノイズの平滑化平均)
    Q_RTS [time, n_dim_force, n_dim_force] {float} : covariance of RTS smoothing forcing noise (外力ノイズの共分散行列)
    '''

    def __init__(self, observation, initial_mean, initial_covariance, transition_matrix, transition_noise_matrix,
                 observation_matrix, transition_covariance, observation_covariance, transition_offsets = None,
                 observation_offsets = None, n_dim_sys = None, n_dim_obs = None) :
        if n_dim_obs is None :
            self.y = observation
            self.n_dim_obs = self.y.shape[1]
        else :
            self.n_dim_obs = n_dim_obs
            if self.n_dim_obs != observation.shape[1] :
                raise IndexError('You mistake dimension of observation.')
            else :
                self.y = observation
        if n_dim_sys is None :
            self.initial_mean = initial_mean
            self.n_dim_sys = self.initial_mean.shape[0]
        else :
            self.n_dim_sys = n_dim_sys
            if self.n_dim_sys != initial_mean.shape[0] :
                raise IndexError('You mistake dimension of initial mean.')
            else :
                self.initial_mean = initial_mean
        self.initial_covariance = initial_covariance
        self.F = transition_matrix
        self.G = transition_noise_matrix
        self.Q = transition_covariance
        if transition_offsets is None :
            self.b = np.zeros(self.n_dim_sys)
        else :
            self.b = transition_offsets
        self.H = observation_matrix
        self.R = observation_covariance
        if observation_offsets is None :
            self.d = np.zeros(self.n_dim_obs)
        else :
            self.d = observation_offsets

    # filter function (フィルタ値を計算する関数)
    def Filter(self) :
        '''
        T : length of data y （時系列の長さ）
        K : Kalman gain (カルマンゲイン)
        '''
        T = len(self.y)
        self.x_pred = np.zeros((T + 1, self.n_dim_sys))
        self.V_pred = np.zeros((T + 1, self.n_dim_sys, self.n_dim_sys))
        self.x_filt = np.zeros((T + 1, self.n_dim_sys))
        self.V_filt = np.zeros((T + 1, self.n_dim_sys, self.n_dim_sys))

        # initial setting (初期分布)
        self.x_pred[0] = self.initial_mean
        self.V_pred[0] = self.initial_covariance
        self.x_filt[0] = self.initial_mean
        self.V_filt[0] = self.initial_covariance

        # GQG^T if G, Q, G is consistent
        GQG = np.dot(self.G, np.dot(self.Q, self.G.T))

        for t in range(T) :
            print("\r filter calculating... t={}".format(t + 1) + "/" + str(T), end="")

            # prediction (予測分布)
            # offset が時間依存するか否かで場合分け
            if self.b.ndim == 1:
                self.x_pred[t + 1] = np.dot(self.F, self.x_filt[t]) + self.b
            elif self.b.ndim == 2:
                self.x_pred[t + 1] = np.dot(self.F, self.x_filt[t]) + self.b[t]
            self.V_pred[t + 1] = np.dot(self.F, np.dot(self.V_filt[t], self.F.T)) + GQG

            # filtering (フィルタ分布)
            K = np.dot(self.V_pred[t + 1], np.dot(self.H.T, np.linalg.inv(np.dot(self.H, np.dot(self.V_pred[t + 1], self.H.T)) + self.R)))
            # offset が時間依存するか否かで場合分け
            if self.d.ndim == 1:
                self.x_filt[t + 1] = self.x_pred[t + 1] + np.dot(K, self.y[t] - (np.dot(self.H, self.x_pred[t + 1]) + self.d))
            elif self.d.ndim == 2:
                self.x_filt[t + 1] = self.x_pred[t + 1] + np.dot(K, self.y[t] - (np.dot(self.H, self.x_pred[t + 1]) + self.d[t]))
            self.V_filt[t + 1] = self.V_pred[t + 1] - np.dot(K, np.dot(self.H, self.V_pred[t + 1]))

    # get predicted value (一期先予測値を返す関数, Filter 関数後に値を得たい時)
    def Get_Predicted_Value(self, dim) :
        return self.x_pred[1:, dim]

    # get filtered value (フィルタ値を返す関数，Filter 関数後に値を得たい時)
    def Get_Filtered_Value(self, dim) :
        return self.x_filt[1:, dim]

    # RTS smooth function (RTSスムーシングを計算する関数，Filter 関数後に)
    def RTS_Smooth(self) :
        '''
        T : length of data y (時系列の長さ)
        A : fixed interval smoothed gain (固定区間平滑化ゲイン)
        FT : smoothed gain by control problem (制御問題の平滑化ゲイン)
        '''
        T = len(self.y)
        self.x_RTS = np.zeros((T, self.n_dim_sys))
        self.V_RTS = np.zeros((T, self.n_dim_sys, self.n_dim_sys))
        self.q_RTS = np.zeros((T, self.n_dim_sys))
        self.Q_RTS = np.zeros((T, self.n_dim_sys, self.n_dim_sys))

        self.x_RTS[T - 1] = self.x_filt[T]
        self.V_RTS[T - 1] = self.V_filt[T]

        FT = np.dot(self.Q, np.dot(self.G.T, np.linalg.inv(self.V_pred[T])))
        self.q_RTS[T - 1] = np.dot(FT, self.x_RTS[T - 1] - self.x_pred[T])
        self.Q_RTS[T - 1] = self.Q + np.dot(FT, np.dot(self.V_RTS[T - 1] - self.V_pred[T], FT.T))

        # t in [1, T] (tが1~Tの逆順であることに注意)
        for t in range(T - 1, 0, -1) :
            print("\r smooth calculating... t={}".format(T - t + 1) + "/" + str(T), end="")

            # fixed interval smoothing (固定区間平滑化分布)
            A = np.dot(self.V_filt[t], np.dot(self.F.T, np.linalg.inv(self.V_pred[t + 1])))
            self.x_RTS[t - 1] = self.x_filt[t] + np.dot(A, self.x_RTS[t] - self.x_pred[t + 1])
            self.V_RTS[t - 1] = self.V_filt[t] + np.dot(A, np.dot(self.V_RTS[t] - self.V_pred[t + 1], A.T))

    # get RTS smoothed value (RTS スムーシング値を返す関数，RTS_Smooth 後に)
    def Get_RTS_Smoothed_Value(self, dim) :
        T = len(self.y)
        return self.x_RTS[:, dim]
