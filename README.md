# Data_Assimilation

データ同化に関するレポジトリ
Repository for Data Assimilation
- phase1(2017.10-2018.1)
    - Mac OS 10.10(Yosemite), python3.6.1, R3.3.1 GUI 1.68
    - implemented KF, 4D-Var and PF, which are basic DA method, by ipynb.
- phase2(2018.2-3)
    - Mac OS 10.13(High Sierra), python3.6.1, GUI 1.68
    - I will make class for KF, EnKF, PF, 4D-Var and UKF.
    - Kalman Filter (KF)
    	- coded basic filter and smoothing
    	$$
    	x_{t+1} = F_t x_t + b_t + v_t,\ v_t\sim N(0, Q_t)
    	y_t = H_t x_t + d_t + w_t,\ w_t\sim N(0, R_t)
    	$$
    	- coded basic EM algorithm while referencing to [pykalman](https://github.com/pykalman/pykalman)
