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
    	<img src="https://latex.codecogs.com/gif.latex?$$&space;\begin{align}&space;x_{t&plus;1}&space;=&space;F_t&space;x_t&space;&plus;&space;b_t&space;&plus;&space;v_t,\&space;v_t\sim&space;N(0,&space;Q_t)&space;y_t&space;=&space;H_t&space;x_t&space;&plus;&space;d_t&space;&plus;&space;w_t,\&space;w_t\sim&space;N(0,&space;R_t)&space;\end{align}&space;$$">
    	- coded basic EM algorithm while referencing to [pykalman](https://github.com/pykalman/pykalman)


