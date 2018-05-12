'''
utils for numpy
'''

import numpy as np


# determine dimensionality function (次元決定関数)
def _determine_dimensionality(variables, default = None):
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
def _parse_observations(obs):
    """Safely convert observations to their expected format"""
    obs = np.ma.atleast_2d(obs)

    # 2軸目の方が大きい場合は，第1軸と第2軸を交換
    if obs.shape[0] == 1 and obs.shape[1] > 1:
        obs = obs.T

    # 欠測値をマスク処理
    obs = np.ma.array(obs, mask = np.isnan(obs))
    return obs


# last dim (各時刻におけるパラメータを決定する関数)
def _last_dims(X, t, ndims = 2):
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