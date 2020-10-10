import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer


class FeatureTransformer():
    """Feature engineering and transformation
    Args:
        X (numpy.array | panda.Dataframe): N x M input matrix or data to transform
    """

    def __init__(self, X):

        # shallow copy
        self._X_orig = X
        # hard copy
        self._X_trans = X.copy()

    @property
    def X_trans(self):
        return self._X_trans

    def fit_min_max_scaling(self, feature):
        """Performs min-max scaling on the 'feature'.

        Args:
            feature (str): feature upon which to perform min-max scaling.
        Returns:
            Panda dataframe with transformed 'feature'.
        """
        mm = MinMaxScaler()
        self._X_trans[[feature]] = mm.fit_transform(self._X_orig[[feature]])
        return self._X_trans

    def fit_log_scaling(self, feature):
        """Performs log scaling on the 'feature'.

        Args:
            feature (str): feature upon which to perform log scaling.
        Returns:
            Panda dataframe with transformed 'feature'.
        """
        self._X_trans[[feature]] = np.log(self._X_orig[[feature]] + 0.01)
        return self._X_trans

    def fit_k_bin_disc(self, feature, n_bins=1000, encode='ordinal', strategy='kmeans'):
        """Performs k bins discretization of a selected 'feature'.
        Based on sklean's KBinsDiscretizer.
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html

        Args:
            feature (str): feature upon which to perform discretization.
            n_bins (int, default = 1000): The number of bins to produce.
            encode (str, default = 'ordinal'): Method used to encode the transformed result.
            strategy (str, default = 'kmeans'): Strategy used to define the widths of the bins.
        Returns:
            Panda dataframe of the transformed matrix.
        """
        k_bin_disc = KBinsDiscretizer(
            n_bins=n_bins, encode=encode, strategy=strategy)
        self._X_trans[[feature]] = k_bin_disc.fit_transform(
            self._X_orig[[feature]])
        return self._X_trans

    def fit_time_trans(self, interval):
        """Performs feature discretization of the 'Time' dividing it into intervals.

        Args:
            interval (int): interval value. For examples transform into 24, 6, 8, 12 hour intervals by providing respectively 24, 4, 3, 2.
        Returns:
            Panda dataframe with transformed 'Time' feature.
        """
        self._X_trans[['Time']] = self._X_orig.loc[:, 'Time'].apply(
            lambda x: int(x / 3600 % interval))
        return self._X_trans

    def reset_feature_trans(self, feature=None):
        """Undo the transformation for the 'feature' by resetting it back to orginal values

        Args:
            feature (str, default = None): feature to reset back to original values. Default 'None' argument resets all values. 
        """
        if feature == None:
            self._X_trans = self._X_orig.loc[:, :]
        elif feature == 'Time':
            self._X_trans = self._X_orig.loc[:, ['Time']]
        elif feature == 'Amount':
            self._X_trans = self._X_orig.loc[:, ['Amount']]
        else:
            print("Allowed args are: None, 'Time', 'Amount'")
