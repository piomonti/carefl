# ------------------------------------------
# Regression error causal inference
# ------------------------------------------
#

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale


class RECI:
    """
    This is heavily adapted from the CDT python toobox

    """

    def __init__(self, form='linear', scale_input=False, k=3):
        """Init the model."""
        self.form = form
        self.scale_input = scale_input
        self.k = k

    def predict_proba(self, data):
        """Prediction method for pairwise causal inference using the ANM model.

        Args:
             - data: np array, one column per variable
        - form: functional form, either linear of GP

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """

        d = data.shape[1] // 2
        x = data[:, :d]
        y = data[:, d:]
        if self.scale_input:
            x = scale(x).reshape((-1, d))
            y = scale(y).reshape((-1, d))
        else:
            # use min max scaler instead - as suggested by Blobaum et al (2018)
            x = MinMaxScaler().fit_transform(x.reshape((-1, d)))
            y = MinMaxScaler().fit_transform(y.reshape((-1, d)))

        p = self.compute_residual(x, y, form=self.form, k=self.k) - self.compute_residual(y, x, form=self.form,
                                                                                          k=self.k)
        causal_dir = 'x->y' if p < 0 else 'y->x'
        return p, causal_dir

    @staticmethod
    def compute_residual(x, y, form='linear', k=3):
        """Compute the fitness score of the ANM model in the x->y direction.

        Args:
            x (np.ndarray): Variable seen as cause
            y (np.ndarray): Variable seen as effect
            form (str): functional form
            k (int): degree of polynom when form == `poly`

        Returns:
            float: ANM fit score
        """

        assert form in ['linear', 'GP', 'poly']

        if form == 'linear':
            # use linear regression
            res = LinearRegression().fit(x, y)
            residuals = y - res.predict(x)
            return np.sum(np.median(residuals ** 2, axis=0))
        elif form == 'poly':
            features = np.hstack([x ** i for i in range(1, k)])
            res = LinearRegression().fit(features, y)
            residuals = y - res.predict(features)
            return np.sum(np.median(residuals ** 2, axis=0))
        else:
            # use Gaussian process regssion
            # kernel = 1.0 * RBF() #+ WhiteKernel()
            x = scale(x)
            y = scale(y)
            gp = GaussianProcessRegressor().fit(x, y)
            residuals = y - gp.predict(x)
            return np.sum(np.mean(residuals ** 2, axis=0))
