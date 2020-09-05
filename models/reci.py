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

    def __init__(self):
        """Init the model."""
        super(RECI, self).__init__()

    def predict_proba(self, data, form='linear', scale_input=False, d=3):
        """Prediction method for pairwise causal inference using the ANM model.

        Args:
             - data: np array, one column per variable
        - form: functional form, either linear of GP

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """

        x = data[:, 0]
        y = data[:, 1]
        if scale_input:
            x = scale(x).reshape((-1, 1))
            y = scale(y).reshape((-1, 1))
        else:
            # use min max scaler instead - as suggested by Blobaum et al (2018)
            x = MinMaxScaler().fit_transform(x.reshape((-1, 1)))
            y = MinMaxScaler().fit_transform(y.reshape((-1, 1)))

        return self.compute_residual(x, y, form=form, d=d) - self.compute_residual(y, x, form=form, d=d)

    def compute_residual(self, x, y, form='linear', d=3):
        """Compute the fitness score of the ANM model in the x->y direction.

        Args:
            x (np.ndarray): Variable seen as cause
            y (np.ndarray): Variable seen as effect

        Returns:
            float: ANM fit score
        """

        assert form in ['linear', 'GP', 'poly']

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))

        if form == 'linear':
            # use linear regression
            res = LinearRegression().fit(x, y)
            residuals = y - res.predict(x)
            return np.median(residuals ** 2)
        elif form == 'poly':
            features = np.hstack([x ** i for i in range(1, d)])
            res = LinearRegression().fit(features, y)
            residuals = y - res.predict(features)
            return np.median(residuals ** 2)
        else:
            # use Gaussian process regssion
            # kernel = 1.0 * RBF() #+ WhiteKernel()
            x = scale(x)
            y = scale(y)
            gp = GaussianProcessRegressor().fit(x, y)
            residuals = y - gp.predict(x)
            return np.mean(residuals ** 2)
