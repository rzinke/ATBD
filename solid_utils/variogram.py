#!/usr/bin/env python3
# recommended usage:
#   from solid_utils import variogram

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from solid_utils.sampling import load_geo, rand_samp, pair_up 


class EmpiricalSemivariogram:
    def __init__(self, data, metadata):
        """Iniitialize object and store parameters.

        Parameters: data - np.ndarray, 2D array of image values
                    metadata - dict, MintPy metadata object
        """
        # Load data
        self.data = data
        self.metadata = metadata

    def __compute_ramp__(self, data, X, Y):
        # Dataset parameters
        n = len(data)

        # Formulate design matrix
        G = np.zeros((n, 6))
        G[:,0] = 1
        G[:,1] = X
        G[:,2] = Y
        G[:,3] = X**2
        G[:,4] = Y**2
        G[:,5] = X * Y

        # Invert for ramp parameters
        mhat = np.linalg.pinv(G).dot(data)

        # Compute ramp
        ramp = G.dot(mhat)

        return ramp

    def __remove_ramp__(self, data, X, Y):
        # Compute ramp
        ramp = self.__compute_ramp__(data, X, Y)

        # Remove ramp
        return data - ramp

    def __collect_data__(self, n_samples, remove_ramp, valid_range):
        # Ensure integer type
        n_samples = int(n_samples)

        # Determine coordinates of each pixel
        x, y = load_geo(self.metadata)
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()

        # Exclude NaNs
        data = self.data.flatten()

        nan_ndx = np.isnan(data)
        data = data[~nan_ndx]
        X = X[~nan_ndx]
        Y = Y[~nan_ndx]

        # Remove ramp
        if remove_ramp == True:
            data = self.__remove_ramp__(data, X, Y)

        # Collect random samples from grid
        samp_data, samp_X, samp_Y = rand_samp(data, X, Y, n_samples)

        # Form random points into pairs and compute distance
        dists, resids = pair_up(samp_X, samp_Y, samp_data)

        # Trim to range
        valid_ndx = (dists >= valid_range[0]) \
                & (dists <= valid_range[1])
        dists = dists[valid_ndx]
        resids = resids[valid_ndx]

        # Compute squared residual values
        semivar = 0.5 * resids**2

        # Record samples
        self.dists = dists
        self.semivar = semivar

    def __bin_data__(self, valid_range, n_bins):
        # Define bins
        bin_edges = np.linspace(valid_range[0], valid_range[1], n_bins+1)
        bin_dists = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Determine bin indices
        bin_ndxs = [(self.dists >= bin_edges[n])
                & (self.dists < bin_edges[n+1])
                for n in range(n_bins)]

        # Loop through bins
        self.bin_dists = []
        self.bin_semivar = []
        for n in range(n_bins):
            bin_val = np.nanmean(self.semivar[bin_ndxs[n]])
            if not np.isnan(bin_val):
                self.bin_dists.append(bin_dists[n])
                self.bin_semivar.append(bin_val)

        self.bin_dists = np.array(self.bin_dists)
        self.bin_semivar = np.array(self.bin_semivar)

    # Sampling
    def sample_data(self, n_samples:int, remove_ramp=True, valid_range=(0.1, 50), n_bins=15):
        # Collect data
        self.__collect_data__(n_samples, remove_ramp, valid_range)

        # Bin data
        self.__bin_data__(valid_range, n_bins)


    # Fitting
    @staticmethod
    def exponential_model(d, nug, rng, sill):
        a = 1 / 3
        p = (sill - nug)
        g = p * (1 - np.exp(-d / (a * rng))) + nug
        return g

    @staticmethod
    def gaussian_model(d, nug, rng, sill):
        p = (sill - nug)
        g = p * (1 - np.exp(-(d**2) / (4/7*rng)**2)) + nug
        return g

    @staticmethod
    def spherical_model(d, nug, rng, sill):
        g = np.zeros(len(d))
        p = (sill - nug)
        g[d<=rng] = p \
                * (((3*d[d<=rng])/(2*rng)) \
                   - ((d[d<=rng]**3)/(2*rng**3))) \
                + nug
        g[d>rng] = p + nug
        return g

    @staticmethod
    def hole_effect_model(d, nug, rng, sill):
        p = (sill - nug)
        g = p * (1 - (1 - (d/(rng/3))) * np.exp(-d/(rng/3))) + nug
        return g

    @staticmethod
    def powerlaw_model(d, a, b):
        g = a * d**b
        return g

    def fit(self, model_type='exponential'):
        # Determine model to use
        self.model_type = model_type
        if model_type in ['exponential']:
            self.model = self.exponential_model
        elif model_type in ['gaussian']:
            self.model = self.gaussian_model
        elif model_type in ['spherical']:
            self.model = self.spherical_model
        elif model_type in ['hole_effect']:
            self.model = self.hole_effect_model
        elif model_type in ['powerlaw']:
            self.model = self.powerlaw_model
        else:
            raise ValueError('Specified model does not exist')

        # Normalize parameters for better fitting
        d_scale = self.bin_dists.max() if self.model_type \
                not in ['powerlaw'] else 1.
        d_norm = self.bin_dists / d_scale

        semivar_scale = self.bin_semivar.max() if self.model_type \
                not in ['powerlaw'] else 1.
        semivar_norm = self.bin_semivar / semivar_scale

        # Fit model to data
        popt, pcov = sp.optimize.curve_fit(self.model, d_norm, semivar_norm)

        # Rescale fit parameters
        if self.model_type not in ['powerlaw']:
            popt[0] *= semivar_scale  # nugget
            popt[1] *= d_scale  # range
            popt[2] *= semivar_scale  # sill

        # Record parameters
        self.fit_params = popt
        self.fit_params_err = np.sqrt(np.diag(pcov))

    def predict(self, d:float|np.ndarray):
        return self.model(d, *self.fit_params)


    # Reporting
    def report(self):
        print(f"{len(self.dists):d} samples")
        print(f"{len(self.bin_dists):d} bins")
        if hasattr(self, 'model'):
            print(f"Model fit: {self.model_type.upper()}")

            if self.model_type in ['powerlaw']:
                print(f"\ta: {self.fit_params[0]:.4f} "
                      f"+- {self.fit_params_err[0]:.4f}")
                print(f"\tb: {self.fit_params[1]:.4f} "
                      f"+- {self.fit_params_err[1]:.4f}")
            else:
                print(f"\tnugget: {self.fit_params[0]:.4f} "
                      f"+- {self.fit_params_err[0]:.4f}")
                print(f"\trange: {self.fit_params[1]:.4f} "
                      f"+- {self.fit_params_err[1]:.4f}")
                print(f"\tsill: {self.fit_params[2]:.4f} "
                      f"+- {self.fit_params_err[2]:.4f}")

    def plot(self):
        # Instantiate figure and axis
        fig, ax = plt.subplots()

        # Plot raw data
        ax.scatter(self.dists, self.semivar, s=0.3, c='k',
                   alpha=0.2, label='raw samps')

        # Plot binned data
        ax.scatter(self.bin_dists, self.bin_semivar, s=8, c='b',
                   label='binned')

        # Basic plot formatting
        ylim = [0, np.percentile(self.semivar, 90)]

        ax.set_title('Semivariogram')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel(f"Semivariance ({self.metadata['UNIT']})")

        # Plot model if available
        if hasattr(self, 'model'):
            # Plot fit curve
            ax.plot(self.bin_dists, self.predict(self.bin_dists),
                    c='dodgerblue', linewidth=3,
                    label=f"{self.model_type} model")

            # Reset y-limits to ensure fit is shown
            ylim[1] = 2 * self.predict(self.dists.max())

        # Further formatting
        ax.set_ylim(ylim)
        ax.legend()

        return fig, ax
