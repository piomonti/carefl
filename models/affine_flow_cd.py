# Compute bivariate Flow based measures of causal direction
#
#
# code for flows is based on the following library:
# https://github.com/karpathy/pytorch-normalizing-flows
#

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.distributions import Laplace, Uniform, TransformedDistribution, SigmoidTransform
from torch.utils.data import DataLoader

from data.generate_synth_data import CustomSyntheticDatasetDensity
from nflib import AffineCL, NormalizingFlowModel, MLP1layer, MAF, NSF_AR


def init_and_train_flow(data, nh, l, prior_dist, epochs=100, device='cpu', opt_method='adam', verbose=False):
    # init and save 2 normalizing flows, 1 for each direction
    d = data.shape[1]
    if prior_dist == 'laplace':
        prior = Laplace(torch.zeros(d), torch.ones(d))
    else:
        prior = TransformedDistribution(Uniform(torch.zeros(d), torch.ones(d)), SigmoidTransform().inv)
    flows = [AffineCL(dim=d, nh=nh, scale_base=True, shift_base=True, net_class=MLP1layer) for _ in range(l)]
    flow = NormalizingFlowModel(prior, flows).to(device)

    dset = CustomSyntheticDatasetDensity(data.astype(np.float32))
    train_loader = DataLoader(dset, shuffle=True, batch_size=128)
    optimizer = optim.Adam(flow.parameters(), lr=1e-4, weight_decay=1e-5)
    if opt_method == 'scheduler':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=verbose)

    flow.train()
    loss_vals = []
    for e in range(epochs):
        loss_val = 0
        for _, x in enumerate(train_loader):
            x.to(device)
            # compute loss
            _, prior_logprob, log_det = flow(x)
            loss = - torch.sum(prior_logprob + log_det)
            loss_val += loss.item()
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if opt_method == 'scheduler':
            scheduler.step(loss_val / len(train_loader))
        if verbose:
            print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
        loss_vals.append(loss_val)
    return flow, loss_vals


class BivariateFlowLR:
    def __init__(self, n_hidden, n_layers, split=.5, prior_dist='laplace', epochs=100, opt_method='adam',
                 device='cpu', verbose=False):

        # initial guess on correct model:
        self.direction = 'none'  # to be updated after each fit
        self.flow_xy = None
        self.flow_yx = None
        self.flow = None
        self.dim = None

        self.opt_method = opt_method
        self.split = split
        self.n_layers = n_layers if type(n_layers) is list else [n_layers]
        self.n_hidden = n_hidden if type(n_hidden) is list else [n_hidden]
        self.prior_dist = prior_dist
        self.epochs = epochs
        self.device = device
        self.verbose = verbose

    def predict_proba(self, data):
        """Prediction method for pairwise causal inference
        using the Affine Flow LR model.

        Args:
             - data: np array, one column per variable
        - form: functional form, either linear of GP

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """

        p, _, _ = self.fit_flows(data)
        return p, self.direction

    def _update_dir(self, p):
        self.flow = self.flow_xy if p >= 0 else self.flow_yx
        self.direction = 'x->y' if p >= 0 else 'y->x'

    def fit_flows(self, x):
        """
        for each direction, fit multiple flow models with varying hidden size and depth, and keep model with best fit
        """
        self.dim = x.shape[1]
        # prepare a results dir, where for each direction we train a flow for different depths and hidden dim
        results = pd.DataFrame({'L': np.repeat(self.n_layers, len(self.n_hidden)),
                                'nh': self.n_hidden * len(self.n_layers),
                                'x->y': [0] * len(self.n_layers) * len(self.n_hidden),
                                'y->x': [0] * len(self.n_layers) * len(self.n_hidden)})

        best_results = {'x->y': {'score': -1e60, 'nh': 0, 'nl': 0},
                        'y->x': {'score': -1e60, 'nh': 0, 'nl': 0}}
        # best_results = {'x->y': -1e60, 'y->x': -1e60}

        # split into training and testing data:
        if self.split == 1.:
            x_test = np.copy(x)
        else:
            x_test = np.copy(x[int(self.split * x.shape[0]):])
            x = x[:int(self.split * x.shape[0])]

        for l in self.n_layers:
            for nh in self.n_hidden:
                # -------------------------------------------------------------------------------
                #         Conditional Flow Model: X->Y
                # -------------------------------------------------------------------------------
                torch.manual_seed(0)
                flow_xy, _ = init_and_train_flow(x, nh, l, self.prior_dist, self.epochs, self.device,
                                                 opt_method=self.opt_method, verbose=self.verbose)
                score = np.nanmean(flow_xy.log_likelihood(x_test))
                if score > best_results['x->y']['score']:
                    best_results['x->y']['score'] = score
                    best_results['x->y']['nh'] = nh
                    best_results['x->y']['nl'] = l
                    self.flow_xy = flow_xy
                results.loc[(results.L == l) & (results.nh == nh), 'x->y'] = score

                # -------------------------------------------------------------------------------
                #         Conditional Flow Model: Y->X
                # -------------------------------------------------------------------------------
                torch.manual_seed(0)
                flow_yx, _ = init_and_train_flow(x[:, [1, 0]], nh, l, self.prior_dist, self.epochs, self.device,
                                                 opt_method=self.opt_method, verbose=self.verbose)
                score = np.nanmean(flow_yx.log_likelihood(x_test[:, [1, 0]]))
                if score > best_results['y->x']['score']:
                    best_results['y->x']['score'] = score
                    best_results['y->x']['nh'] = nh
                    best_results['y->x']['nl'] = l
                    self.flow_yx = flow_yx
                results.loc[(results.L == l) & (results.nh == nh), 'y->x'] = score

        p = best_results['x->y']['score'] - best_results['y->x']['score']
        self._update_dir(p)

        return p, best_results, results

    def fit_to_sem(self, data, dag):
        """
        assuming data columns follow the causal ordering, we fit the associated SEM
        """
        self.dim = data.shape[1]
        flow, _ = init_and_train_flow(data, self.n_layers[0], self.n_hidden[0], self.prior_dist,
                                      self.epochs, self.device, verbose=self.verbose, opt_method=self.opt_method)
        self.flow = flow

    def invert_flow(self, data):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.forward(torch.tensor(data.astype(np.float32)))[0][-1].detach().cpu().numpy()

    def backward_flow(self, latent):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.backward(torch.tensor(latent.astype(np.float32)))[0][-1].detach().cpu().numpy()

    def predict_intervention(self, x0_val, n_samples=100, iidx=0):
        """
        we predict the value of x given an intervention on x_iidx (the causal variable -- assuming it is a root)

        this proceeds in 3 steps:
         1) invert flow to find corresponding entry for z_iidx at x_iidx=x0_val
         2) sample z from prior (number of samples is n_samples), and replace z_iidx by inferred value from strep 1
         3) propagate z through flow to get samples for x | do(x_iidx=x0_val)
        """
        # invert flow to infer value of latent corresponding to interventional variable
        x_int = np.zeros((1, self.dim))
        x_int[0, iidx] = x0_val
        z_int = self.invert_flow(x_int)[0, iidx]
        # sample from prior and ensure z_intervention_index = z_int
        z = self.flow.prior.sample((n_samples,)).cpu().detach().numpy()
        z_est = np.zeros((1, self.dim))
        z[:, iidx] = z_est[:, iidx] = z_int
        # propagate the latent sample through flow
        x = self.backward_flow(z)
        x_from_z_est = self.backward_flow(z_est)  # to compare to x when expectation is taken after pass through flow
        # sanity check: check x_intervention_index == x0_val
        assert (np.abs(x[:, iidx] - x0_val) < 1e-3).all()
        return x.mean(0).reshape((1, self.dim)), x_from_z_est

    def predict_counterfactual(self, x_obs, cf_val, iidx=0):
        """
        given observation x_obs we estimate the counterfactual of setting
        x_obs[intervention_index] = cf_val

        this proceeds in 3 steps:
         1) abduction - pass-forward through flow to infer latents for x_obs
         2) action - pass-forward again for latent associated with cf_val
         3) prediction - backward pass through the flow
        """
        # abduction:
        z_obs = self.invert_flow(x_obs)
        # action (get latent variable value under counterfactual)
        x_cf = np.copy(x_obs)
        x_cf[0, iidx] = cf_val
        z_cf_val = self.invert_flow(x_cf)[0, iidx]
        z_obs[0, iidx] = z_cf_val
        # prediction (pass through the flow):
        x_post_cf = self.backward_flow(z_obs)
        return x_post_cf

# ______________________________
# keep the CLassCondFlow implementation for reference in case it is needed
#
# class ClassCondFlow(nn.Module):
#     """
#
#     A normalizing flow that also takes classes as inputs
#
#     This is a special architecture in an attempt to solve nonlinear ICA
#     via maximum likelihood.
#
#     As such, we assume data is generated as a smooth invertible mixture, f, of
#     latent variables s. Further, we assume latent variables follow a piecewise
#     stationary distribution (see Hyvarinen & Morioka, 2016 or Khemakhem et al 2020 for details)
#
#     The flow will be composed of two parts:
#      - the first, will seek to invert the nonlinear mixing (ie to
#        compute g = f^{-1})
#      - the second to estimate the exponential family parameters associated
#        with each segment (the Lambdas in the above papers)
#
#     This essentially means each segment will have a distinct prior distribution (I think!)
#
#
#     """
#
#     def __init__(self, prior, flows, classflows, device='cpu'):
#         super().__init__()
#         # print('initializing with: ' + str(device))
#         self.device = device
#         self.prior = prior
#         self.flow_share = NormalizingFlowModel(prior, flows).to(device)
#         self.flow_segments = [NormalizingFlowModel(prior, nf).to(device) for nf in
#                               classflows]  # classflows should be a list of flows, one per class
#         self.nclasses = len(classflows)
#
#     def __repr__(self):
#         print("number of params (shared model): ", sum(p.numel() for p in self.flow_share.parameters()))
#         print("number of params (segment model): ", sum(p.numel() for p in self.flow_segments[0].parameters()))
#         return ""
#
#     def load_data(self, data, labels):
#         """
#         load in data
#         """
#         dset = CustomSyntheticDatasetDensityClasses(data.astype(np.float32), labels.astype(np.int32),
#                                                     device=self.device)
#         self.train_loader = DataLoader(dset, shuffle=True, batch_size=128)
#
#     def train(self, epochs=100, verbose=False):
#         """
#         train networks
#         """
#
#         # define parameters
#         # print(self.flow_share.parameters())
#         params = list(self.flow_share.parameters())
#         for c in range(self.nclasses):
#             params += list(self.flow_segments[c].parameters())
#
#         # define optimizer
#         optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-5)  # todo tune WD
#
#         if self.device != 'cpu':
#             # print('here')
#             self.flow_share.to(self.device)
#             for c in range(self.nclasses):
#                 self.flow_segments[c].to(self.device)
#             # print('here.')
#
#         # begin training
#         loss_vals = []
#
#         self.flow_share.train()
#         for c in range(self.nclasses):
#             self.flow_segments[c].train()
#
#         for e in range(epochs):
#             loss_val = 0
#             for _, dat in enumerate(self.train_loader):
#                 dat, seg = dat
#                 dat.to(self.device)
#                 seg.to(self.device)
#
#                 # forward pass - run through shared network first:
#                 z_share, prior_logprob, log_det_share = self.flow_share(dat)
#
#                 # now pass through class flows, concatenate for each class then multiple by segment one hot encoding
#                 prior_logprob_final = torch.zeros((dat.shape[0], self.nclasses))
#                 for c in range(self.nclasses):
#                     z_c, prior_logprob_c, log_det_c = self.flow_segments[c](z_share[-1])
#                     prior_logprob_final[:, c] = prior_logprob_c + log_det_share + log_det_c
#
#                 # take only correct classes
#                 logprob = (prior_logprob_final * seg)
#                 loss = - torch.sum(logprob)
#                 loss_val += loss.item()
#
#                 #
#                 self.flow_share.zero_grad()
#                 for c in range(self.nclasses):
#                     self.flow_segments[c].zero_grad()
#
#                 optimizer.zero_grad()
#
#                 # compute gradients
#                 loss.backward()
#
#                 # update parameters
#                 optimizer.step()
#
#             if verbose:
#                 print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
#             loss_vals.append(loss_val)
#         return loss_vals
#
#     def EvalLL(self, x, lab):
#         """
#         lab should be one hot encoded.
#         """
#         log_probs = np.zeros(x.shape[0])
#
#         # pass through shared flow
#         x_forward, prior_logprob, log_det_share = self.flow_share(torch.tensor(x.astype(np.float32)))
#
#         # pass through conditional flows
#         prior_logprob_final = torch.zeros((x.shape[0], self.nclasses))
#         for c in range(self.nclasses):
#             z_c, prior_logprob_c, log_det_c = self.flow_segments[c](x_forward[-1])
#             prior_logprob_final[:, c] = prior_logprob_c + log_det_share + log_det_c
#
#         # take only correct classes
#         logprob = (prior_logprob_final * torch.tensor(lab.astype(np.float32))).sum(axis=1)
#         return logprob.cpu().detach().numpy()
#
#     def forwardPassFlow(self, x, fullPass=False, labels=None):
#         """
#         pass samples through the flow
#
#         fullPass: we pass through the segment flows as well, otherwise just pass through the shared flow
#
#         """
#         if fullPass:
#             x_forward = np.zeros(x.shape)
#             for c in range(self.nclasses):
#                 ii = np.where(labels[:, c] != 0)[0]
#
#                 # pass through shared
#                 x_share, _, _ = self.flow_share(torch.tensor(x[ii, :].astype(np.float32)))
#                 x_final, _, _ = self.flow_segments[c](x_share[-1])
#                 x_forward[ii, :] = x_final[-1].cpu().detach().numpy()
#
#             return x_forward
#
#         else:
#             x_forward, _, _ = self.flow_share(torch.tensor(x.astype(np.float32)))
#             return x_forward[-1].cpu().detach().numpy()
