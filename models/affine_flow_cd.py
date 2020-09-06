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
# load flows
from nflib.flows import AffineFullFlow, AffineFullFlowGeneral, NormalizingFlowModel
from nflib.nets import MLP1layer


class BivariateFlowLR:
    def __init__(self, n_hidden, n_layers, split=.5, prior_dist='laplace', epochs=100, opt_method='adam',
                 device='cpu', verbose=False):

        # initial guess on correct model:
        self.direction = 'none'  # to be updated after each fit
        self.flow_xy = None
        self.flow_yx = None
        self.flow = None

        self.split = split
        self.n_layers = n_layers
        self.n_hidden = n_hidden
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
                flow_xy, _ = self._init_and_train_flow(x, nh, l, self.prior_dist, self.epochs, self.device,
                                                       verbose=False)
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
                flow_yx, _ = self._init_and_train_flow(x[:, [1, 0]], nh, l, self.prior_dist, self.epochs, self.device,
                                                       verbose=False)
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

    @staticmethod
    def _init_and_train_flow(data, nh, l, prior_dist, epochs, device, opt_method='adam', verbose=False):
        # init and save 2 normalizing flows, 1 for each direction
        d = data.shape[1]
        if d > 2:
            print('using higher D implementation')
            affine_flow = AffineFullFlowGeneral
        else:
            affine_flow = AffineFullFlow
        if prior_dist == 'laplace':
            prior = Laplace(torch.zeros(d), torch.ones(d))
        else:
            prior = TransformedDistribution(Uniform(torch.zeros(d), torch.ones(d)), SigmoidTransform().inv)
        flows = [affine_flow(dim=d, nh=nh, parity=False, net_class=MLP1layer) for _ in range(l)]
        flow = NormalizingFlowModel(prior, flows).to(device)

        dset = CustomSyntheticDatasetDensity(data.astype(np.float32), device=device)
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

    def fit_to_sem(self, data, n_layers, n_hidden):
        """
        assuming data columns follow the causal ordering, we fit the associated SEM
        """
        flow, _ = self._init_and_train_flow(data, n_layers, n_hidden,
                                            self.prior_dist, self.epochs, self.device, verbose=self.verbose)
        self.flow = flow

    def invert_flow(self, data):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.forward(torch.tensor(data.astype(np.float32)))[0][-1].detach().cpu().numpy()

    def backward_flow(self, latent):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.backward(torch.tensor(latent.astype(np.float32)))[0][-1].detach().cpu().numpy()

    def predict_intervention(self, x0_val, n_samples=100, d=2, intervention_index=0):
        """
        we predict the value of x1 given an intervention on x0 (the causal variable)

        this proceeds in 3 steps:
         - invert flow to find corresponding entry for z0 at x0=x0_val (this is invariant to x1 as x0 is the cause)
         - sample z1 from prior (number of samples is n_samples)
         - pass [z0, z1 samples] through flow to get predictive samples for x1| do(x0=x0val)

        for now we only support 2D and 4D examples, will generalize in future!
        """

        if d == 2:
            # first we invert the flow:
            input_ = np.array([x0_val, 0]).reshape((1, 2))  # value of x1 here is indifferent
            z0 = self.invert_flow(input_)[0, 0]

            # now generate samples of z1 from prior
            z1 = self.flow.prior.sample((n_samples,))[:, 1].cpu().detach().numpy()

            # now we pass forward
            latentSpace = np.vstack(([z0] * n_samples, z1)).T
            latentExp = np.array([z0, 0]).reshape((1, 2))

            # finally pass through the generative model to get a distribution over x1 | do(x0=x0val)
            x1Intervention = self.backward_flow(latentSpace)
            assert np.abs(x0_val - x1Intervention[0, 0]) < 1e-5
            return x1Intervention[:, 1], self.backward_flow(latentExp)[0, 1]
        elif d == 4:
            # print('intervention for high (4) D case')
            # we are in the 4D case. Assume [x0,x1] are causes of [x2,x3]
            # the interventionIndex variable tells us which cause we intervene over (either 0th or 1st entry)

            # first we invert the flow:
            cause_input = np.zeros((1, 2))
            cause_input[0, intervention_index] = x0_val

            input_ = np.hstack((cause_input, np.zeros((1, 2))))  # value of other variables here is indifferent
            latentVal = self.invert_flow(input_)[0, intervention_index]

            # prepare latentExpectation (do sampling later)
            latentExp = np.zeros((1, 4))
            latentExp[0, intervention_index] = latentVal

            # now we pass forward
            x1Intervention = self.backward_flow(latentExp)

            return x1Intervention

            # now generate samples of z1 from prior
            # z1 = self.flow.flow_share.prior.sample( (nSamples, ) )[ :,1 ].cpu().detach().numpy()

            # now we pass forward
            # latentSpace = np.vstack(( [z0]*nSamples, z1 )).T
            # latentExp   = np.array( [z0, 0] ).reshape((1,2))

            # finally pass through the generative model to get a distribution over x1 | do(x0=x0val)
            # x1Intervention = self.flow.backwardPassFlow( latentSpace )
            # assert np.abs(x0val - x1Intervention[0,0]) < 1e-5
            # return x1Intervention[:,1], self.flow.backwardPassFlow( latentExp )[0,1]

    def predict_counterfactual(self, observation, cf_value, intervention_index=0):
        """

        given observation xObs we estimate the counterfactual of setting
        xObs[ interventionIndex ] = xCFval

        we follow the 3 steps for counterfactuals
         1) abduction - pass-forward through flow to infer latents for xObs
         2) action - pass-forward again for latent associated with xCFval
         3) prediction - backward pass through the flow
        """

        # abduction:
        latent_obs = self.invert_flow(observation)

        # action (get latent variable value under counterfactual)
        observation_cf = np.copy(observation)
        observation_cf[0, intervention_index] = cf_value
        latent_obs[0, intervention_index] = self.invert_flow(observation_cf)[0, intervention_index]

        # prediction (pass through the flow):
        return self.backward_flow(latent_obs)

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
