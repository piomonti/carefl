# Compute bivariate Flow based measures of causal direction
#
#
# code for flows is based on the following library:
# https://github.com/karpathy/pytorch-normalizing-flows
#

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Laplace, Uniform, TransformedDistribution, SigmoidTransform
from torch.utils.data import DataLoader

from data.generate_synth_data import CustomSyntheticDatasetDensity
from nflib import AffineCL, NormalizingFlowModel, MLP1layer, MAF, NSF_AR, ARMLP, MLP4


class CAReFl:
    def __init__(self, config):
        self.config = config
        self.n_layers = config.flow.nl
        self.n_hidden = config.flow.nh
        self.epochs = config.training.epochs
        self.device = config.device
        self.verbose = config.training.verbose

        # initial guess on correct model to be updated after each fit
        self.dim = None
        self.direction = 'none'
        self.flow_xy = self.flow_yx = self.flow = None
        self._nhxy = self._nhyx = self._nlxy = self._nlyx = None

    def _get_optimizer(self, parameters):

        optimizer = optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                               betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        if self.config.optim.scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=self.verbose)
        else:
            scheduler = None
        return optimizer, scheduler

    def _get_flow_arch(self, dim):
        # prior
        if self.config.flow.prior_dist == 'laplace':
            prior = Laplace(torch.zeros(dim).to(self.device), torch.ones(dim).to(self.device))
        else:
            prior = TransformedDistribution(Uniform(torch.zeros(dim).to(self.device), torch.ones(dim).to(self.device)),
                                            SigmoidTransform().inv)
        # net type for flow parameters
        if self.config.flow.net_class.lower() == 'mlp':
            net_class = MLP1layer
        elif self.config.flow.net_class.lower() == 'mlp4':
            net_class = MLP4
        elif self.config.flow.net_class.lower() == 'armlp':
            net_class = ARMLP
        else:
            raise NotImplementedError('net_class {} not understood.'.format(self.config.flow.net_class))

        # flow type
        def affine_flow(hidden_dim):
            if self.config.flow.architecture.lower() in ['cl', 'realnvp']:
                return AffineCL(dim=dim, nh=hidden_dim, scale_base=self.config.flow.scale_base,
                                shift_base=self.config.flow.shift_base, net_class=net_class)
            elif self.config.flow.architecture.lower() == 'maf':
                return MAF(dim=dim, nh=hidden_dim, net_class=net_class, parity=False)
            elif self.config.flow.architecture.lower() == 'spline':
                return NSF_AR(dim=dim, hidden_dim=hidden_dim, base_network=net_class)
            else:
                raise NotImplementedError('Architecture {} not understood.'.format(self.config.flow.architecture))

        # support training multiple flows for varying depth and width, and keep only best
        self.n_layers = self.n_layers if type(self.n_layers) is list else [self.n_layers]
        self.n_hidden = self.n_hidden if type(self.n_hidden) is list else [self.n_hidden]
        normalizing_flows = []
        for nl in self.n_layers:
            for nh in self.n_hidden:
                # construct normalizing flows
                flow_list = [affine_flow(nh) for _ in range(nl)]
                normalizing_flows.append(NormalizingFlowModel(prior, flow_list).to(self.device))
        return normalizing_flows

    def _train(self, data):
        dim = data.shape[1]
        dset = CustomSyntheticDatasetDensity(data.astype(np.float32))
        train_loader = DataLoader(dset, shuffle=True, batch_size=128)
        flows = self._get_flow_arch(dim)
        all_loss_vals = []
        print('Training {} flows'.format(len(flows)))
        for flow in flows:
            optimizer, scheduler = self._get_optimizer(flow.parameters())
            flow.train()
            loss_vals = []
            for e in range(self.epochs):
                loss_val = 0
                for _, x in enumerate(train_loader):
                    x = x.to(self.device)
                    # compute loss
                    _, prior_logprob, log_det = flow(x)
                    loss = - torch.sum(prior_logprob + log_det)
                    loss_val += loss.item()
                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if self.config.optim.scheduler:
                    scheduler.step(loss_val / len(train_loader))
                if self.verbose:
                    print('epoch {}/{} \tloss: {}'.format(e, self.epochs, loss_val))
                loss_vals.append(loss_val)
            all_loss_vals.append(loss_vals)
        return flows, all_loss_vals

    def _evaluate(self, flows, data):
        best_score, best_flow, nl, nh = -1e60, flows[0], 0, 0
        print('Evaluating {} flows'.format(len(flows)))
        for idx, flow in enumerate(flows):
            score = np.nanmean(flow.log_likelihood(torch.tensor(data.astype(np.float32)).to(self.device)))
            if score > best_score:
                best_flow = flow
                best_score = score
                nl, nh = self._get_params_from_idx(idx)  # for debug
        return best_flow, best_score, nl, nh

    def _get_params_from_idx(self, idx):
        return self.n_layers[idx // len(self.n_hidden)], self.n_hidden[idx % len(self.n_hidden)]

    def predict_proba(self, data):
        """Prediction method for pairwise causal inference using the Affine Flow LR model."""
        p = self.flow_lr(data)
        return p, self.direction

    def _update_dir(self, p):
        self.flow = self.flow_xy if p >= 0 else self.flow_yx
        self.direction = 'x->y' if p >= 0 else 'y->x'

    def flow_lr(self, data):
        """
        for each direction, fit a flow model, then compute the log-likelihood ratio to determine causal direction
        """
        self.dim = data.shape[1]
        # split into training and testing data:
        if self.config.training.split == 1.:
            data_test = np.copy(data)
        else:
            data_test = np.copy(data[int(self.config.training.split * data.shape[0]):])
            data = data[:int(self.config.training.split * data.shape[0])]
        # Conditional Flow Model: X->Y
        torch.manual_seed(self.config.training.seed)
        flows_xy, _ = self._train(data)
        self.flow_xy, score_xy, self._nlxy, self._nhxy = self._evaluate(flows_xy, data_test)
        # Conditional Flow Model: Y->X
        torch.manual_seed(self.config.training.seed)
        flows_yx, _ = self._train(data[:, [1, 0]])
        self.flow_yx, score_yx, self._nlyx, self._nhyx = self._evaluate(flows_yx, data_test[:, [1, 0]])
        # compute LR
        p = score_xy - score_yx
        self._update_dir(p)
        return p

    def _forward_flow(self, data):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.forward(torch.tensor(data.astype(np.float32)).to(self.device))[0][-1].detach().cpu().numpy()

    def _backward_flow(self, latent):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.backward(torch.tensor(latent.astype(np.float32)).to(self.device))[0][-1].detach().cpu().numpy()

    def fit_to_sem(self, data, dag):
        """
        assuming data columns follow the causal ordering, we fit the associated SEM
        """
        self.dim = data.shape[1]
        torch.manual_seed(self.config.training.seed)
        flows, _ = self._train(data)
        self.flow, _, self._nlxy, self._nhxy = self._evaluate(flows, data)

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
        z_int = self._forward_flow(x_int)[0, iidx]
        # sample from prior and ensure z_intervention_index = z_int
        z = self.flow.prior.sample((n_samples,)).cpu().detach().numpy()
        z_est = np.zeros((1, self.dim))
        z[:, iidx] = z_est[:, iidx] = z_int
        # propagate the latent sample through flow
        x = self._backward_flow(z)
        x_from_z_est = self._backward_flow(z_est)  # to compare to x when expectation is taken after pass through flow
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
        z_obs = self._forward_flow(x_obs)
        # action (get latent variable value under counterfactual)
        x_cf = np.copy(x_obs)
        x_cf[0, iidx] = cf_val
        z_cf_val = self._forward_flow(x_cf)[0, iidx]
        z_obs[0, iidx] = z_cf_val
        # prediction (pass through the flow):
        x_post_cf = self._backward_flow(z_obs)
        return x_post_cf
