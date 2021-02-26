################################################
#       CAREFL: Causal AutoRegressive FLow     #
#                                              #
# Authors: Ilyes Khemakhem & Ricardo Pio Monti #
################################################


import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Laplace, Uniform, TransformedDistribution, SigmoidTransform
from torch.utils.data import DataLoader, Dataset

from data.generate_synth_data import CustomSyntheticDatasetDensity
from nflib import AffineCL, NormalizingFlowModel, MLP1layer, MAF, NSF_AR, ARMLP, MLP4


class CAREFL:
    """
    The CAREFL model.

    This class defines the CAREFL model developed in Causal Autoregressive Flows, by Khemakhem et al. (2021)
    manuscript available at: https://arxiv.org/abs/2011.02268

    CAREFL can be used to find causal direction between paris of (multivariate) random variables, or
    to perform interventions and answer counterfactual queries.

    Parameters:
    ----------
    config: dict
        A configuration dict that defines all necessary parameters.
        Refer to one of the provided config files for more info/

    Methods:
    ----------
    flow_lr: Init and train two normalizing flow models for each direction.
        Return their likelihood ratio evaluated on held out test data.
        The causal direction is x->y if the likelihood is positive, and y->x instead.
    predict_proba: A wrapper around flow_lr which also returns the direction.
    fit_to_sem: fits an autoregressive flow model to an SEM.
    predict_intervention: Perform an intervention on a given variable in a fitted DAG.
    predict_counterfactual: Answer counterfactual queries on a fitted DAG.

    """
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

    def flow_lr(self, data, return_scores=False):
        """
        For each direction, fit a flow model, then compute the log-likelihood ratio to determine causal direction.

        If `n_layers` and/or `n_hidden` are lists, then, *for each direction*:
            - create a flow for each combination
            - return the flow with highest test likelihood
        Note that this means that the flow parameters can be different for each direction.

        Parameters:
        ----------
        data: numpy.ndarray
            A dataset where the first half of the columns are observations of a (multivariate)
            random variable X, and the second half are those of a (multivariate) r.v. Y.
        return_scores: bool
            If True, return the lists of the test likelihoods of the multiple flows trained for each direction.

        Returns:
        ----------
        p: float
            The test likelihood which indicates direction.
        score_xy: list
            If `return_scores` is True, return all test likelihoods of the different flows trained for 'x->y'
        score_yx: list
            If `return_scores` is True, return all test likelihoods of the different flows trained for 'y->x'
        """
        dset, test_dset, dim = self._get_datasets(data)
        self.dim = dim
        # Conditional Flow Model: X->Y
        torch.manual_seed(self.config.training.seed)
        flows_xy, _ = self._train(dset)
        self.flow_xy, score_xy, self._nlxy, self._nhxy = self._evaluate(flows_xy, test_dset)
        # Conditional Flow Model: Y->X
        torch.manual_seed(self.config.training.seed)
        flows_yx, _ = self._train(dset, parity=True)
        self.flow_yx, score_yx, self._nlyx, self._nhyx = self._evaluate(flows_yx, test_dset, parity=True)
        # compute LR
        p = score_xy - score_yx
        self._update_dir(p)
        if return_scores:
            return p, score_xy, score_yx
        else:
            return p

    def predict_proba(self, data, return_scores=False):
        """
        Prediction method for pairwise causal inference using the Affine Flow LR model.

        A wrapper around `flow_lr` to have a consistent signature with the other implementations (of ANM, RECI, etc..)
        """
        if return_scores:
            p, sxy, syx = self.flow_lr(data, return_scores=return_scores)
            return p, self.direction, sxy, syx
        else:
            p = self.flow_lr(data, return_scores=return_scores)
            return p, self.direction

    def fit_to_sem(self, data, dag=None, return_scores=False):
        """
        Assuming data columns follow the causal ordering, we fit the associated SEM.

        Parameters:
        ----------
        data: numpy.ndarray
            A dataset whose columns are ordered following the causal ordering \pi. The causal ordering
            can be a result of causal discovery algorithm, or expert judgement.
        dag: None
            Not in use. Only purpose is to have the same method signature as the other algorithms.
        return_scores: bool
            If True, return the scors of all the different flows tested when fitting, else return None
        """
        dset, test_dset, dim = self._get_datasets(data)
        self.dim = dim
        torch.manual_seed(self.config.training.seed)
        flows, _ = self._train(dset)
        self.flow, score, self._nlxy, self._nhxy = self._evaluate(flows, test_dset)
        return score if return_scores else None

    def predict_intervention(self, x0_val, n_samples=100, iidx=0):
        """
        We predict the value of x given an intervention do(x_iidx = x0_val)

        As explained in Appendix E.1, we do not need to invert the flow to perform interventions,
        but doing so provides a simpler and more efficient algorithm. Thus, this implementation
        will follow Algorithm 2 of Appendix E.1.

        This proceeds in 3 steps:
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
        Given observation x_obs we estimate the counterfactual of setting x_obs[intervention_index] = cf_val

        This proceeds in 3 steps:
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

    def _get_optimizer(self, parameters):
        """
        Returns an optimizer according to the config file
        """
        optimizer = optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                               betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        if self.config.optim.scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=self.verbose)
        else:
            scheduler = None
        return optimizer, scheduler

    def _get_flow_arch(self, parity=False):
        """
        Returns a normalizing flow according to the config file.

        Parameters:
        ----------
        parity: bool
            If True, the flow follows the (1, 2) permutations, otherwise it follows the (2, 1) permutation.
        """
        # this method only gets called by _train, which in turn is only called after self.dim has been initialized
        dim = self.dim
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
        def ar_flow(hidden_dim):
            if self.config.flow.architecture.lower() in ['cl', 'realnvp']:
                return AffineCL(dim=dim, nh=hidden_dim, scale_base=self.config.flow.scale_base,
                                shift_base=self.config.flow.shift_base, net_class=net_class, parity=parity,
                                scale=self.config.flow.scale)
            elif self.config.flow.architecture.lower() == 'maf':
                return MAF(dim=dim, nh=hidden_dim, net_class=net_class, parity=parity)
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
                flow_list = [ar_flow(nh) for _ in range(nl)]
                normalizing_flows.append(NormalizingFlowModel(prior, flow_list).to(self.device))
        return normalizing_flows

    def _train(self, dset, parity=False):
        """
        Train one or multiple flors for a single direction, specified by `parity`.
        """
        train_loader = DataLoader(dset, shuffle=True, batch_size=self.config.training.batch_size)
        flows = self._get_flow_arch(parity)
        all_loss_vals = []
        for flow in flows:
            optimizer, scheduler = self._get_optimizer(flow.parameters())
            flow.train()
            loss_vals = []
            for e in range(self.epochs):
                loss_val = 0
                for _, x in enumerate(train_loader):
                    x = x.to(self.device)
                    if parity and self.config.flow.architecture == 'spline':
                        # spline flows don't have parity option and should only be used with 2D numpy data:
                        x = x[:, [1, 0]]
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

    def _get_params_from_idx(self, idx):
        return self.n_layers[idx // len(self.n_hidden)], self.n_hidden[idx % len(self.n_hidden)]

    def _evaluate(self, flows, test_dset, parity=False):
        """
        Evaluate a set of flows on test dataset, and return the one with best test likelihood.
        """
        loader = DataLoader(test_dset, batch_size=128)
        scores = []
        for idx, flow in enumerate(flows):
            if parity and self.config.flow.architecture == 'spline':
                # spline flows don't have parity option and should only be used with 2D numpy data:
                score = np.nanmean(np.concatenate([flow.log_likelihood(x.to(self.device)[:, [1, 0]]) for x in loader]))
            else:
                score = np.nanmean(np.concatenate([flow.log_likelihood(x.to(self.device)) for x in loader]))
            scores.append(score)
        try:
            # in case all scores are nan, this will raise a ValueError
            idx = np.nanargmax(scores)
        except ValueError:
            # arbitrarily pick flows[0], this doesn't matter since best_score = nan, which will
            idx = 0
        # unlike nanargmax, nanmax only raises a RuntimeWarning when all scores are nan, and will return nan
        best_score = np.nanmax(scores)
        best_flow = flows[idx]
        nl, nh = self._get_params_from_idx(idx)  # for debug
        return best_flow, best_score, nl, nh

    def _get_datasets(self, input):
        """
        Check data type, which can be:
            - an np.ndarray, in which case split it and wrap it into a train Dataset and and a test Dataset
            - a Dataset, in which case duplicate it (test dataset is the same as train dataset)
            - a tuple of Datasets, in which case just return.
        return a train Dataset, and a test Dataset
        """
        assert isinstance(input, (np.ndarray, Dataset, tuple, list))
        if isinstance(input, np.ndarray):
            dim = input.shape[-1]
            if self.config.training.split == 1.:
                data_test = np.copy(input)
            else:
                data_test = np.copy(input[int(self.config.training.split * input.shape[0]):])
                input = input[:int(self.config.training.split * input.shape[0])]
            dset = CustomSyntheticDatasetDensity(input.astype(np.float32))
            test_dset = CustomSyntheticDatasetDensity(data_test.astype(np.float32))
            return dset, test_dset, dim
        if isinstance(input, Dataset):
            dim = input[0].shape[-1]
            return input, input, dim
        if isinstance(input, (tuple, list)):
            dim = input[0][0].shape[-1]
            return input[0], input[1], dim

    def _update_dir(self, p):
        self.flow = self.flow_xy if p >= 0 else self.flow_yx
        self.direction = 'x->y' if p >= 0 else 'y->x'

    def _forward_flow(self, data):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.forward(torch.tensor(data.astype(np.float32)).to(self.device))[0][-1].detach().cpu().numpy()

    def _backward_flow(self, latent):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.backward(torch.tensor(latent.astype(np.float32)).to(self.device))[0][-1].detach().cpu().numpy()
