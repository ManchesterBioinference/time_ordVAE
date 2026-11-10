import torch
from .base_vae import BaseVAE
from torch import nn
from torch import optim
from torch.nn import functional as F
from ..types import *
from ..utils import FC_block, smooth_log
from .rna_vae import VanillaVAE, Gen_rna_vae_nb
import numpy as np

EPS = 1e-10


class TimeVaeGausMixin():

    def forward(self, input: Tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        pred_time = self.time_regressor(z)
        return [self.decode(z), input, mu, log_var, pred_time]
    
    def training_step(self, batch, batch_idx):
        x  =  batch[0]
        time = batch[1]['time_levels']

        losses = self.loss_function(*self.forward(x), time)
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch[0]
        time = batch[1]['time_levels']
        losses = self.loss_function(*self.forward(x), time)
        self.log_dict(losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch[0]
        time = batch[1]['time_levels']
        losses = self.loss_function(*self.forward(x), time)

        self.log_dict(losses, on_step=False, on_epoch=True)
        val_loss = losses['loss']
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        val_unweighted = losses['Reconstruction_Loss'] + losses['KLD'] + losses['reg_loss']
        self.log("val_unweighted", val_unweighted, on_step=False, on_epoch=True)


class TimeVaeNegBinMixin():
    def forward(self, input: Tensor, batch):
        mu, log_var = self.encode(input, batch)
        z = self.reparameterize(mu, log_var)
        pred_time = self.time_regressor(z)
        pred_proportions = self.decode(z, batch)
        pred_mean =  pred_proportions * input.sum(axis=1).unsqueeze(1)
        return [pred_mean, input, mu, log_var, pred_time]
    
    def training_step(self, batch, batch_idx):
        x  =  batch[0]
        time = batch[1]['time_levels']
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id), time)
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch[0]
        time = batch[1]['time_levels']
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id), time)
        self.log_dict(losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch[0]
        time = batch[1]['time_levels']
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id), time)

        self.log_dict(losses, on_step=False, on_epoch=True)
        val_loss = losses['loss']
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        val_unweighted = losses['Reconstruction_Loss'] + losses['KLD'] + losses['reg_loss']
        self.log("val_unweighted", val_unweighted, on_step=False, on_epoch=True)


class TimeVaeGausMixin_z_is_time(TimeVaeGausMixin):
    def forward(self, input: Tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        pred_time = z[0]
        return [self.decode(z), input, mu, log_var, pred_time]


class TimeVaeNegBinMixin_z_is_time(TimeVaeGausMixin):
    def forward(self, input: Tensor, batch):
        mu, log_var = self.encode(input, batch)
        z = self.reparameterize(mu, log_var)
        pred_time = z[0]
        pred_proportions = self.decode(z, batch)
        pred_mean =  pred_proportions * input.sum(axis=1).unsqueeze(1)
        return [pred_mean, input, mu, log_var, pred_time]

#  
class Time_Vae_ord(TimeVaeGausMixin, VanillaVAE):

    def __init__(self,
                 y_stars: torch.Tensor,
                 edges: np.array,
                 in_channels: int,
                 latent_dim: int,
                 time_weight = 0.2,
                 l1_weight = None,
                 time_noise = 0.3,
                 linear_time_head = True,
                 time_noise_span: List | None = None,
                 **kwargs) -> None:
                
                super().__init__(in_channels, latent_dim, **kwargs)
                self.save_hyperparameters()
                self.y_stars = y_stars
                self.edges = edges
                self.time_weight = time_weight
                if linear_time_head:
                    self.time_regressor = RegressorLinear(self.latent_dim)
                else:
                    self.time_regressor = RegressorNonLinear(self.latent_dim)
                self.l1_weight = l1_weight 
                self.time_noise = time_noise
                self.time_noise_span = time_noise_span


    def loss_function(self, recons, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = F.mse_loss(recons, input)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        if self.time_noise_span is not None:
            if self.time_noise_span[2] < self.current_epoch:
                time_noise = self.time_noise_span[1]
            else:
                time_noise = self.time_noise_span[0] + ((self.time_noise_span[1] - self.time_noise_span[0])/self.time_noise_span[2])*self.current_epoch
        else:
            time_noise = self.time_noise
        ord_loss = self.ord_loss(pred_time, time, scale=time_noise)

        time_weight = self.time_weight

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + ord_loss*time_weight

        if self.l1_weight is not None:
            loss += self.l1_weight*torch.norm(self.time_regressor.linear.weight, p=1)
    
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':ord_loss}  

    def ord_loss(self, pred_ys: torch.Tensor, ys: torch.Tensor, scale=0.3) -> torch.Tensor:
        
        std_normal = torch.distributions.normal.Normal(0, scale)

        log_likelihood = torch.tensor(0, dtype=torch.float, requires_grad=True)

        for y, y_pred in zip(ys, pred_ys):
            for i, y_star in enumerate(self.y_stars):
                if y_star[y] == 0:
                    continue
                else:
                    log_likelihood = log_likelihood - \
                        y_star[y]*smooth_log(
                          (std_normal.cdf(self.edges[i][1] - y_pred)
                          - std_normal.cdf(self.edges[i][0] - y_pred)))
                                                                                
                                                                                
            # log_likelihood = log_likelihood - \
            #                     torch.sum(  
            #                       torch.tensor([y_star[y]*torch.log(std_normal.cdf(self.edges[i][1] - y_pred) -
            #                         std_normal.cdf(self.edges[i][0] - y_pred))
            #                         for i, y_star in enumerate(self.y_stars)]))

        return log_likelihood[0]/pred_ys.numel()
    
class Time_Vae_cumulative_logit(TimeVaeGausMixin, VanillaVAE):

    def __init__(self,
                 y_stars: torch.Tensor,
                 edges: np.array,
                 in_channels: int,
                 latent_dim: int,
                 time_weight = 0.2,
                 l1_weight = None,
                 linear_time_head = True,
                 **kwargs) -> None:
                # edges must be a 1d vector of the edges in this case
                
                super().__init__(in_channels, latent_dim, **kwargs)
                self.save_hyperparameters()
                self.y_stars = y_stars
                self.edges = edges
                self.time_weight = time_weight
                if linear_time_head:
                    self.time_regressor = RegressorLinear(self.latent_dim)
                else:
                    self.time_regressor = RegressorNonLinear(self.latent_dim)
                self.l1_weight = l1_weight 


    def loss_function(self, recons, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = F.mse_loss(recons, input)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        ord_loss = self.cumulative_logit(pred_time, time)

        time_weight = self.time_weight

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + ord_loss*time_weight

        if self.l1_weight is not None:
            loss += self.l1_weight*torch.norm(self.time_regressor.linear.weight, p=1)
    
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':ord_loss}  

    def cumulative_logit(self, pred_ys: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        
        pred_ys = pred_ys # (batch, 1)
        edges = self.edges.unsqueeze(0)  # (1, K+1)
        ys = ys.reshape(-1)
        if (self.edges.shape[0] != self.y_stars.shape[1] + 1) and (len(self.edges.shape) != 1):
            raise ValueError('edges should be 1d and length must be number of (new) classes + 1')

        cum_probs = torch.sigmoid(edges - pred_ys)  # (batch, K+1)

        # Convert cumulative probs to class probabilities
        probs = torch.zeros(ys.shape[0], self.y_stars.shape[0], device=pred_ys.device)

        for k in range(0, self.y_stars.shape[0]):
            probs[:, k] = cum_probs[:, k+1] - cum_probs[:, k]
        

        # Step 1: get the predicted probability for the correct class
        
        true_class_probs = (probs * self.y_stars[:, ys].T).sum(dim=1)

        if true_class_probs.isnan().any():
            raise ValueError('NaN values in true class probabilities')
        if true_class_probs.shape != ys.shape:
            raise ValueError('Shape mismatch in true class probabilities. Shape is {}, expected {}'.format(true_class_probs.shape, ys.shape))

        # Step 2: take the negative log-likelihood
        nll = -torch.log(true_class_probs + EPS)

        return nll.mean()
    
class Time_Vae_continuation_ratio(TimeVaeGausMixin, VanillaVAE):

    def __init__(self,
                 y_stars: torch.Tensor,
                 edges: np.array,
                 in_channels: int,
                 latent_dim: int,
                 time_weight = 0.2,
                 l1_weight = None,
                 linear_time_head = True,
                 **kwargs) -> None:
                
                super().__init__(in_channels, latent_dim, **kwargs)
                self.save_hyperparameters()
                self.y_stars = y_stars
                self.edges = edges
                self.time_weight = time_weight
                if linear_time_head:
                    self.time_regressor = RegressorLinear(self.latent_dim)
                else:
                    self.time_regressor = RegressorNonLinear(self.latent_dim)
                self.l1_weight = l1_weight
    
    def loss_function(self, recons, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = F.mse_loss(recons, input)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        ord_loss = self.continuation_ratio(pred_time, time)

        time_weight = self.time_weight

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + ord_loss*time_weight

        if self.l1_weight is not None:
            loss += self.l1_weight*torch.norm(self.time_regressor.linear.weight, p=1)
    
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':ord_loss}  

    def continuation_ratio(self, pred_ys: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        
        pred_ys = pred_ys # (batch, 1)
        edges = self.edges.unsqueeze(0)  # (1, K+1)

        conditional_probs = torch.sigmoid(edges - pred_ys)  # (batch, K+1)
        continuation_probability = 1 - conditional_probs
        cumprod_continuation = torch.cumprod(continuation_probability, dim=1)

        # Convert conditional probabilities to class probabilities
        probs = torch.zeros(ys.shape[0], edges.shape[1] + 1, device=pred_ys.device)

        probs[:, 0] = conditional_probs[:, 0]
        for k in range(1, edges.shape[1] - 1):
            probs[:, k] = conditional_probs[:, k] * cumprod_continuation[:, k - 1]
        probs[:, -1] = cumprod_continuation[:, -1]

        # Step 1: get the predicted probability for the correct class
        true_class_probs = torch.zeros(ys.shape[0], self.y_stars.shape[1])
        
        true_class_probs = (probs * self.y_stars[:, ys].T).sum(dim=1)

        # Step 2: take the negative log-likelihood
        nll = -torch.log(true_class_probs + EPS)

        return nll.mean()

class Time_Vae_ord_z_is_time(TimeVaeGausMixin_z_is_time, VanillaVAE):
    def __init__(self,
                 y_stars: torch.Tensor,
                 edges: np.array,
                 in_channels: int,
                 latent_dim: int,
                 time_weight = 0.2,
                 time_noise = 0.3,
                 **kwargs) -> None:
                
            super().__init__(in_channels, latent_dim, **kwargs)
            self.save_hyperparameters()
            self.y_stars = y_stars
            self.edges = edges
            self.time_weight = time_weight
            self.time_noise = time_noise

    def loss_function(self, recons, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = F.mse_loss(recons, input)
        
        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]
                
        ord_loss = self.ord_loss(pred_time, time, scale=self.time_noise)

        time_weight = self.time_weight

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + ord_loss*time_weight
    
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':ord_loss}
    
    def ord_loss(self, pred_ys: torch.Tensor, ys: torch.Tensor, current_epoch=None,
                 max_epoch=None, scale=0.3, start_scale=None) -> torch.Tensor:
        
        #burnin_scale = start_scale*(0.1*max_epoch - current_epoch)/(0.1*max_epoch)  \
        #    if (self.align_burnin and (0.1*max_epoch > current_epoch)) else 0
        #scale = burnin_scale + scale

        std_normal = torch.distributions.normal.Normal(0, scale)

        log_likelihood = torch.tensor(0, dtype=torch.float, requires_grad=True)

        for y, y_pred in zip(ys, pred_ys):
            for i, y_star in enumerate(self.y_stars):
                if y_star[y] == 0:
                    continue
                else:
                    log_likelihood = log_likelihood - \
                        y_star[y]*smooth_log(
                          std_normal.cdf(self.edges[i][1] - y_pred)
                          - std_normal.cdf(self.edges[i][0] - y_pred))
                                                                                
                                                                                
            # log_likelihood = log_likelihood - \
            #                     torch.sum(  
            #                       torch.tensor([y_star[y]*torch.log(std_normal.cdf(self.edges[i][1] - y_pred) -
            #                         std_normal.cdf(self.edges[i][0] - y_pred))
            #                         for i, y_star in enumerate(self.y_stars)]))

        return log_likelihood[0]/pred_ys.numel()
    

class Time_Vae_reg(TimeVaeGausMixin, VanillaVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 time_weight = 0.2,
                 l1_weight=None,
                 linear_time_head = True,
                 **kwargs) -> None:
        super().__init__(in_channels, latent_dim, **kwargs)

        self.save_hyperparameters()
        self.time_weight = time_weight
        if linear_time_head:
            self.time_regressor = RegressorLinear(self.latent_dim)
        else:
            self.time_regressor = RegressorNonLinear(self.latent_dim)
        self.l1_weight = l1_weight 

    def loss_function(self, recons, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        kld_weight =  self.kl_weight
        time_weight = self.time_weight
        recons_loss = F.mse_loss(input, recons)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        reg_loss = F.mse_loss(pred_time, time)

        loss = self.recon_weight * recons_loss + kld_weight * kld_loss + reg_loss*time_weight
        if self.l1_weight is not None:
            loss += self.l1_weight*torch.norm(self.time_regressor.linear.weight, p=1)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':reg_loss} 


class Time_Vae_reg_z_is_time(TimeVaeGausMixin_z_is_time, VanillaVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 time_weight = 0.2,
                 **kwargs) -> None:
        super().__init__(in_channels, latent_dim, **kwargs)

        self.save_hyperparameters()
        self.time_weight = time_weight

    def loss_function(self, recons, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        kld_weight =  self.kl_weight
        time_weight = self.time_weight
        recons_loss = F.mse_loss(input, recons)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        reg_loss = F.mse_loss(pred_time, time)

        loss = self.recon_weight * recons_loss + kld_weight * kld_loss + reg_loss*time_weight
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':reg_loss} 
    


class Time_Vae_reg_non_linear(Time_Vae_reg, VanillaVAE):
    def __init__(self,
        *args,
        time_weight = 0.2,
        time_reg_hidden = [100, 50, 10],
        **kwargs) -> None:
        VanillaVAE.__init__(self, *args, **kwargs)
        self.time_weight = time_weight
        self.time_regressor = FC_block(self.latent_dim, 1, hidden_dims=time_reg_hidden)

class Time_Vae_ord_nb(TimeVaeNegBinMixin, Gen_rna_vae_nb):
    def __init__(self,
                 y_stars: torch.Tensor,
                 edges: np.array,
                 *args,
                 time_weight = 0.2,
                 l1_weight = None,
                 time_noise = 0.3,
                 linear_time_head = True,
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.time_weight = time_weight
        self.y_stars = y_stars
        self.edges = edges
        self.l1_weight = l1_weight 
        self.time_noise = time_noise

        if linear_time_head:
            self.time_regressor = RegressorLinear(self.latent_dim)
        else:
           self.time_regressor = RegressorNonLinear(self.latent_dim)

    def loss_function(self, scaled_pred_mean, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        time_weight = self.time_weight
        kld_weight =  self.kl_weight
        recon_weight = self.recon_weight

        recons_loss = self.get_nb_loss(input, scaled_pred_mean)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        ord_loss = self.ord_loss(pred_time, time, scale=self.time_noise)

        loss = recon_weight * recons_loss + kld_weight * kld_loss + ord_loss*time_weight
        if self.l1_weight is not None:
            loss += self.l1_weight*torch.norm(self.time_regressor.linear.weight, p=1)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':ord_loss}

    def ord_loss(self, pred_ys: torch.Tensor, ys: torch.Tensor, current_epoch=None,
                 max_epoch=None, scale=0.3, start_scale=None) -> torch.Tensor:
        
        #burnin_scale = start_scale*(0.1*max_epoch - current_epoch)/(0.1*max_epoch)  \
        #    if (self.align_burnin and (0.1*max_epoch > current_epoch)) else 0
        #scale = burnin_scale + scale

        std_normal = torch.distributions.normal.Normal(0, scale)

        log_likelihood = torch.tensor(0, dtype=torch.float, requires_grad=True)

        for y, y_pred in zip(ys, pred_ys):
            for i, y_star in enumerate(self.y_stars):
                if y_star[y] == 0:
                    continue
                else:
                    log_likelihood = log_likelihood - \
                        y_star[y]*smooth_log(
                          std_normal.cdf(self.edges[i][1] - y_pred)
                          - std_normal.cdf(self.edges[i][0] - y_pred))
                                        
        

        return log_likelihood[0]/pred_ys.numel()

class Time_Vae_cumulative_logit_nb(TimeVaeNegBinMixin, Gen_rna_vae_nb):
    def __init__(self,
                 y_stars: torch.Tensor,
                 edges: np.array,
                 *args,
                 time_weight = 0.2,
                 l1_weight = None,
                 linear_time_head = True,
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.time_weight = time_weight
        self.y_stars = y_stars
        self.edges = edges
        self.l1_weight = l1_weight 

        if linear_time_head:
            self.time_regressor = RegressorLinear(self.latent_dim)
        else:
           self.time_regressor = RegressorNonLinear(self.latent_dim)

    def loss_function(self, scaled_pred_mean, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = self.get_nb_loss(input, scaled_pred_mean)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        ord_loss = self.cumulative_logit(pred_time, time)

        time_weight = self.time_weight

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + ord_loss*time_weight

        if self.l1_weight is not None:
            loss += self.l1_weight*torch.norm(self.time_regressor.linear.weight, p=1)
    
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':ord_loss}
    
    def cumulative_logit(self, pred_ys: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        
        pred_ys = pred_ys # (batch, 1)
        edges = self.edges.unsqueeze(0)  # (1, K+1)
        ys = ys.reshape(-1)

        if (self.edges.shape[0] != self.y_stars.shape[1] + 1) and (len(self.edges.shape) != 1):
            raise ValueError('edges should be 1d and length must be number of (new) classes + 1')

        cum_probs = torch.sigmoid(edges - pred_ys)  # (batch, K+1)

        # Convert cumulative probs to class probabilities
        probs = torch.zeros(ys.shape[0], self.y_stars.shape[0], device=pred_ys.device)

        for k in range(0, self.y_stars.shape[0]):
            probs[:, k] = cum_probs[:, k+1] - cum_probs[:, k]
        

        # Step 1: get the predicted probability for the correct class
        
        true_class_probs = (probs * self.y_stars[:, ys].T).sum(dim=1)

        if true_class_probs.isnan().any():
            raise ValueError('NaN values in true class probabilities')
        if true_class_probs.shape != ys.shape:
            raise ValueError('Shape mismatch in true class probabilities. Shape is {}, expected {}'.format(true_class_probs.shape, ys.shape))

        # Step 2: take the negative log-likelihood
        nll = -torch.log(true_class_probs + EPS)

        return nll.mean()


class Time_Vae_reg_nb(TimeVaeNegBinMixin, Gen_rna_vae_nb):
    def __init__(self,
                *args,
                time_weight = 0.2,
                linear_time_head = True,
                **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_weight = time_weight

        if linear_time_head:
            self.time_regressor = RegressorLinear(self.latent_dim)
        else:
           self.time_regressor = RegressorNonLinear(self.latent_dim)        
    def loss_function(self, scaled_pred_mean, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = self.get_nb_loss(input, scaled_pred_mean)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        reg_loss = F.mse_loss(pred_time, time)
        
        loss = self.recon_weight*recons_loss + self.kl_weight * kld_loss + reg_loss*self.time_weight
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':reg_loss} 
            

class RegressorLinear(torch.nn.Module):

    r"""
    Linear regression

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """

    def __init__(self, inputSize, outputSize=1):
        super(RegressorLinear, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out
    
class RegressorNonLinear(torch.nn.Module):

    r"""
    Non-Linear regression

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality (default: 1)
    hidden_dims
        Output dimensionality (default: 100, 50, 10)
    """

    def __init__(self, first_layer: int,
                 last_layer: int = 1,
                 hidden_dims: List | None = [100, 50, 10]):
        super(RegressorNonLinear, self).__init__()

        self.non_linear = nn.Sequential(nn.Linear(first_layer, hidden_dims[0]),
                                         nn.BatchNorm1d(hidden_dims[0]),
                                         nn.LeakyReLU(),
                                         nn.Dropout(p=0.5),  # 2024-09-25 11:10:07 remove; 2023-09-21 17:06:34 add
                                         nn.Linear(hidden_dims[0], hidden_dims[1]),
                                         nn.BatchNorm1d(hidden_dims[1]),
                                         nn.Tanh(),
                                         nn.Dropout(p=0.5),  # 2024-09-25 11:09:54 remove; 2023-09-21 17:06:34 add
                                         nn.Linear(hidden_dims[1],  hidden_dims[2]),  # 2023-09-21 17:06:34 add
                                         nn.BatchNorm1d(hidden_dims[2]),  # 2023-09-21 17:06:34 add
                                         # nn.Tanh(),# mark here 2024-09-02 22:49:47 remove, 2023-09-21 17:06:34 add,
                                         # nn.Dropout(p=0.5)， #2023-09-21 17:06:34 add
                                         nn.Linear(hidden_dims[2], last_layer))
        

    def forward(self, x):
        out = self.non_linear(x)
        return out