import torch
from .base_vae import BaseVAE
from torch import nn
from torch import optim
from torch.nn import functional as F
from ..types import *
from ..utils import FC_block

EPS = 1e-10

class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List | None = [512, 256, 128, 64, 32],
                 encoder: nn.Sequential | None = None,
                 decoder: nn.Sequential | None = None,
                 recon_weight = 1,
                 kl_weight = 0.2,
                 learning_rate = 0.001) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.learning_rate = learning_rate

        # Build Encoder
        if encoder is None:
            encoder = FC_block(in_channels, hidden_dims[-1], hidden_dims[:-2])

        self.encoder = encoder

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        if decoder is None:
           hidden_dims.reverse()
           decoder = FC_block(self.latent_dim, in_channels, hidden_dims)
           hidden_dims.reverse()
        
        self.decoder = decoder

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x F]
        :return: (Tensor) List of latent codes [[N x D], [N x D]]
        """
        result = self.encoder(input)
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x F]
        """
        result = self.decoder(z)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-8, max=1e8)
        eps = torch.randn_like(std)
        return (eps * std) + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = self.kl_weight # Account for the minibatch samples from the dataset
        recon_weight = self.recon_weight
        recons_loss =F.mse_loss(recons, input)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        loss = recon_weight*recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input cell x, returns the reconstructed image
        :param x: (Tensor) [1 x F]
        :return: (Tensor) [1 x F]
        """

        return self.forward(x)[0]
    
    def training_step(self, batch, batch_idx):
        x  =  batch[0]
        losses = self.loss_function(*self.forward(x))
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch[0]
        losses = self.loss_function(*self.forward(x))
        self.log_dict(losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch[0]
        losses = self.loss_function(*self.forward(x))
        self.log_dict(losses, on_step=False, on_epoch=True)
        val_loss = losses['loss']
        val_unweighted = losses['Reconstruction_Loss'] + losses['recons_loss']
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log("val_unweighted", val_unweighted, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
class Gen_rna_vae_nb(VanillaVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List | None = [512, 256, 128, 64, 32],
                 encoder: Any | None = None,
                 decoder: Any | None = None,
                 recon_weight = 1,
                 kl_weight = 0.2,
                 epsilon = 1e-8,
                 disp_range = (-10, 10),
                 **kwargs) -> None:

        # Build Encoder. This is not handled in VanillaVAE as the batch_id is appended to the input vector 
        if encoder is None:
            encoder = FC_block(in_channels + 1, hidden_dims[-1], hidden_dims[:-2])

        # Build Decoder
        if decoder is None:
            hidden_dims.reverse()
            activation_funcs = len(hidden_dims)*[nn.ReLU()] + [nn.Softmax()]
            decoder = FC_block(latent_dim + 1, in_channels, hidden_dims,
                               activations=activation_funcs)
            hidden_dims.reverse()

        super().__init__(in_channels, latent_dim, hidden_dims=hidden_dims,
                         encoder=encoder, decoder=decoder, kl_weight=kl_weight,
                         recon_weight=recon_weight, **kwargs)
            
        self.log_dispersion = nn.Parameter(torch.rand(in_channels))
        self.epsilon = epsilon
        self.disp_range = disp_range


    def encode(self, input: Tensor, batch: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x F]
        :param batch: (Tensor) batch of cells ([N x 1])
        :return: (Tensor) List of latent codes [[N x D], [N x D]]
        """
        input = torch.cat((input, batch), 1)
        result = self.encoder(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z: Tensor, batch: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :param batch: (Tensor) batch of cells ([N x 1])
        :param theta: (Tensor) dispersion of cells ([N x 1])
        :return: (Tensor) [B x F]
        """
        input = torch.cat((z, batch), 1)
        pred_proportions = self.decoder(input)
        return pred_proportions
    
    def forward(self, input: Tensor, batch, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input, batch)
        z = self.reparameterize(mu, log_var)
        pred_proportions = self.decode(z, batch)  #mean gamma (scVI px_scale)
        pred_mean = pred_proportions * input.sum(axis=1).unsqueeze(1) #multiply by read depth
                                                                        
        return  [input, mu, log_var, pred_mean]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        input = args[0]
        mu = args[1]
        log_var = args[2]
        scaled_pred_mean = args[3]

        kld_weight =  self.kl_weight
        recon_weight = self.recon_weight
        recons_loss = self.get_nb_loss(input, scaled_pred_mean)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        loss = recon_weight * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def training_step(self, batch, batch_idx):
        x  =  batch[0]
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id))
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        x  =  batch[0]
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id))
        self.log_dict(losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x  =  batch[0]
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id))
        self.log_dict(losses, on_step=False, on_epoch=True)
        val_loss = losses['loss']
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        val_unweighted = losses['Reconstruction_Loss'] + losses['recons_loss']
        self.log("val_unweighted", val_unweighted, on_step=False, on_epoch=True)
    
    
    def get_nb_loss(self, x, mean_poiss):
        # For stability
        dispersion = torch.exp(torch.clamp(self.log_dispersion, min=self.disp_range[0], max=self.disp_range[1]))
        mean_poiss = torch.clamp(mean_poiss, min=self.epsilon)
        disp_plus_mu = torch.clamp(mean_poiss + dispersion, min=self.epsilon)

        # Log-Gamma terms
        log_gamma_terms = torch.lgamma(x + dispersion) - torch.lgamma(dispersion) -  torch.lgamma(x + 1)  # log(y!)
        
        # Log-probability components
        log_p1 = dispersion * torch.log(dispersion / disp_plus_mu)
        log_p2 = x * torch.log(mean_poiss / disp_plus_mu)
        
        # Negative Binomial loss
        loss = log_gamma_terms + log_p1 + log_p2
        return -loss.sum()/x.numel()  # Return negative log-likelihood


class Gen_rna_vae_zinb(Gen_rna_vae_nb):
    """
    This has not been finished yet
    """
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List | None = [512, 256, 128, 64, 32],
                 encoder: Any | None = None,
                 decoder: Any | None = None,
                 dropout_nn: Any | None = None,
                 kl_weight = 0.2,
                 **kwargs) -> None:
        
        # Build dropout
        if dropout_nn is None:
            dropout_nn = FC_block(self.latent_dim + 1, in_channels, [128],
                                  activations=[nn.ReLU(), nn.Sigmoid()])

        self.dropout_nn = dropout_nn


        super().__init__(in_channels, latent_dim, hidden_dims=hidden_dims,
                         encoder=encoder, decoder=decoder, kl_weight=kl_weight)
        
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        #recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        mean_poisson = args[4]
        p_drop_out = args[4]


        kld_weight =  self.kl_weight
        recons_loss = self.get_zinb_loss(input, mean_poisson, p_drop_out, self.dispersion)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]
        
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    
    def forward(self, input: Tensor, batch, theta, read_depth, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input, batch)
        z = self.reparameterize(mu, log_var)
        proportions = self.decode(z, batch, theta)  #mean gamma (scVI px_scale)

        mean_poisson = proportions * torch.exp(read_depth)  #(scVI px_rate)
        sampled_poisson = torch.distributions.poisson.Poisson(mean_poisson).sample()

        p_drop_out = self.dropout_nn(torch.cat((z, batch), 1))
        sampled_drop_out = torch.distributions.bernoulli.Bernoulli(probs=p_drop_out).sample()
                                                                
        drop_out_mask = sampled_drop_out.type(torch.bool)^1
        
        observed = drop_out_mask*sampled_poisson
        return  [observed, input, mu, log_var, mean_poisson, p_drop_out]
