import torch
from torch import nn
import torch.nn.functional as F


class UVAECriterion(nn.Module):
    """
        Calculate the loss of UVae.
        loss = E_{z~Q}[log(P(X|z))]-D[Q(z|X)||P(z)]
        which can be transformed into:
        reconstructed: ||X-X_{reconstructed}||^2/(\sigma)^2
        KL: [<L2norm(u)>^2+<L2norm(diag(\Sigma))>^2 + <L2norm(diag(ln(\Sigma)))>^2-1]
    """
    def __init__(self, x_sigma=1):
        super(UVAECriterion, self).__init__()
        self.x_sigma = x_sigma

    def forward(self, recon_x, x, mu, logvar):
        """
        args:
            recon_x: reconstructed x after UVae
            x: ground-truth x
            mu: the latent mu of uvae
            logvar: the latent logvar of uvae
        returns:
            the loss of UVae
        """
        batch_size = x.size(0)
        # calculate reconstruct loss, sum in instance
        reconstruct_loss = F.mse_loss(recon_x, x, reduction='sum') / (self.x_sigma ** 2 * batch_size)
        # KL_Distance : 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        # notice here we duplicate the 0.5 by each part
        return reconstruct_loss - KLD


def normal_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL_Distance : 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return BCE - KLD
