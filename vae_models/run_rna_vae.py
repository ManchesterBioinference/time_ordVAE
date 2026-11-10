
import torch
import models.rna_vae as vae
import data.anndata as pytorch_data
import lightening as L

rna_data = pytorch_data.RNA_AnnDataset('/mnt/mr01-data01/j72687wm/drosophlia_grns/furlong_single_cell_timeseries/rna_rds/rna_seurat.h5ad',
                                       ['time'])
n_genes = rna_data.anndata.shape[1]
n_hidden = 50

autoencoder = vae.VanillaVAE(n_genes, n_hidden)


train_loader = torch.utils.data.DataLoader(rna_data, batch_size=512, shuffle=True)
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)



