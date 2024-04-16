import pandas as pd
import numpy as np
import scanpy as sc

adata = sc.read('../../../../../../../media/spencer/Extra HDD/xenium/Xenium_GW18.h5ad')

import squidpy as sq

radius_dict = {}
for ctype in adata.obs['predicted.celltype'].unique():
    radius_dict[ctype] = []

radii = np.arange(0,225,25)[1:]

for radius in radii:
    
    print(radius)
    
    sq.gr.spatial_neighbors(adata, spatial_key='centroids', coord_type='generic',
                       radius=float(radius))
    
    TP_idx = np.where(adata.obs['predicted.celltype']=='SCGB3A2+SFTPB+CFTR+ cells')[0]
    
    for ctype in adata.obs['predicted.celltype'].unique():
        radius_dict[ctype].append(0)
    
    for i in range(len(TP_idx)):
        
        neigh_idx = np.where(np.array(adata.obsp['spatial_connectivities'][TP_idx[i]].todense())[0]==1)[0]
        neigh_types, neigh_counts = np.unique(adata.obs['predicted.celltype'][neigh_idx].values.astype(str), return_counts=True)
        
        for ctype,count in zip(neigh_types, neigh_counts):
            radius_dict[ctype][-1] += count
        
df_TP = pd.DataFrame(radius_dict, index=radii)/pd.DataFrame(radius_dict, index=radii).sum(1)[:,None]

df_TP.replace([np.inf, -np.inf], np.nan).fillna(0).to_csv('TP_radius_scores_GW18.csv')
