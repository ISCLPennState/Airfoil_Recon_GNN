import torch
import wandb
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import torch_geometric.nn as tgnn
from pooling import avg_pool_mod, avg_pool_mod_no_x
import multiprocessing
import time
import tqdm.autonotebook
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR




from gnn import FeaturePropagation_zero, FeaturePropagation_average, Multiscale_MessagePassing_UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)                 
    np.random.seed(seed)              
    torch.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)  



seed_everything(42)

# data load
num_training = 2222
num_validation = 247

data_path = '/storage/group/rmm7011/default/hjkim/Recon_GNN/dataset'
file_list = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pt")])

REMOVE_KEYS = ["triangles", "triangle_points", "node_feat_labels", "edge_feat_labels", "global_feat_labels"]
    
def load_and_process(f):
    data = torch.load(f, weights_only=False)

    # remove unnecessary keys
    for key in REMOVE_KEYS:
        if key in data:
            del data[key]

    # integrate x with globals
    num_nodes = data.x.shape[0]  # the number of nodes
    expanded_globals = data.globals.repeat(num_nodes, 1).float()  # (num_nodes, num_globals)

    # create new x
    new_x = torch.cat([data.x.float(), expanded_globals], dim=1)  # (num_nodes, num_features + num_globals)

    # create new data object
    return Data(
        x=new_x, # pressure, u, v, distance, u_inf, v_inf, turbulence
        edge_index=data.edge_index,
        edge_attr=data.edge_attr.float(),
        pos=data.pos.float(),
        node_type=data.node_type
    )

start = time.time()

num_workers = min(8, multiprocessing.cpu_count())  
with multiprocessing.Pool(processes=num_workers) as pool:
    dataset = pool.map(load_and_process, file_list)


print(f"Processing time: {time.time() - start:.2f} seconds")




# normalization : normalize pressure, ux, uy, distance, u_inf_x, u_inf_y, turbulence parameter
all_x = torch.cat([data.x for data in dataset], dim=0)  # (the number of nodes, the number of features)

# Feature-wise statistics
x_mean = torch.mean(all_x, dim=0)
x_std = torch.std(all_x, dim=0)


# torch.save({"mean": x_mean, "std": x_std}, "stats.pt")



# create new normalized dataset
normalized_dataset = []
for data in dataset:

    normalized_x = (data.x - x_mean) / x_std


    new_data = Data(
        x=normalized_x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        node_type=data.node_type,
        pos = data.pos
        
    )
    
    normalized_dataset.append(new_data)



g = torch.Generator()
g.manual_seed(42)

batch_size = 2

# training data
train_loader = DataLoader(normalized_dataset[:num_training], batch_size=batch_size, shuffle=True, generator=g)


# validation data
val_loader = DataLoader(normalized_dataset[num_training:num_training+num_validation], batch_size=batch_size, shuffle=False, generator=g)





# model initialization
n_mlp_mp = 2
n_repeat_mp_up = 1
lr = 1e-4


name = f'Feature-average_n_mlp_mp={n_mlp_mp}_n_repeat_mp_up_{n_repeat_mp_up}_lr_{lr}_scheduler_cosine_lrmin_1e-6_epoch_400'
# name = f'Feature-average_n_mlp_mp={n_mlp_mp}_n_repeat_mp_up_{n_repeat_mp_up}_lr_{lr}_epoch_400'


model = Multiscale_MessagePassing_UNet(
    in_channels_node = 6,
    out_channels_node = 3,
    in_channels_edge = 4,
    hidden_channels = 128,
    n_mlp_encode = 3,
    n_mlp_mp = n_mlp_mp,
    n_mp_down = [2, 2, 4],
    n_mp_up = [2, 2],
    n_repeat_mp_up = n_repeat_mp_up,
    lengthscales = [0.01, 0.02],
    bounding_box = [0.5-1.0/np.sqrt(2), 0.5+1.0/np.sqrt(2), -1.0/np.sqrt(2), 1.0/np.sqrt(2)],
    name = name
).to(device)


total_params = sum(p.numel() for p in model.parameters())


# optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# scheduler = CosineAnnealingLR(optimizer, T_max=250, eta_min=1e-6) # cosine_long_long
# scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=5e-6) cosine_long
scheduler = CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-6) # cosine_long_long

loss_fn = torch.nn.MSELoss()  


run = wandb.init(project='Recon_GNN', mode='online', name=name)


#Training
num_epochs = 400
best_loss = 1e5

progress = tqdm.autonotebook.tqdm(desc=f'Epoch 0/{num_epochs-1}', total=len(train_loader))
avg_loss = 0; val_loss = 0

for epoch in range(num_epochs):
    model.train() 
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)  # Move batch to GPU

        # Choose only field target data
        target_graphs_field = torch.stack([batch.x[:, 0][batch.node_type==0], 
                                                  batch.x[:, 1][batch.node_type==0], 
                                                  batch.x[:, 2][batch.node_type==0]], dim=1)   
        
             
        
        
        x = torch.stack([batch.x[:, 0], batch.node_type, batch.x[:, 3], batch.x[:, 4], batch.x[:, 5], batch.x[:, 6]], dim=1)
        
        # Feature propagation
        # x_initialized = FeaturePropagation_zero(x)
        x_initialized = FeaturePropagation_average(x, batch.batch)
        
        del x
        

        # model prediction: choose only field output data
        output_graphs_field = model(x_initialized, batch.edge_index, batch.edge_attr, batch.pos, batch.batch)[batch.node_type==0, :]
        
        
        # loss
        loss = loss_fn(output_graphs_field, target_graphs_field)
        
        
        progress.update(1)
        progress.set_postfix_str(f'Loss: {loss.item():.4e}, Epoch loss: {avg_loss:.4e}, Val loss: {val_loss:.4e}')
        wandb.log({'train_loss': loss.item()})
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item()
        
        
    model.eval() 
    val_total_loss = 0
    
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)  # Move batch to GPU

            
            target_graphs_field = torch.stack([batch.x[:, 0][batch.node_type==0], 
                                                  batch.x[:, 1][batch.node_type==0], 
                                                  batch.x[:, 2][batch.node_type==0]], dim=1)   
        
             
        
        
            x = torch.stack([batch.x[:, 0], batch.node_type, batch.x[:, 3], batch.x[:, 4], batch.x[:, 5], batch.x[:, 6]], dim=1)
            
                    
            # Feature propagation
            # x_initialized = FeaturePropagation_zero(x)
            x_initialized = FeaturePropagation_average(x, batch.batch)
            del x
            

            

            # model prediction: choose only field output data
            output_graphs_field = model(x_initialized, batch.edge_index, batch.edge_attr, batch.pos, batch.batch)[batch.node_type==0, :]
            
            
            # loss
            loss = loss_fn(output_graphs_field, target_graphs_field)
            
            val_total_loss += loss.item()
        
        
    avg_loss = total_loss / len(train_loader)
    val_loss = val_total_loss / len(val_loader)
    
    
    if epoch != num_epochs-1: 
        progress.reset(); progress.set_description(f'Epoch {epoch+1}/{num_epochs-1}')
    else:
        progress.close()
    progress.set_postfix_str(f'Epoch loss: {avg_loss:.4e}, Val loss: {val_loss:.4e}')


    wandb.log({'epoch_loss': avg_loss, 'val_loss': val_loss, 'Epoch': epoch})
    
    torch.save(model.state_dict(), 'model/'+ name+'_model.pt')
    torch.save(optimizer.state_dict(), 'model/'+ name+'_opt.pt')
    
    if val_loss < best_loss: 
        best_loss = val_loss
        torch.save(model.state_dict(), 'model/'+name+'_model_best.pt')
        torch.save(optimizer.state_dict(), 'model/'+name+'_opt_best.pt')
        
    # scheduler.step(val_loss)
    scheduler.step()
    print(f"Learning Rate: {scheduler.get_last_lr()}")
    


