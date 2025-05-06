import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata
from torch_geometric.data import Data

from gnn import FeaturePropagation_zero, FeaturePropagation_average, Multiscale_MessagePassing_UNet, FeaturePropagation_average_test




def denormalize_x_subset(x_normalized, mean, std):
    """
    recover the first three columns of x_normalized using mean[0:3] & std[0:3]

    Args:
    - x_normalized (torch.Tensor): normalized data (N, D)
    - mean (torch.Tensor)
    - std (torch.Tensor)

    Returns:
    - torch.Tensor
    """
    x_restored = x_normalized.clone()  
    x_restored[:, :3] = x_normalized[:, :3] * std[:3] + mean[:3]  
    return x_restored


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



data_range = range(0, 2400+1, 200)  # 0, 200, 400, ..., 2400


dataset_path = "dataset/"

for idx in data_range:
    data_name = f"graph_{idx:04d}"  
    data = torch.load(f"{dataset_path}{data_name}.pt", weights_only=False).to(device)
    
    

    U_inf_x = data.globals[0].cpu().numpy()
    U_inf_y = data.globals[1].cpu().numpy()

    U_inf = np.sqrt(U_inf_x**2+U_inf_y**2)

    AOA = np.degrees(np.arctan(U_inf_y/U_inf_x))
    

    num_nodes = data.x.shape[0]  
    expanded_globals = data.globals.repeat(num_nodes, 1).float()  # (num_nodes, num_globals)
    new_x = torch.cat([data.x.float(), expanded_globals], dim=1)  # (num_nodes, num_features + num_globals)
    data = Data(
            x=new_x, # pressure, u, v, distance, u_inf, v_inf, turbulence
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float(),
            pos=data.pos.float(),
            node_type=data.node_type,
            triangles=data.triangles,
            triangle_points=data.triangle_points
        )



    stats = torch.load("stats.pt", weights_only=False)
    x_mean = stats["mean"].to(device)
    x_std = stats["std"].to(device)


    normalized_x = (data.x - x_mean) / x_std

    normalized_data = Data(
            x=normalized_x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            node_type=data.node_type,
            pos = data.pos,
            triangles=data.triangles,
            triangle_points=data.triangle_points
        )



    x = torch.stack([normalized_data.x[:, 0], normalized_data.node_type, normalized_data.x[:, 3], normalized_data.x[:, 4], normalized_data.x[:, 5], normalized_data.x[:, 6]], dim=1)

    x_initialized = FeaturePropagation_average_test(x)


    model = Multiscale_MessagePassing_UNet(
        in_channels_node = 6,
        out_channels_node = 3,
        in_channels_edge = 4,
        hidden_channels = 128,
        n_mlp_encode = 3,
        n_mlp_mp = 2,
        n_mp_down = [2, 2, 4],
        n_mp_up = [2, 2],
        n_repeat_mp_up = 1,
        lengthscales = [0.01, 0.02],
        bounding_box = [0.5-1/np.sqrt(2), 0.5+1/np.sqrt(2), -1.0/np.sqrt(2), 1.0/np.sqrt(2)],
    ).to(device)

    name = 'Feature-average_n_mlp_mp=2_n_repeat_mp_up_1_lr_0.0001_scheduler_cosine_lrmin_1e-6_model_best'
    model.load_state_dict(torch.load(f'model/{name}.pt'))


    output_graphs = model(x_initialized, normalized_data.edge_index, normalized_data.edge_attr, normalized_data.pos)

    node_type = data.node_type.unsqueeze(1)

    output_graphs_new = torch.cat((output_graphs, node_type), dim=1)


    output_graphs[data.node_type==1, :3] = normalized_data.x[data.node_type==1, :3]


    output_restored = denormalize_x_subset(output_graphs, x_mean, x_std)

    target_retosred = denormalize_x_subset(normalized_data.x[:, :3], x_mean, x_std)



    data_output = Data(
        x=output_restored.detach().cpu(),  # (56827, 5) (x + pos)
        pos = data.pos.cpu(),
        triangles=data.triangles.cpu(),  # trianlges (53736, 3)
        triangle_points=data.triangle_points.cpu()  # triangle coordinates (2, 75590)
    )


    data = Data(
        x=data.x.detach().cpu(),  # (56827, 5) (x + pos)
        pos = data.pos.cpu(),
        triangles=data.triangles.cpu(),  # trianlges (53736, 3)
        triangle_points=data.triangle_points.cpu()  # triangle coordinates (2, 75590)
    )





    fig, ax = plt.subplots(1, 4, figsize=(30, 4), gridspec_kw={'width_ratios': [2, 2, 2, 2]})


    plt.cla()

    # triangle_points: coordinates, shape = [2, 75590]
    triangle_points = data.triangle_points.numpy()
    x_mesh = triangle_points[0, :]
    y_mesh = triangle_points[1, :]

    # triangles: triangle indices, shape = [53736, 3]
    triangles = data.triangles.numpy()

    # pos: the location where x is defined, shape = [56827, 2]
    pos = data.pos.numpy()

    # data.x[:, 1]: shape = [56827]
    values = data.x[:, 1].numpy()

    # interpolate to triangle coordinates
    points_mesh = np.vstack((x_mesh, y_mesh)).T
    interpolated_values = griddata(pos, values, points_mesh, method='linear')


    nan_mask = np.isnan(interpolated_values)
    if np.any(nan_mask):
        interpolated_values[nan_mask] = griddata(pos, values, points_mesh[nan_mask], method='nearest')



    vmin = np.min(interpolated_values)
    vmax = np.max(interpolated_values)


    mesh_triangulation = tri.Triangulation(x_mesh, y_mesh, triangles)

    tpc_target = ax[0].tripcolor(mesh_triangulation, interpolated_values, shading='gouraud', cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(tpc_target, ax=ax[0], label="Ux")
    ax[0].triplot(mesh_triangulation, color="black", linewidth=0.3, alpha=0.3)


    ax[0].set_title(f"Ux_true (U_inf = {U_inf:.2f}m/s, AOA = {AOA:.2f}degree)")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    # ax.axis("equal")  
    # ax[0].set_xlim(-0.25, 1.25)
    # ax[0].set_ylim(-0.25, 0.25)
    ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylim(-1, 1)





    # triangle_points: coordinates, shape = [2, 75590]
    triangle_points = data_output.triangle_points.numpy()
    x_mesh = triangle_points[0, :]
    y_mesh = triangle_points[1, :]

    # triangles: triangle indices, shape = [53736, 3]
    triangles = data_output.triangles.numpy()

    # pos: the location where x is defined, shape = [56827, 2]
    pos = data_output.pos.numpy()

    # data.x[:, 1]: shape = [56827]
    values = data_output.x[:, 1].numpy()

    # interpolate to triangle coordinates
    points_mesh = np.vstack((x_mesh, y_mesh)).T
    interpolated_values = griddata(pos, values, points_mesh, method='linear')

    nan_mask = np.isnan(interpolated_values)
    if np.any(nan_mask):
        interpolated_values[nan_mask] = griddata(pos, values, points_mesh[nan_mask], method='nearest')

    mesh_triangulation = tri.Triangulation(x_mesh, y_mesh, triangles)

    tpc = ax[1].tripcolor(mesh_triangulation, interpolated_values, shading='gouraud', cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(tpc, ax=ax[1], label="Ux")

    ax[1].triplot(mesh_triangulation, color="black", linewidth=0.3, alpha=0.3)



    ax[1].set_title("Ux_pred")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    # ax.axis("equal")  
    # ax[1].set_xlim(-0.25, 1.25)
    # ax[1].set_ylim(-0.25, 0.25)
    ax[1].set_xlim(-0.5, 1.5)
    ax[1].set_ylim(-1, 1)





    # triangle_points: coordinates, shape = [2, 75590]
    triangle_points = data_output.triangle_points.numpy()
    x_mesh = triangle_points[0, :]
    y_mesh = triangle_points[1, :]

    # triangles: triangle indices, shape = [53736, 3]
    triangles = data_output.triangles.numpy()

    # pos: the location where x is defined, shape = [56827, 2]
    pos = data_output.pos.numpy()

    values = data.x[:, 1].numpy() - data_output.x[:, 1].numpy()

    # interpolate to triangle coordinates
    points_mesh = np.vstack((x_mesh, y_mesh)).T
    interpolated_values = griddata(pos, values, points_mesh, method='linear')

    nan_mask = np.isnan(interpolated_values)
    if np.any(nan_mask):
        interpolated_values[nan_mask] = griddata(pos, values, points_mesh[nan_mask], method='nearest')

    mesh_triangulation = tri.Triangulation(x_mesh, y_mesh, triangles)


    tpc = ax[2].tripcolor(mesh_triangulation, interpolated_values, shading='gouraud', cmap="bwr", vmin=-28, vmax=28)

    plt.colorbar(tpc, ax=ax[2], label="Ux-Ux_pred")


    ax[2].triplot(mesh_triangulation, color="black", linewidth=0.3, alpha=0.3)

    ax[2].set_title("Ux_error")
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("Y")
    # ax.axis("equal")  
    # ax[2].set_xlim(-0.25, 1.25)
    # ax[2].set_ylim(-0.25, 0.25)
    ax[2].set_xlim(-0.5, 1.5)
    ax[2].set_ylim(-1, 1)
    
    
    
    


    tpc = ax[3].tripcolor(mesh_triangulation, interpolated_values/U_inf, shading='gouraud', cmap="bwr", vmin=-0.2, vmax=0.2)
    plt.colorbar(tpc, ax=ax[3], label="(Ux-Ux_pred)/U_inf")


    ax[3].triplot(mesh_triangulation, color="black", linewidth=0.3, alpha=0.3)


    ax[3].set_title("Ux_error/U_inf")
    ax[3].set_xlabel("X")
    ax[3].set_ylabel("Y")
    # ax.axis("equal")  
    # ax[3].set_xlim(-0.25, 1.25)
    # ax[3].set_ylim(-0.25, 0.25)
    ax[3].set_xlim(-0.5, 1.5)
    ax[3].set_ylim(-1, 1)


    plt.savefig(f'Ux_{data_name}_{name}.png')
    plt.close()