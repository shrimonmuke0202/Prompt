import sys
#sys.path.append("/yourpath/comformer"
from train_props import train_prop_model 
props = [
    "e_form",
    "gap pbe",
    "bulk modulus",
    "shear modulus",
]
train_prop_model(learning_rate=1e-3,name="matformer", dataset="megnet", prop=props[-1], pyg_input=True, n_epochs=1000, max_neighbors=25, cutoff=4.0, batch_size=64, use_lattice=False, output_dir="bg_mp", use_angle=False, save_dataloader=False, random_seed=123,
                mp_id_list="shear")
