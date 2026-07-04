from train_props import train_prop_model 
props = [
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "optb88vdw_total_energy",
    "ehull",
    "mbj_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "slme",
    "spillage",
]
prop = props[0]
train_prop_model(learning_rate=1e-3,name="matformer", prop=prop, pyg_input=True, n_epochs=1000, batch_size=64, use_lattice=True, output_dir=f"ouput_{prop}", use_angle=False, save_dataloader=False, atom_features = "cgcnn",random_seed=123,test_only=False)
