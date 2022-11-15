from sklearn.utils.random import sample_without_replacement

num_population = 1079

with open("/mnt/P41/Repositories/ProteinMPNN/training/pdb_2021aug02_RFP_DESIGNING/test_clusters.txt", "w") as f:
    samples = sample_without_replacement(num_population, int(0.1 * num_population))

    for sample in samples:
        f.write(f"{sample}\n")
