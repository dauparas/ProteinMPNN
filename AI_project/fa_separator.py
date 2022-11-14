import pathlib

DESIGN_RESULT = "./vanila_model/mRFP_SWISS_MODEL.fa"
TARGET_DIR = "./vanila_model/separated"


if __name__ == "__main__":
    # Read the fasta file
    fasta = open(DESIGN_RESULT, "r")
    lines = fasta.readlines()
    fasta.close()

    # Write the fasta file
    pathlib.Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)

    for i in range(0, len(lines), 2):
        id = lines[i].split(",")[0][1:]

        with open(f"{pathlib.Path(TARGET_DIR) / id}.fa", "wt") as fa:
            fa.write(lines[i])
            fa.write(lines[i+1])

    print("Done")
