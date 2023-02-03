from dataclasses import dataclass, field


@dataclass
class Args:
    ## Standard Options we aren't changing
    suppress_print: int = 0
    ca_only: bool = False
    path_to_model_weights: str = ""
    model_name: str = "v_48_020"
    seed: int = 0
    save_score: int = 0
    save_probs: int = 0
    score_only: int = 0
    conditional_probs_only: int = 0
    conditional_probs_only_backbone: int = 0
    unconditional_probs_only: int = 0
    pssm_jsonl: str = ""
    pssm_multi: float = 0.0
    pssm_threshold: float = 0.0
    pssm_log_odds_flag: int = 0
    pssm_bias_flag: int = 0

    backbone_noise: float = 0
    batch_size: int = 1
    max_length: int = 200000
    sampling_temp: str = "0.3"

    path_to_fasta: str = ""
    pdb_path: str = ""  ## we're always parsing the jsonl SO we don't specify this
    pdb_path_chains: str = ""  ## Setting this to "" means all chains will be designed if no fixed chains are defined
    bias_by_res_jsonl: str = ""
    omit_AA_jsonl: str = ""

    ## Options we are changing
    out_folder: str = ""
    num_seq_per_target: int = 1
    omit_AAs: list = field(default_factory=list)
    jsonl_path: str = ""
    chain_id_jsonl: str = ""
    fixed_positions_jsonl: str = ""
    bias_AA_jsonl: str = ""
    tied_positions_jsonl: str = ""
