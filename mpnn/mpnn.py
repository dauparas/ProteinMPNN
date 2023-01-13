from ..protein_mpnn_run import main as mpnn_main

from ..helper_scripts.parse_multiple_chains import main as parse_multiple_chains
from ..helper_scripts.assign_fixed_chains import main as assign_fixed_chains
from ..helper_scripts.make_fixed_positions_dict import main as make_fixed_positions

## Don't need omitAA script
from ..helper_scripts.make_bias_AA import main as make_bias_AA
from ..helper_scripts.make_tied_positions_dict import main as make_tied_positions
from ..helper_scripts.make_bias_per_res_dict import main as make_bias_per_res

from pyrosetta.distributed import packed_pose
from collections import namedtuple
from collections.abc import Iterable
import tempfile


class MPNN:
    def __init__(
        self,
        pose,
        fixed_chains="",
        fixed_positions="",
        omit_AAs=[],
        bias_AA={},
        tied_positions="",
        bias_by_res={},
    ):
        """Initialize the MPNN settings"""

        # ugh
        # Blame the bad design of the MPNN code
        self.args = namedtuple(
            "args",
            "out_folder pdb_path jsonl_path chain_id_jsonl fixed_positions_jsonl omit_AAs bias_AA_jsonl tied_positions_jsonl bias_by_res_jsonl omit_AA_jsonl suppress_print ca_only path_to_model_weights model_name seed save_score save_probs score_only conditional_probs_only conditional_probs_only_backbone unconditional_probs_only pssm_multi pssm_threshold pssm_log_odds_flag pssm_bias_flag backbone_noise num_seq_per_target batch_size max_length sampling_temp pssm_jsonl path_to_fasta pdb_path_chains",
        )

        ## Standard Options we aren't changing
        self.args.suppress_print = 0
        self.args.ca_only = False
        self.args.path_to_model_weights = (
            ""  # the protein_mpnn_run module should find the weights on its own
        )
        self.args.model_name = "v_48_020"
        self.args.seed = 0
        self.args.save_score = 0
        self.args.save_probs = 0
        self.args.score_only = 0
        self.args.conditional_probs_only = 0
        self.args.conditional_probs_only_backbone = 0
        self.args.unconditional_probs_only = 0
        self.args.pssm_multi = 0.0
        self.args.pssm_threshold = 0.0
        self.args.pssm_log_odds_flag = 0
        self.args.pssm_bias_flag = 0

        self.args.backbone_noise = 0
        self.args.num_seq_per_target = 1
        self.args.batch_size = 1
        self.args.max_length = 200000
        self.args.sampling_temp = 0.1

        self.args.pssm_jsonl = ""
        self.args.path_to_fasta = ""
        self.args.pdb_path_chains = ""  ## Assume given PDB is chained A, B, C...

        ## Options we are changing

        # set the output folder (temp, will create pose(s) at end of run)
        self.temp_output_folder = tempfile.TemporaryDirectory()
        self.args.out_folder = self.temp_output_folder.name

        # set the input folder - this can handle
        self.temp_input_folder = tempfile.TemporaryDirectory()
        self.args.pdb_path = self.temp_input_folder.name
        if isinstance(pose, Iterable):
            for i, p in enumerate(pose):
                wpose = packed_pose.to_pose(p)
                wpose.dump_pdb(
                    f"{self.args.pdb_path}/{packed_pose.to_packed(wpose).scores['name']}.pdb"
                )
        else:
            wpose = packed_pose.to_pose(pose)
            wpose.dump_pdb(
                f"{self.args.pdb_path}/{packed_pose.to_packed(wpose).scores['name']}.pdb"
            )

        # Create temp folder to store jsonl files
        self.temp_jsonl_folder = tempfile.TemporaryDirectory()
        jsonl_folder_name = self.temp_jsonl_folder.name

        jsonl_path = ""
        chain_id_jsonl = ""
        fixed_positions_jsonl = ""
        omit_AAs = ""
        bias_AA_jsonl = ""
        tied_positions_jsonl = ""
        bias_by_res_jsonl = ""
        omit_AA_jsonl = ""

    def __del__(self):
        self.temp_output_folder.cleanup()
        self.temp_input_folder.cleanup()
        self.temp_jsonl_folder.cleanup()

    def execute(self):
        """Execute the MPNN"""
        mpnn_main(self.args)
