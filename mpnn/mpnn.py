from .protein_mpnn_run import main as mpnn_main

from .types import Args
from .helper_scripts.parse_multiple_chains import main as parse_multiple_chains
from .helper_scripts.assign_fixed_chains import main as assign_fixed_chains
from .helper_scripts.make_fixed_positions_dict import main as make_fixed_positions
from .helper_scripts.make_bias_AA import main as make_bias_AA
from .helper_scripts.make_tied_positions_dict import main as make_tied_positions

from opbdesign.core.cluster import Task
from pyrosetta.rosetta.protocols.simple_moves import SimpleThreadingMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.pose import get_resnums_for_chain, setPoseExtraScore
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.distributed import packed_pose
from collections import namedtuple
import tempfile
import glob
import re


class MPNN(Task):
    def __init__(
        self,
        poses,
        nstruct=1,
        designable_chains="",
        fixed_positions="",
        omit_AAs=[],
        bias_AA={},
        tied_positions="",
        homooligomer=False,
        sampling_temp=0.3,
        repack_neighbors=True,
        name="MPNN_Run",
    ):
        """Initialize the MPNN settings

        Args:
            pose (pyrosetta.Pose): The pose to design
            designable_chains (str, optional): The chains to design. Defaults to ""
                                                Format: "A B C"
            fixed_positions (str, optional): The fixed positions. Defaults to ""
                                                Format: "1 2 3 4, 3 5 8, 4"
                                                Must have same number of lists as designable_chains
            omit_AAs (list, optional): The AAs to omit. Defaults to []
                                                Format: ["K", "C"]
            bias_AA (dict, optional): The bias for each AA. Defaults to {}
                                                Format: {"A": 0.1, "C": 0.2}
                                                positive bias shows up more, negative less
            tied_positions (str, optional): Positions that must be the same AA. Defaults to ""
                                                Format: "1 2 3 4, 1 2 3 4"
                                                Lists must be the same length
                                                There must be N lists for N chains defined in designable_chains
            homo_oligomer (bool, optional): Whether the pose is a homooligomer. Defaults to False


        """

        super().__init__(name=name)

        if not isinstance(poses, list):
            poses = [poses]

        self.input_poses = poses
        self.args = Args()

        #############################
        ## Options we are changing ##
        #############################

        ##### OUTPUT FOLDER SETUP #####

        # set the output folder (temp, will create pose(s) at end of run)
        self.temp_output_folder = tempfile.TemporaryDirectory()
        self.args.out_folder = self.temp_output_folder.name

        # set the input folder - this can handle
        self.temp_input_folder = tempfile.TemporaryDirectory()
        input_path = self.temp_input_folder.name
        self.pose_dict = {}
        for i, ppose in enumerate(self.input_poses):
            ppose.pose.dump_pdb(f"{input_path}/{self.name}_{i}.pdb")
            self.pose_dict[f"{self.name}_{i}"] = ppose

        ##### Parameters with no additional setup #####

        jsonl_folder = self.temp_output_folder.name
        self.args.num_seq_per_target = nstruct
        self.args.omit_AAs = omit_AAs
        self.args.sampling_temp = str(sampling_temp)  # must be str
        self.designable_chains = designable_chains
        self.repack_neighbors = repack_neighbors

        ###### PARSE CHAINS ######

        ## Parse the PDB into chains in format MPNN reads
        path_for_parsed_chains = f"{jsonl_folder}/parsed_pdbs.jsonl"
        ParseMultiChain = namedtuple(
            "ParseMultiChain", "input_path output_path ca_only"
        )
        parse_multiple_chains(
            ParseMultiChain(input_path, path_for_parsed_chains, self.args.ca_only)
        )
        self.args.jsonl_path = path_for_parsed_chains

        ##### CONDITIONAL INPUTS #####

        ## Which chains are designable
        if designable_chains != "":
            path_for_assigned_chains = f"{jsonl_folder}/assigned_pdbs.jsonl"
            AssignFixedChains = namedtuple(
                "AssignFixedChains", "input_path output_path chain_list"
            )
            assign_fixed_chains(
                AssignFixedChains(
                    input_path=path_for_parsed_chains,
                    output_path=path_for_assigned_chains,
                    chain_list=designable_chains,
                )
            )
            self.args.chain_id_jsonl = path_for_assigned_chains
        else:
            self.args.chain_id_jsonl = ""

        ## Which positions are fixed - must define designable chains as well
        if fixed_positions != "":
            if designable_chains == "":
                raise ValueError(
                    "Cannot specify fixed positions without designable chains"
                )

            path_for_fixed_positions = f"{jsonl_folder}/fixed_positions.jsonl"
            MakeFixedPositions = namedtuple(
                "MakeFixedPositions",
                "input_path output_path chain_list position_list specify_non_fixed",
            )
            make_fixed_positions(
                MakeFixedPositions(
                    input_path=path_for_parsed_chains,
                    output_path=path_for_fixed_positions,
                    chain_list=designable_chains,
                    position_list=fixed_positions,
                    specify_non_fixed=False,
                )
            )
            self.args.fixed_positions_jsonl = path_for_fixed_positions
        else:
            self.args.fixed_positions_jsonl = ""

        ## Bias the AAs
        if bias_AA != {}:

            path_for_bias_AA = f"{jsonl_folder}/bias_AA.jsonl"
            MakeBiasAA = namedtuple("MakeBiasAA", "output_path AA_list bias_list")

            AA_list = " ".join(bias_AA.keys())
            bias_list = " ".join([str(x) for x in bias_AA.values()])
            make_bias_AA(
                MakeBiasAA(
                    output_path=path_for_bias_AA,
                    AA_list=AA_list,
                    bias_list=bias_list,
                )
            )
            self.args.bias_AA_jsonl = path_for_bias_AA
        else:
            self.args.bias_AA_jsonl = ""

        ## Define which positions must be the same AA
        if tied_positions != "":
            path_for_tied_positions = f"{jsonl_folder}/tied_positions.jsonl"
            MakeTiedPositions = namedtuple(
                "MakeTiedPositions",
                "input_path output_path chain_list position_list homooligomer",
            )
            make_tied_positions(
                MakeTiedPositions(
                    input_path=path_for_parsed_chains,
                    output_path=path_for_tied_positions,
                    chain_list=designable_chains,
                    position_list=tied_positions,
                    homooligomer=0 if not homooligomer else 1,
                )
            )
            self.args.tied_positions_jsonl = path_for_tied_positions
        else:
            self.args.tied_positions_jsonl = ""

        # Print contents of temp folders
        self.logger.debug("Input folder contents:")
        for file in glob.glob(f"{input_path}/*"):
            self.logger.debug(file)
            self.logger.debug(open(file).read())
        self.logger.debug("Output folder contents:")
        for file in glob.glob(f"{jsonl_folder}/*"):
            self.logger.debug(file)
            self.logger.debug(open(file).read())

    def execute(self):
        """Execute the MPNN"""
        mpnn_main(self.args)

        # Read the self.temp_output_folder for the output pose(s)
        sequence_folder = f"{self.temp_output_folder.name}/seqs"
        output_structures = []
        p = re.compile("[BJOUXZ]")
        for file in glob.glob(f"{sequence_folder}/*.fa"):
            wpose = self.pose_dict[file.split("/")[-1].split(".")[0]].pose.clone()
            with open(file) as f:
                for i, sequence in enumerate(
                    [line.rstrip() for line in f.readlines()][3::2]
                ):
                    sequences = {
                        a: b
                        for a, b in zip(
                            self.designable_chains.split(), sequence.split("/")
                        )
                    }

                    self.logger.debug(sequences)
                    reject = False
                    for chain in sequences:
                        if p.match(sequence):
                            reject = True
                            break
                        if not reject:
                            chain_start = get_resnums_for_chain(wpose, chain)[1]
                            stm = SimpleThreadingMover()
                            stm.set_sequence(sequences[chain], chain_start)
                            stm.set_pack_neighbors(self.repack_neighbors)
                            stm.apply(wpose)
                    if reject:
                        continue

                    ppose = packed_pose.to_packed(wpose)
                    self.logger.debug(ppose.scores.keys())
                    output_structures.append(ppose)

        return output_structures


class FastDesign_MPNN(Task):
    """This Task generates structures using a FastDesign like protocol
    but using MPNN instead of the Rosetta packer.  The overall routine
    is to:
        1) Generate N (nstruct_per_cycle) sequences using MPNN on the input pose
        2) Thread the sequences onto the input pose allowing all rotamers to repack
            - The MPNN Task by default allows all neighbors to repack upon threading
            - Can change the "repack_neighbors" param to False if you dont want this
        3) Minimizes each structure
            - By default, none of the backbones are allowed to move
            - By default, chi and jumps can move
            - Control which chain BB can move by the "moveable_chains" param
            - Control which residue BBs can move by the "moveable_residues" param
        4) Select the top K (select_top) structures and repeat from step 1, generating
            N (nstruct_per_cycle) new sequences for eack K input structures
        5) Repeat steps 2-4 for C (cycles) times
        6) Return the top K (select_top) structures from the final cycle
    """

    def __init__(
        self,
        pose,
        cycles=3,
        nstruct_per_cycle=5,
        select_top=2,
        designable_chains="",
        movable_chains="",
        fixed_positions="",
        movable_residues="",
        name="FastDesign_MPNN",
        **mpnn_args,
    ):
        """Initialize the FastDesign_MPNN Task

        Args:
            pose (Pose): The input pose
            cycles (int, optional): The number of cycles to run. Defaults to 3.
            nstruct_per_cycle (int, optional): The number of structures to generate per cycle. Defaults to 5.
            select_top (int, optional): The number of structures to select from each cycle. Defaults to 2.
            designable_chains (str, optional): The chains to design. Defaults to "" = all chains.
            movable_chains (str, optional): The chains to allow backbone movement. Defaults to "" = no chains.
            fixed_positions (str, optional): The positions to fix. Defaults to "".
            movable_residues (str, optional): The residues to allow backbone movement. Defaults to "".
            name (str, optional): The name of the Task. Defaults to "FastDesign_MPNN".

            **mpnn_args: The arguments to pass to the MPNN Task, allowed kwargs are:
                - "omit_AAs" (str): The AAs to omit from the design
                - "bias_AA" (str): The AAs to bias the design towards
                - "tied_positions" (str): The positions within the pose that should be the same identity
                - "homooligomer" (bool): Whether or not the complex is a homooligomer
                - "sampling_temp" (float): The sampling temperature for the MPNN
                - "repack_neighbors" (bool): Whether or not to repack the neighbors of the designable positions
        """

        super().__init__(name=name)
        self.input_pose = pose
        self.poses = [pose]
        self.cycles = cycles

        if nstruct_per_cycle < select_top:
            raise ValueError(
                "nstruct_per_cycle must be greater than or equal to select_top"
            )
        self.nstruct_per_cycle = nstruct_per_cycle
        self.select_top = select_top
        self.designable_chains = designable_chains
        self.movable_chains = movable_chains
        self.fixed_positions = fixed_positions
        self.movable_residues = movable_residues

        allowed_keys = [
            "omit_AAs",
            "bias_AA",
            "tied_positions",
            "homooligomer",
            "sampling_temp",
            "repack_neighbors",
        ]
        for key in mpnn_args:
            if key not in allowed_keys:
                raise ValueError(f"Key {key} not allowed in mpnn_args")
        self.mpnn_args = mpnn_args

    def execute(self):

        n_cycles = 3
        for _ in range(n_cycles):
            ### Run MPNN
            mpnn_task = MPNN(
                self.poses,
                nstruct=self.nstruct_per_cycle,
                designable_chains=self.designable_chains,
                fixed_positions=self.fixed_positions,
                **self.mpnn_args,
            )
            new_pposes = mpnn_task.execute()

            self.poses = []
            for ppose in new_pposes:
                ### Minimize
                wpose = packed_pose.to_pose(ppose)
                mm = MoveMap()
                mm.set_bb(False)
                for residue in self.movable_residues.split():
                    mm.set_bb(residue, True)
                for chain in self.movable_chains.split():
                    for i in get_resnums_for_chain(wpose, chain):
                        mm.set_bb(i, True)
                mm.set_chi(True)
                mm.set_jump(True)

                minmover = MinMover()
                minmover.movemap(mm)
                minmover.apply(wpose)
                self.poses.append(packed_pose.to_packed(wpose))
            self.poses = sorted(self.poses, key=lambda x: x.pose.scores["total_score"])[
                : self.select_top
            ]

        return self.poses
