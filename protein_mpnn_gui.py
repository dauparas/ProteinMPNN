"""A no-extra-dependencies desktop interface for ProteinMPNN.

Launch with ``bin\\pmpnn-gui.bat`` on Windows, or run this module with the
same Python interpreter used for ProteinMPNN.
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


REPO_ROOT = Path(__file__).resolve().parent
RUNNER = REPO_ROOT / "protein_mpnn_run.py"
MODELS = ("v_48_020", "v_48_010", "v_48_030", "v_48_002")


class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, highlightthickness=0, background="#f6f8fb")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.content = ttk.Frame(canvas, padding=18)
        self.content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        self.canvas = canvas

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class ProteinMPNNApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ProteinMPNN Studio")
        self.geometry("1080x760")
        self.minsize(900, 620)
        self.configure(background="#f6f8fb")
        icon_path = REPO_ROOT / "assets" / "proteinmpnn.ico"
        if icon_path.is_file():
            try:
                self.iconbitmap(default=str(icon_path))
            except tk.TclError:
                pass
        self.process = None
        self.log_queue = queue.Queue()
        self._setup_style()
        self._variables()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f6f8fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#f6f8fb", foreground="#172033", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#172a46", foreground="white", font=("Segoe UI Semibold", 24))
        style.configure("SubTitle.TLabel", background="#172a46", foreground="#c8d7ee", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#ffffff", foreground="#172a46", font=("Segoe UI Semibold", 12))
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=(12, 7))
        style.configure("Accent.TButton", background="#1676d2", foreground="white")
        style.map("Accent.TButton", background=[("active", "#0f5eaf")])
        style.configure("TEntry", padding=6)
        style.configure("TNotebook", background="#f6f8fb", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Segoe UI Semibold", 10))

    def _variables(self):
        self.input_kind = tk.StringVar(value="pdb")
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value=str(REPO_ROOT / "outputs" / "design"))
        self.design_chains = tk.StringVar()
        self.num_sequences = tk.StringVar(value="8")
        self.batch_size = tk.StringVar(value="1")
        self.temperature = tk.StringVar(value="0.1")
        self.model_name = tk.StringVar(value="v_48_020")
        self.ca_only = tk.BooleanVar(value=False)
        self.soluble = tk.BooleanVar(value=False)
        self.seed = tk.StringVar(value="")
        self.backbone_noise = tk.StringVar(value="0.0")
        self.max_length = tk.StringVar(value="200000")
        self.model_weights = tk.StringVar()
        self.fixed_positions = tk.StringVar()
        self.tied_positions = tk.StringVar()
        self.aa_bias = tk.StringVar()
        self.save_score = tk.BooleanVar(value=False)
        self.save_probs = tk.BooleanVar(value=False)
        self.score_only = tk.BooleanVar(value=False)
        self.conditional_probs = tk.BooleanVar(value=False)
        self.unconditional_probs = tk.BooleanVar(value=False)
        self.fasta_path = tk.StringVar()
        self.chain_id_jsonl = tk.StringVar()
        self.fixed_jsonl = tk.StringVar()
        self.tied_jsonl = tk.StringVar()
        self.omit_jsonl = tk.StringVar()
        self.bias_jsonl = tk.StringVar()
        self.command_preview = tk.StringVar(value="Choose a PDB or parsed JSONL file to begin.")
        self.status = tk.StringVar(value="Ready to design")

    def _build(self):
        header = ttk.Frame(self, style="Card.TFrame")
        header.configure(style="Card.TFrame")
        header.pack(fill="x")
        banner = tk.Frame(header, background="#172a46", padx=28, pady=18)
        banner.pack(fill="x")
        ttk.Label(banner, text="ProteinMPNN Studio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(banner, text="Structure-guided protein sequence design — local, reproducible, and ready to run.", style="SubTitle.TLabel").pack(anchor="w", pady=(3, 0))

        body = ttk.Frame(self, padding=(24, 18, 24, 12))
        body.pack(fill="both", expand=True)
        notebook = ttk.Notebook(body)
        notebook.pack(fill="both", expand=True)
        design_tab = ScrollFrame(notebook)
        advanced_tab = ScrollFrame(notebook)
        run_tab = ttk.Frame(notebook, padding=18)
        notebook.add(design_tab, text="Design setup")
        notebook.add(advanced_tab, text="Constraints & advanced")
        notebook.add(run_tab, text="Run & results")
        self._build_design(design_tab.content)
        self._build_advanced(advanced_tab.content)
        self._build_run(run_tab)
        self._update_preview()

    def _card(self, parent, title, note=None):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.pack(fill="x", pady=(0, 14))
        ttk.Label(card, text=title, style="Section.TLabel").pack(anchor="w")
        if note:
            ttk.Label(card, text=note, background="#ffffff", foreground="#61708a", wraplength=850).pack(anchor="w", pady=(3, 12))
        return card

    def _field(self, parent, label, variable, browse=None, width=72, row=None):
        frame = ttk.Frame(parent, style="Card.TFrame")
        if row is None:
            frame.pack(fill="x", pady=5)
        else:
            frame.grid(row=row, column=0, sticky="ew", pady=5)
        ttk.Label(frame, text=label, background="#ffffff", width=22).pack(side="left")
        ttk.Entry(frame, textvariable=variable, width=width).pack(side="left", fill="x", expand=True)
        if browse:
            ttk.Button(frame, text="Browse…", command=browse).pack(side="left", padx=(8, 0))

    def _build_design(self, parent):
        input_card = self._card(parent, "1. Input structure", "Choose a normal PDB file for one structure, or a parsed JSONL file for a batch.")
        radio = ttk.Frame(input_card, style="Card.TFrame")
        radio.pack(anchor="w", pady=(0, 8))
        ttk.Radiobutton(radio, text="PDB structure", variable=self.input_kind, value="pdb", command=self._update_preview).pack(side="left", padx=(0, 18))
        ttk.Radiobutton(radio, text="Parsed JSONL batch", variable=self.input_kind, value="jsonl", command=self._update_preview).pack(side="left")
        self._field(input_card, "Input file", self.input_path, self._choose_input)

        output_card = self._card(parent, "2. Output folder", "A new sequence FASTA file will be written here. Existing result folders are safe to reuse.")
        self._field(output_card, "Output folder", self.output_path, self._choose_output)

        design_card = self._card(parent, "3. Design settings")
        self._field(design_card, "Chains to design", self.design_chains)
        ttk.Label(design_card, text="Leave blank to redesign every chain. Example: A C", background="#ffffff", foreground="#61708a").pack(anchor="w", padx=176)
        grid = ttk.Frame(design_card, style="Card.TFrame")
        grid.pack(fill="x", pady=(14, 0))
        for column in range(3): grid.columnconfigure(column, weight=1)
        for column, (label, variable) in enumerate((("Sequences", self.num_sequences), ("Batch size", self.batch_size), ("Temperatures", self.temperature))):
            box = ttk.Frame(grid, style="Card.TFrame")
            box.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 10, 0))
            ttk.Label(box, text=label, background="#ffffff").pack(anchor="w")
            ttk.Entry(box, textvariable=variable).pack(fill="x", pady=(4, 0))
        model_card = self._card(parent, "4. Model")
        model_row = ttk.Frame(model_card, style="Card.TFrame")
        model_row.pack(fill="x")
        ttk.Label(model_row, text="Model checkpoint", background="#ffffff", width=22).pack(side="left")
        ttk.Combobox(model_row, textvariable=self.model_name, values=MODELS, state="readonly", width=18).pack(side="left")
        ttk.Checkbutton(model_row, text="CA-only model", variable=self.ca_only).pack(side="left", padx=20)
        ttk.Checkbutton(model_row, text="Soluble-protein model", variable=self.soluble).pack(side="left")

    def _build_advanced(self, parent):
        constraints = self._card(parent, "Direct constraints", "These apply to every target. Residue indices are one-based positions within a chain, not PDB residue numbers.")
        self._field(constraints, "Fixed positions", self.fixed_positions)
        self._field(constraints, "Tied positions", self.tied_positions)
        self._field(constraints, "Amino-acid bias", self.aa_bias)
        ttk.Label(constraints, text="Examples:  fixed A:1-8,23;C:10   •   tied A:1=C:1;A:2=C:2   •   bias D:1.39,E:1.39", background="#ffffff", foreground="#61708a", wraplength=850).pack(anchor="w", padx=176, pady=(4, 0))

        output = self._card(parent, "Additional outputs")
        row = ttk.Frame(output, style="Card.TFrame")
        row.pack(anchor="w")
        ttk.Checkbutton(row, text="Save per-sequence scores", variable=self.save_score).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(row, text="Save probabilities", variable=self.save_probs).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(row, text="Score input sequence only", variable=self.score_only).pack(side="left")
        row2 = ttk.Frame(output, style="Card.TFrame")
        row2.pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(row2, text="Conditional probabilities", variable=self.conditional_probs).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(row2, text="Unconditional probabilities", variable=self.unconditional_probs).pack(side="left")
        self._field(output, "FASTA to score", self.fasta_path, lambda: self._choose_file(self.fasta_path, [("FASTA files", "*.fa *.fasta *.faa"), ("All files", "*.*")]))

        tuning = self._card(parent, "Model tuning")
        self._field(tuning, "Weights folder", self.model_weights, lambda: self._choose_directory(self.model_weights))
        self._field(tuning, "Random seed", self.seed)
        self._field(tuning, "Backbone noise (Å)", self.backbone_noise)
        self._field(tuning, "Maximum length", self.max_length)

        batch = self._card(parent, "Per-target JSONL settings", "Optional advanced dictionaries for parsed JSONL batch workflows.")
        self._field(batch, "Chain assignment", self.chain_id_jsonl, lambda: self._choose_file(self.chain_id_jsonl, [("JSONL files", "*.jsonl"), ("All files", "*.*")]))
        self._field(batch, "Fixed positions", self.fixed_jsonl, lambda: self._choose_file(self.fixed_jsonl, [("JSONL files", "*.jsonl"), ("All files", "*.*")]))
        self._field(batch, "Tied positions", self.tied_jsonl, lambda: self._choose_file(self.tied_jsonl, [("JSONL files", "*.jsonl"), ("All files", "*.*")]))
        self._field(batch, "Omit amino acids", self.omit_jsonl, lambda: self._choose_file(self.omit_jsonl, [("JSONL files", "*.jsonl"), ("All files", "*.*")]))
        self._field(batch, "Residue bias", self.bias_jsonl, lambda: self._choose_file(self.bias_jsonl, [("JSONL files", "*.jsonl"), ("All files", "*.*")]))

    def _build_run(self, parent):
        top = self._card(parent, "Ready when you are", "The run uses this repository’s Python environment and writes all model output to the folder you selected.")
        ttk.Label(top, textvariable=self.status, background="#ffffff", foreground="#1676d2", font=("Segoe UI Semibold", 11)).pack(anchor="w", pady=(0, 10))
        button_row = ttk.Frame(top, style="Card.TFrame")
        button_row.pack(anchor="w")
        self.run_button = ttk.Button(button_row, text="Run ProteinMPNN", style="Accent.TButton", command=self._start_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(button_row, text="Stop", command=self._stop_run, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(button_row, text="Open output folder", command=self._open_output).pack(side="left")
        preview = self._card(parent, "Command preview")
        ttk.Label(preview, textvariable=self.command_preview, background="#ffffff", foreground="#42526b", wraplength=900, justify="left").pack(anchor="w")
        log_card = self._card(parent, "Live activity")
        self.log = tk.Text(log_card, height=17, wrap="word", borderwidth=0, background="#101827", foreground="#d8e5f5", insertbackground="white", font=("Cascadia Mono", 9), padx=12, pady=10)
        self.log.pack(fill="both", expand=True)

    def _choose_input(self):
        kinds = [("PDB files", "*.pdb *.ent"), ("JSONL files", "*.jsonl"), ("All files", "*.*")]
        self._choose_file(self.input_path, kinds)

    def _choose_file(self, variable, filetypes):
        path = filedialog.askopenfilename(title="Choose file", filetypes=filetypes)
        if path:
            variable.set(path)
            self._update_preview()

    def _choose_output(self):
        self._choose_directory(self.output_path)

    def _choose_directory(self, variable):
        path = filedialog.askdirectory(title="Choose folder")
        if path:
            variable.set(path)
            self._update_preview()

    def _arguments(self):
        path = self.input_path.get().strip()
        output = self.output_path.get().strip()
        if not path:
            raise ValueError("Choose an input PDB or parsed JSONL file.")
        if not Path(path).is_file():
            raise ValueError(f"Input file does not exist:\n{path}")
        if not output:
            raise ValueError("Choose an output folder.")
        try:
            sequences, batch = int(self.num_sequences.get()), int(self.batch_size.get())
            if sequences < 1 or batch < 1 or sequences % batch:
                raise ValueError
        except ValueError:
            raise ValueError("Sequences and batch size must be positive integers, and sequences must be divisible by batch size.")
        analysis_modes = sum((self.score_only.get(), self.conditional_probs.get(), self.unconditional_probs.get()))
        if analysis_modes > 1:
            raise ValueError("Choose only one analysis mode: score-only, conditional probabilities, or unconditional probabilities.")
        args = [str(RUNNER), "--out_folder", output, "--num_seq_per_target", str(sequences), "--batch_size", str(batch), "--sampling_temp", self.temperature.get().strip(), "--model_name", self.model_name.get()]
        args += ["--pdb-path" if self.input_kind.get() == "pdb" else "--jsonl-path", path]
        direct = (("--design-chains", self.design_chains.get()), ("--fixed-positions", self.fixed_positions.get()), ("--tied-positions", self.tied_positions.get()), ("--aa-bias", self.aa_bias.get()), ("--path_to_model_weights", self.model_weights.get()), ("--seed", self.seed.get()), ("--backbone_noise", self.backbone_noise.get()), ("--max_length", self.max_length.get()), ("--path_to_fasta", self.fasta_path.get()), ("--chain_id_jsonl", self.chain_id_jsonl.get()), ("--fixed_positions_jsonl", self.fixed_jsonl.get()), ("--tied_positions_jsonl", self.tied_jsonl.get()), ("--omit_AA_jsonl", self.omit_jsonl.get()), ("--bias_AA_jsonl", self.bias_jsonl.get()))
        for flag, value in direct:
            if value.strip(): args += [flag, value.strip()]
        for flag, value in (("--ca_only", self.ca_only.get()), ("--use_soluble_model", self.soluble.get())):
            if value:
                args.append(flag)
        for flag, value in (("--save_score", self.save_score.get()), ("--save_probs", self.save_probs.get()), ("--score_only", self.score_only.get()), ("--conditional_probs_only", self.conditional_probs.get()), ("--unconditional_probs_only", self.unconditional_probs.get())):
            if value:
                args += [flag, "1"]
        return args

    def _update_preview(self):
        try:
            command = [Path(sys.executable).name] + self._arguments()
            self.command_preview.set(" ".join(f'"{part}"' if " " in part else part for part in command))
        except ValueError as error:
            self.command_preview.set(str(error))

    def _start_run(self):
        try:
            args = self._arguments()
        except ValueError as error:
            messagebox.showerror("ProteinMPNN Studio", str(error))
            return
        Path(self.output_path.get().strip()).mkdir(parents=True, exist_ok=True)
        self.log.delete("1.0", "end")
        self.status.set("Starting ProteinMPNN…")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self._run_worker, args=(args,), daemon=True).start()
        self.after(100, self._drain_log)

    def _run_worker(self, args):
        try:
            self.process = subprocess.Popen([sys.executable] + args, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in self.process.stdout:
                self.log_queue.put(("line", line))
            self.log_queue.put(("complete", self.process.wait()))
        except Exception as error:
            self.log_queue.put(("error", str(error)))

    def _drain_log(self):
        active = self.process is not None
        while True:
            try:
                kind, value = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self.log.insert("end", value)
                self.log.see("end")
                if value.strip(): self.status.set(value.strip().replace("\r", ""))
            elif kind == "complete":
                active = False
                self.process = None
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.status.set("Design completed successfully." if value == 0 else f"Design stopped or failed (exit code {value}).")
            elif kind == "error":
                active = False
                self.process = None
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.status.set("Could not start ProteinMPNN.")
                self.log.insert("end", f"\nERROR: {value}\n")
        if active or self.process is not None:
            self.after(100, self._drain_log)

    def _stop_run(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status.set("Stopping ProteinMPNN…")

    def _open_output(self):
        folder = Path(self.output_path.get().strip())
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def _close(self):
        if self.process and self.process.poll() is None and not messagebox.askyesno("ProteinMPNN Studio", "A design is still running. Stop it and close?"):
            return
        self._stop_run()
        self.destroy()


if __name__ == "__main__":
    ProteinMPNNApp().mainloop()
