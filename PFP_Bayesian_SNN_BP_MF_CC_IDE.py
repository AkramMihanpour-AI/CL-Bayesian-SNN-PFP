"""Protein function prediction with a Bayesian-weighted SNN ensemble.

Readable research version (VS Code > Run; no command-line arguments).

Main method: ESM-2 embeddings -> three SNNs, trained independently with
CPC-Prot-teacher knowledge distillation -> Soft-F1 fine-tuning -> Bayesian
weighting of their prediction probabilities.

This file contains the SNN architecture and the training loops explicitly.
Only existing dataset, GO metric and Teacher utilities are imported.
The ESM-2 and CPC-Prot encoders are NOT pretrained by this script.

For an educational, read-only check, keep RUN_MODE='load_saved'.
For a fresh training run in a NEW directory, set RUN_MODE='train_new'.
"""

# =====================================================================
# 1. IMPORTS: all dependencies are visible here
# =====================================================================
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from snntorch import surrogate

from ak_ontology_new import Ontology
from ak_metrics_new_protein_centric import binary_to_term_sets, Fmax_Smin_AUPR
from run_pfp_efficiency_goproformer_ide import (
    create_model,
    build_ancestor_matrix,
    fast_protein_centric_fmax,
)
from compare_esm_pfp_ide import (
    read_pkl,
    load_X,
    check_labels,
    seed_all,
    load_state,
)


# =====================================================================
# 2. FILE PATHS AND EXPERIMENT SETTINGS
# =====================================================================
HERE = Path(__file__).resolve().parent

# The branch-specific GOProFormer split, including CPC-Prot (512-D) Train features.
GO_BRANCH = "CC"  # Change to "MF" or "BP" or "CC" for the other GO branch.
if GO_BRANCH not in ("MF", "CC"):
    raise ValueError("This separate multibranch research script supports MF and CC only")
DATA_SPLITS_PKL = HERE / f"go_dataset_splits_GOProFormer_original_{GO_BRANCH}_23_Pach-Conv_GRU.pkl"

GO_BRANCH_DIR = Path(
    r"E:\MyThesis\CPCProt_Clone_Code\CPCProt\data\pfp_goproformer_files\goa\random_split_leakage"
) / GO_BRANCH
TRAIN_LABELS_PKL = GO_BRANCH_DIR / "train.pkl"
VAL_LABELS_PKL = GO_BRANCH_DIR / "val.pkl"
TEST_LABELS_PKL = GO_BRANCH_DIR / "test.pkl"

GO_OBO_FILE = Path(
    r"E:\MyThesis\CPCProt_Clone_Code\CPCProt\data\pfp_goproformer_files\go.obo"
)

# ESM-2 t33 650M: one 1280-D embedding .pkl per protein, named by protein ID.
ESM2_EMBEDDING_DIR = Path(
    r"E:\MyThesis\testPytorch\GOProFormer\data\uniprotkb\yeast_sequences_rep_mean_esm2_t33_650m"
)

# GO classification Teacher trained using CPC-Prot 512-D representations.
CPC_TEACHER_CHECKPOINT = (
    HERE / f"pfp_efficiency_results_GOProFormer_original_{GO_BRANCH}" / "teacher_best.pt"
)

# Existing experimental checkpoints (read only in load_saved mode).
EXISTING_RESULTS_DIR = HERE / f"pfp_bayesian_snn_{GO_BRANCH}_new_run"

# Fresh training always writes to a SEPARATE directory.
NEW_RESULTS_DIR = HERE / f"pfp_bayesian_snn_{GO_BRANCH}_new_run"
REPORT_DIR = HERE / f"pfp_bayesian_snn_{GO_BRANCH}_reports"

# 'load_saved': inspect the previously trained models; NO training.
# 'train_new' : train three independent SNNs with KD -> Soft-F1.
RUN_MODE = "load_saved"  # train_new Later: "load_saved" for test; do not retrain after selection.

# Default False because the original held-out Test was already evaluated.
EVALUATE_TEST = True #True FOR TEST #False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_SEED = 42
NUM_LABELS = None  # Set from the selected branch split inside main()
INPUT_DIM_ESM2 = 1280
SNN_TIMESTEPS = 10
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128

KD_EPOCHS = 65
SOFTF1_EPOCHS = 30
KD_LR = 1e-3
SOFTF1_LR = 1e-4
SOFTF1_WEIGHT_DECAY = 1e-5
SOFTF1_PATIENCE = 8
KD_ALPHA = 0.6
KD_TEMPERATURE = 1.0
SOFTF1_LAMBDA = 0.15
GRAD_CLIP_NORM = 1.0

IC_STRENGTH = 0.5
IC_MAX = 5.0
IC_CAP = 2.0
BAYES_EVIDENCE = 32.0

# These are the three architectures from the original BaseSNN implementation.
SNN_CONFIGS = [
    {"hidden": (128,), "beta": 0.90, "dropout": 0.10},
    {"hidden": (256, 128), "beta": 0.85, "dropout": 0.15},
    {"hidden": (128, 64), "beta": 0.95, "dropout": 0.10},
]
SNN_SEEDS = [42, 55, 68]


# =====================================================================
# 3. DEFINE THE SPIKING NEURAL NETWORK (BaseSNN)
#    This class preserves the original forward-pass implementation.
# =====================================================================
class BaseSNN(nn.Module):
    """Constant input for 10 time steps; LIF hidden layer(s); analog readout."""

    def __init__(self, hidden=(128,), n_features=512, n_labels=None,
                 steps=10, beta=0.9, dropout=0.1):
        super().__init__()
        if not isinstance(n_labels, int) or n_labels < 1:
            raise ValueError("Specify the current branch's number of output labels")
        self.steps = steps
        sizes = (n_features,) + tuple(hidden)
        self.linears = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        self.lifs = nn.ModuleList([
            snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            for _ in hidden
        ])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(sizes[-1], n_labels)

    def forward(self, x):
        # x: [batch_size, 1280]; the same input is applied at every time step.
        membranes = [lif.init_leaky() for lif in self.lifs]
        logits_sum = None
        for _ in range(self.steps):
            h = x
            for i, (linear, lif) in enumerate(zip(self.linears, self.lifs)):
                h, membranes[i] = lif(linear(h), membranes[i])
                h = self.dropout(h)
            out = self.readout(h)
            logits_sum = out if logits_sum is None else logits_sum + out
        return logits_sum / self.steps  # logits, NOT probabilities


def new_snn(config):
    if NUM_LABELS is None or NUM_LABELS < 1:
        raise RuntimeError("Load and validate the GO branch split before creating an SNN")
    return BaseSNN(
        hidden=config["hidden"],
        n_features=INPUT_DIM_ESM2,
        n_labels=NUM_LABELS,
        steps=SNN_TIMESTEPS,
        beta=config["beta"],
        dropout=config["dropout"],
    ).to(DEVICE)


# =====================================================================
# 4. LOSS FUNCTIONS AND DATA LOADERS
# =====================================================================
def soft_f1_loss(logits, labels):
    """Differentiable F1 approximation over GO terms with positives in batch."""
    probabilities = torch.sigmoid(logits)
    tp = (probabilities * labels).sum(dim=0)
    fp = (probabilities * (1 - labels)).sum(dim=0)
    fn = ((1 - probabilities) * labels).sum(dim=0)

    active_terms = labels.sum(dim=0) > 0
    if active_terms.any().item():
        per_term_soft_f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        return 1 - per_term_soft_f1[active_terms].mean()
    return probabilities.sum() * 0.0


def make_loader(x, y, shuffle=False, seed=42, soft_targets=None, batch_size=BATCH_SIZE):
    tensors = [torch.from_numpy(x), torch.from_numpy(y)]
    if soft_targets is not None:
        tensors.append(soft_targets)
    dataset = TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


@torch.no_grad()
def predict_logits(model, x, y):
    model.eval()  # turns off Dropout for prediction
    loader = make_loader(x, y, batch_size=EVAL_BATCH_SIZE)
    parts = []
    for batch in loader:
        xb = batch[0].to(DEVICE)
        parts.append(model(xb).cpu().numpy())
    return np.concatenate(parts, axis=0)


def validation_fmax(model, x_val, y_val, ancestor_matrix):
    logits = predict_logits(model, x_val, y_val)
    prob = torch.sigmoid(torch.from_numpy(logits)).numpy()
    _, fmax = fast_protein_centric_fmax(prob, y_val, ancestor_matrix)
    return float(fmax)


# =====================================================================
# 5. TRAINING STAGE A: TEACHER -> STUDENT (KNOWLEDGE DISTILLATION)
# =====================================================================
def train_kd_model(config, seed, x_train, y_train, x_val, y_val,
                   teacher_probabilities, ancestor_matrix, checkpoint_path):
    seed_all(seed)
    model = new_snn(config)  # independent KD training from a NEW initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=KD_LR)
    bce = nn.BCEWithLogitsLoss()

    loader = make_loader(x_train, y_train, shuffle=True, seed=seed,
                         soft_targets=teacher_probabilities)
    best_fmax = -1.0

    for epoch in range(1, KD_EPOCHS + 1):
        model.train()
        for xb, yb, teacher_y in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            teacher_y = teacher_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            student_logits = model(xb)
            label_loss = bce(student_logits, yb)
            teacher_loss = bce(student_logits / KD_TEMPERATURE, teacher_y)
            loss = (1 - KD_ALPHA) * label_loss + (
                KD_ALPHA * (KD_TEMPERATURE ** 2) * teacher_loss
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        fmax = validation_fmax(model, x_val, y_val, ancestor_matrix)
        if fmax > best_fmax + 1e-8:
            best_fmax = fmax
            torch.save(model.state_dict(), checkpoint_path)
        if epoch == 1 or epoch % 10 == 0 or epoch == KD_EPOCHS:
            print(f"    KD epoch {epoch:03d}: VAL Fmax={fmax:.4f}, best={best_fmax:.4f}")

    return best_fmax


# =====================================================================
# 6. TRAINING STAGE B: FINE-TUNING WITH WEIGHTED BCE + SOFT-F1
# =====================================================================
def train_softf1_model(config, seed, kd_checkpoint, output_checkpoint,
                       x_train, y_train, x_val, y_val, ancestor_matrix, class_weights):
    seed_all(seed)
    model = new_snn(config)
    model.load_state_dict(load_state(kd_checkpoint, DEVICE), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SOFTF1_LR,
                                  weight_decay=SOFTF1_WEIGHT_DECAY)
    loader = make_loader(x_train, y_train, shuffle=True, seed=seed)

    # The KD checkpoint is also a valid epoch-0 candidate.
    best_fmax = validation_fmax(model, x_val, y_val, ancestor_matrix)
    torch.save(model.state_dict(), output_checkpoint)
    epochs_without_improvement = 0

    for epoch in range(1, SOFTF1_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            per_label_bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            weighted_bce = (per_label_bce * class_weights.unsqueeze(0)).mean()
            f1_loss = soft_f1_loss(logits, yb)
            loss = weighted_bce + SOFTF1_LAMBDA * f1_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        fmax = validation_fmax(model, x_val, y_val, ancestor_matrix)
        if fmax > best_fmax + 1e-8:
            best_fmax = fmax
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_checkpoint)
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == SOFTF1_EPOCHS:
            print(f"    Soft-F1 epoch {epoch:03d}: VAL Fmax={fmax:.4f}, best={best_fmax:.4f}")
        if epochs_without_improvement >= SOFTF1_PATIENCE:
            print("    Early stopping (Soft-F1)")
            break

    return best_fmax


# =====================================================================
# 7. BAYESIAN ENSEMBLE AND GO EVALUATION
# =====================================================================
def calculate_bayesian_weights(all_member_val_logits, y_val):
    """Generalized Bayesian weighting over FIXED models, NOT Bayesian layers."""
    y = torch.from_numpy(y_val)
    val_bce = []
    for logits in all_member_val_logits:
        loss = F.binary_cross_entropy_with_logits(torch.from_numpy(logits), y)
        val_bce.append(loss.item())

    scores = -BAYES_EVIDENCE * torch.tensor(val_bce, dtype=torch.float64)
    weights = torch.softmax(scores, dim=0).numpy()
    return weights


def weighted_ensemble_probabilities(all_member_logits, weights):
    probabilities = [torch.sigmoid(torch.from_numpy(z)).numpy()
                     for z in all_member_logits]
    return np.tensordot(weights, np.stack(probabilities), axes=(0, 0))


def evaluate_go(prob, labels, index_to_term, ontology, term_to_index):
    truth_sets = binary_to_term_sets(labels, index_to_term)
    threshold, fmax, smin, auprc = Fmax_Smin_AUPR(
        prob, truth_sets, index_to_term, ontology, set(term_to_index)
    )
    return {"Fmax": float(fmax), "Smin": float(smin),
            "AUPRC": float(auprc), "threshold": float(threshold)}


def save_results(filename, rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (REPORT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# 8. MAIN: EXPLICIT STEP-BY-STEP PIPELINE
# =====================================================================
def main():
    print("Device:", DEVICE)
    print("Run mode:", RUN_MODE)
    global NUM_LABELS
    if RUN_MODE not in ("load_saved", "train_new"):
        raise ValueError("RUN_MODE must be 'load_saved' or 'train_new'")

    # -------- Step 1: Check and load GO labels / protein IDs --------
    required_files = [DATA_SPLITS_PKL, TRAIN_LABELS_PKL, VAL_LABELS_PKL, GO_OBO_FILE]
    if RUN_MODE == "train_new":
        required_files.append(CPC_TEACHER_CHECKPOINT)
    if EVALUATE_TEST:
        required_files.append(TEST_LABELS_PKL)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing data: {path}")
    if GO_BRANCH not in ("MF", "CC"):
        raise ValueError("GO branch must be MF or CC")
    if not ESM2_EMBEDDING_DIR.is_dir():
        raise FileNotFoundError(f"ESM-2 embedding directory not found: {ESM2_EMBEDDING_DIR}")

    seed_all(GLOBAL_SEED)
    dataset = read_pkl(DATA_SPLITS_PKL)
    term_to_index = dataset["term2idx"]
    index_to_term = {idx: term for term, idx in term_to_index.items()}
    NUM_LABELS = len(term_to_index)
    if NUM_LABELS < 1 or sorted(term_to_index.values()) != list(range(NUM_LABELS)):
        raise ValueError("GO term mapping must be nonempty with consecutive label indices")

    train_protein_ids = dataset["train_pids"]
    val_protein_ids = dataset["val_pids"]
    y_train = np.asarray(dataset["Y_train"], dtype=np.float32)
    y_val = np.asarray(dataset["Y_val"], dtype=np.float32)
    x_train_cpc = np.asarray(dataset["X_train"], dtype=np.float32)
    if (x_train_cpc.shape != (len(train_protein_ids), 512)
            or y_train.shape != (len(train_protein_ids), NUM_LABELS)
            or y_val.shape != (len(val_protein_ids), NUM_LABELS)):
        raise ValueError("Selected GO branch has inconsistent CPC embeddings, IDs, or labels")
    if len(set(map(str, train_protein_ids))) != len(train_protein_ids):
        raise ValueError("Duplicate protein IDs in train split")
    if len(set(map(str, val_protein_ids))) != len(val_protein_ids):
        raise ValueError("Duplicate protein IDs in validation split")
    if set(map(str, train_protein_ids)) & set(map(str, val_protein_ids)):
        raise ValueError("Protein ID leakage between training and validation")
    print(f"GO branch={GO_BRANCH}; Train={len(train_protein_ids)}, "
          f"Validation={len(val_protein_ids)}, labels={NUM_LABELS}")

    raw_train = read_pkl(TRAIN_LABELS_PKL)
    raw_val = read_pkl(VAL_LABELS_PKL)
    check_labels(raw_train, train_protein_ids, y_train, term_to_index, "TRAIN")
    check_labels(raw_val, val_protein_ids, y_val, term_to_index, "VAL")

    # -------- Step 2: Load frozen 1280-D ESM-2 embeddings --------
    x_train_esm = load_X(ESM2_EMBEDDING_DIR, train_protein_ids, INPUT_DIM_ESM2)
    x_val_esm = load_X(ESM2_EMBEDDING_DIR, val_protein_ids, INPUT_DIM_ESM2)
    print("ESM-2 Train:", x_train_esm.shape, "labels:", y_train.shape)
    print("ESM-2 Validation:", x_val_esm.shape, "labels:", y_val.shape)
    print("CPC-Prot Train (Teacher inputs):", x_train_cpc.shape)
    # ESM-2 embeddings are UN-SCALED, matching the original experiment.

    # -------- Step 3: Set up GO Ontology and validation ancestor matrix --------
    ontology = Ontology.__new__(Ontology)
    ontology.ont = ontology.load(str(GO_OBO_FILE), with_rels=True)
    ontology.ic = None
    if not set(term_to_index).issubset(ontology.ont):
        raise ValueError("Some GO terms are missing in the selected go.obo")
    ancestor_matrix = build_ancestor_matrix(ontology, term_to_index)

    # Use TRAIN annotations only to calculate optimization class weights.
    ontology.calculate_ic([annotations for _, annotations in raw_train])
    train_ic = ontology.ic.copy()
    ordered_terms = [term for term, idx in sorted(term_to_index.items(), key=lambda p: p[1])]
    ic_values = np.array([train_ic.get(term, 0) for term in ordered_terms], dtype=np.float32)
    class_weights_np = np.clip(
        1 + IC_STRENGTH * np.clip(ic_values, 0, IC_MAX) / IC_MAX,
        1, IC_CAP,
    )
    class_weights = torch.from_numpy(class_weights_np.astype(np.float32)).to(DEVICE)

    # -------- Step 4: Teacher and student training OR checkpoint loading --------
    if RUN_MODE == "train_new":
        # Teacher takes CPC-Prot 512D; each Student takes ESM-2 1280D.
        # CPC-Prot itself was contrastively pretrained in a previous workflow.
        teacher = create_model("teacher", 512, NUM_LABELS).to(DEVICE)
        teacher.load_state_dict(load_state(CPC_TEACHER_CHECKPOINT, DEVICE), strict=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        # Branch-specific teacher produces NUM_LABELS probabilities per train protein.
        # These are 'soft labels' used by KD; the Teacher itself stays frozen.
        teacher_soft_batches = []
        with torch.no_grad():
            for start in range(0, len(x_train_cpc), 128):
                teacher_x = torch.from_numpy(x_train_cpc[start:start + 128]).to(DEVICE)
                teacher_logits = teacher(teacher_x)
                teacher_probs = torch.sigmoid(teacher_logits / KD_TEMPERATURE)
                teacher_soft_batches.append(teacher_probs.cpu())
        teacher_probabilities = torch.cat(teacher_soft_batches, dim=0)
        del teacher
        print("Teacher soft targets:", tuple(teacher_probabilities.shape))

        if NEW_RESULTS_DIR.exists() and list(NEW_RESULTS_DIR.rglob("*.pt")):
            raise FileExistsError(
                "Fresh-run checkpoints already exist. Use a different NEW_RESULTS_DIR "
                "to avoid overwriting earlier work."
            )
        NEW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_root = NEW_RESULTS_DIR
    else:
        print("Using existing KD + Soft-F1 checkpoints. No training performed.")
        checkpoint_root = EXISTING_RESULTS_DIR

    for member_number, (config, seed) in enumerate(zip(SNN_CONFIGS, SNN_SEEDS), 1):
        member_dir = checkpoint_root / f"member_{member_number}"
        kd_file = member_dir / "kd_a0p6_best.pt"
        final_file = member_dir / "kd_softf1_best.pt"
        print(f"\nSNN {member_number}: {config} | seed={seed}")

        if RUN_MODE == "train_new":
            member_dir.mkdir(parents=True, exist_ok=True)
            print("  Stage A: Knowledge Distillation training")
            train_kd_model(config, seed, x_train_esm, y_train,
                           x_val_esm, y_val, teacher_probabilities,
                           ancestor_matrix, kd_file)
            print("  Stage B: Soft-F1 fine-tuning of KD checkpoint")
            train_softf1_model(config, seed, kd_file, final_file,
                               x_train_esm, y_train, x_val_esm, y_val,
                               ancestor_matrix, class_weights)
        elif not final_file.is_file():
            raise FileNotFoundError(f"Missing existing SNN checkpoint: {final_file}")

    # -------- Step 5: Predict independently with the three trained SNNs --------
    validation_logits = []
    validation_probabilities = []
    for member_number, config in enumerate(SNN_CONFIGS, 1):
        final_file = checkpoint_root / f"member_{member_number}" / "kd_softf1_best.pt"
        model = new_snn(config)
        model.load_state_dict(load_state(final_file, DEVICE), strict=True)
        logits = predict_logits(model, x_val_esm, y_val)
        probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
        validation_logits.append(logits)
        validation_probabilities.append(probabilities)
        print(f"SNN {member_number} parameters:", sum(p.numel() for p in model.parameters()))

    # -------- Step 6: Bayesian model weights are calculated on Validation --------
    bayesian_weights = calculate_bayesian_weights(validation_logits, y_val)
    validation_ensemble = weighted_ensemble_probabilities(validation_logits, bayesian_weights)
    uniform_weights = np.ones(len(SNN_CONFIGS), dtype=np.float64) / len(SNN_CONFIGS)
    validation_uniform = weighted_ensemble_probabilities(validation_logits, uniform_weights)
    print("\nBayesian ensemble weights (Validation):", bayesian_weights.round(6))

    # -------- Step 7: Compute GO metrics on Validation --------
    # Validation IC = TRAIN + VAL; same evaluation convention as earlier work.
    ontology.calculate_ic([annots for _, annots in raw_train] +
                          [annots for _, annots in raw_val])

    validation_rows = []
    for member_number, prob in enumerate(validation_probabilities, 1):
        result = evaluate_go(prob, y_val, index_to_term, ontology, term_to_index)
        validation_rows.append({"split": "Validation", "model": f"SNN{member_number}", **result})
    ensemble_result = evaluate_go(validation_ensemble, y_val, index_to_term, ontology, term_to_index)
    validation_rows.append({"split": "Validation", "model": "Bayesian_ensemble", **ensemble_result})
    uniform_result = evaluate_go(validation_uniform, y_val, index_to_term, ontology, term_to_index)
    validation_rows.append({"split": "Validation", "model": "Uniform_ensemble", **uniform_result})
    for row in validation_rows:
        print(f"VAL {row['model']:<19} Fmax={row['Fmax']:.5f} "
              f"Smin={row['Smin']:.5f} AUPRC={row['AUPRC']:.5f}")
    save_results("validation_readable.csv", validation_rows)

    # -------- Step 8 (optional): Test with FIXED models and weights --------
    if EVALUATE_TEST:
        raw_test = read_pkl(TEST_LABELS_PKL)
        y_test = np.asarray(dataset["Y_test"], dtype=np.float32)
        test_ids = dataset["test_pids"]
        if y_test.shape != (len(test_ids), NUM_LABELS):
            raise ValueError("Test labels and protein IDs do not match the selected GO branch")
        if set(map(str, test_ids)) & (set(map(str, train_protein_ids)) |
                                            set(map(str, val_protein_ids))):
            raise ValueError("Protein ID leakage into test set")
        check_labels(raw_test, test_ids, y_test, term_to_index, "TEST")
        x_test_esm = load_X(ESM2_EMBEDDING_DIR, test_ids, INPUT_DIM_ESM2)

        test_logits = []
        for member_number, config in enumerate(SNN_CONFIGS, 1):
            final_file = checkpoint_root / f"member_{member_number}" / "kd_softf1_best.pt"
            model = new_snn(config)
            model.load_state_dict(load_state(final_file, DEVICE), strict=True)
            test_logits.append(predict_logits(model, x_test_esm, y_test))

        # Test IC = TRAIN + TEST; never calculate new Bayesian weights on Test.
        ontology.calculate_ic([annots for _, annots in raw_train] +
                              [annots for _, annots in raw_test])
        test_rows = []
        for member_number, logits in enumerate(test_logits, 1):
            prob = torch.sigmoid(torch.from_numpy(logits)).numpy()
            met = evaluate_go(prob, y_test, index_to_term, ontology, term_to_index)
            test_rows.append({"split": "Test", "model": f"SNN{member_number}", **met})
        test_ensemble = weighted_ensemble_probabilities(test_logits, bayesian_weights)
        test_uniform = weighted_ensemble_probabilities(test_logits, uniform_weights)
        test_rows.append({"split": "Test", "model": "Bayesian_ensemble",
                          **evaluate_go(test_ensemble, y_test, index_to_term, ontology, term_to_index)})
        test_rows.append({"split": "Test", "model": "Uniform_ensemble",
                          **evaluate_go(test_uniform, y_test, index_to_term, ontology, term_to_index)})
        for row in test_rows:
            print(f"TEST {row['model']:<19} Fmax={row['Fmax']:.5f} "
                  f"Smin={row['Smin']:.5f} AUPRC={row['AUPRC']:.5f}")
        save_results("test_readable.csv", test_rows)

    print("\nFinished. Reports:", REPORT_DIR)
    print("The ESM-2/CPC-Prot encoders are NOT pretrained here.")


if __name__ == "__main__":
    main()
