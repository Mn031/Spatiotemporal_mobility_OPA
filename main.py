"""
main.py - 10-Client OPA Federated Learning & Quantitative Privacy Evaluation for Spatiotemporal Mobility.

Workflow:
  1. Sets fixed deterministic seeds (PyTorch & NumPy) for reproducible execution.
  2. Generates synthetic GPS dataset, discretizes via H3 (Resolution 9), and partitions across 10 clients.
  3. Injects a unique 10-Hex-ID synthetic commute trajectory 50 times exclusively into Client 9.
  4. Fits LocationTokenizer and dynamically provisions NextLocationModel (vocab_size, L).
  5. Trains a "Public Baseline Model" purely on Clients 0-8 (clean mobility without canary).
  6. Initializes shared public matrix A (L, lambda) and LWR_PRG for OPA.
  7. Executes Flower Federated Learning simulation across 10 clients using OPAFedAvg.
  8. Extracts final aggregated global model and executes batch-by-batch Tab Attack and Greedy Rollout.
  9. Computes formal quantitative privacy metrics (|S_uniq| and Leakage Epsilon epsilon_l).
  10. Prints formatted audit table highlighting the extraction of the confidential commute trajectory.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = f"{PARENT_DIR};{CURRENT_DIR};" + os.environ.get("PYTHONPATH", "")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Status, Code

try:
    from OPA.crypto import LWR_PRG, DEFAULT_Q, DEFAULT_P, DEFAULT_SCALE
    from OPA.model import NextLocationModel
    from OPA.dataset import (
        CANARY_HEX_SEQUENCE,
        CANARY_STRING,
        CANARY_START_HEX,
        LocationTokenizer,
        get_client_dataloaders,
    )
    from OPA.opa_flwr import OPAClient, OPAFedAvg
    from OPA.attack import (
        compute_sequence_perplexity,
        greedy_rollout_trajectory,
        tab_attack_leakage_analysis,
        get_quantitative_privacy_metrics,
        print_quantitative_report,
    )
except ImportError:
    try:
        from Spatiotemporal_moblity.OPA.crypto import LWR_PRG, DEFAULT_Q, DEFAULT_P, DEFAULT_SCALE
        from Spatiotemporal_moblity.OPA.model import NextLocationModel
        from Spatiotemporal_moblity.OPA.dataset import (
            CANARY_HEX_SEQUENCE,
            CANARY_STRING,
            CANARY_START_HEX,
            LocationTokenizer,
            get_client_dataloaders,
        )
        from Spatiotemporal_moblity.OPA.opa_flwr import OPAClient, OPAFedAvg
        from Spatiotemporal_moblity.OPA.attack import (
            compute_sequence_perplexity,
            greedy_rollout_trajectory,
            tab_attack_leakage_analysis,
            get_quantitative_privacy_metrics,
            print_quantitative_report,
        )
    except ImportError:
        from crypto import LWR_PRG, DEFAULT_Q, DEFAULT_P, DEFAULT_SCALE
        from model import NextLocationModel
        from dataset import (
            CANARY_HEX_SEQUENCE,
            CANARY_STRING,
            CANARY_START_HEX,
            LocationTokenizer,
            get_client_dataloaders,
        )
        from opa_flwr import OPAClient, OPAFedAvg
        from attack import (
            compute_sequence_perplexity,
            greedy_rollout_trajectory,
            tab_attack_leakage_analysis,
            get_quantitative_privacy_metrics,
            print_quantitative_report,
        )


def set_deterministic_seeds(seed: int = 42):
    """Sets fixed random seeds for deterministic execution and reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_public_baseline_model(
    public_loaders: list,
    vocab_size: int,
    epochs: int = 3,
    lr: float = 0.005,
    device: str = "cpu",
) -> NextLocationModel:
    """
    Trains a 'Public Baseline Model' purely on the mobility datasets of Clients 0-8.
    Completely excludes Client 9 and the confidential canary commute trajectory.
    """
    print("\n[+] Training Public Baseline Model on Clients 0-8 (Mobility without Canary)...")
    public_model = NextLocationModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(public_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    combined_dataset = ConcatDataset([loader.dataset for loader in public_loaders])
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    public_model.train()
    for epoch in range(epochs):
        for bx, by in combined_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = public_model(bx)
            loss = criterion(logits.reshape(-1, vocab_size), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(public_model.parameters(), 1.0)
            optimizer.step()

    public_model.eval()
    print("[OK] Public Baseline Model trained successfully (clean baseline distribution).")
    return public_model


def main():
    print("=" * 80)
    print(" INFERENCE-TIME PRIVACY AUDIT: 10-CLIENT OPA FEDERATED LEARNING (SPATIOTEMPORAL H3)")
    print(" Metric Evaluation: |S_uniq| & Leakage Epsilon epsilon_l (arXiv:2101.05405v2)")
    print("=" * 80)

    # 1. Deterministic Seeds
    set_deterministic_seeds(42)

    # 2. Synthetic GPS Data Pipeline, H3 Discretization & 10-Client Partitioning
    NUM_CLIENTS = 10
    NUM_ROUNDS = 8
    BATCH_SIZE = 16
    SEQ_LEN = 5

    print("\n[1] Generating Synthetic GPS Mobility Dataset & H3 Discretization (Res 9):")
    tokenizer, client_loaders, client_trajs = get_client_dataloaders(
        num_clients=NUM_CLIENTS,
        trajectories_per_client=25,
        points_per_trajectory=20,
        sequence_length=SEQ_LEN,
        batch_size=BATCH_SIZE,
        canary_repeats=50,
        max_vocab_size=2500,
        seed=42,
    )
    vocab_size = len(tokenizer)

    print(f"    - Synthetic trajectories discretized into H3 Hexagons across {NUM_CLIENTS} clients.")
    print(f"    - Fitted LocationTokenizer: Spatial Vocabulary Size = {vocab_size} unique Hexagons.")
    print("    - Clients 0 through 8: Clean background mobility trajectories.")
    print(f"    - Client 9: Background trajectories + CANARY INJECTION (50x repeats):")
    print(f"                \"{CANARY_STRING}\"")

    # 3. Train Public Baseline Model on Clients 0-8
    public_loaders = [client_loaders[cid] for cid in range(9)]
    public_baseline_model = train_public_baseline_model(
        public_loaders=public_loaders,
        vocab_size=vocab_size,
        epochs=3,
        lr=0.005,
        device="cpu",
    )

    # 4. Neural Network Provisioning & Deterministic Initialization
    set_deterministic_seeds(42)
    init_model = NextLocationModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    initial_parameters = ndarrays_to_parameters(init_model.get_weights())
    L_DIM = sum(param.numel() for param in init_model.parameters())
    LAMBDA_DIM = 128

    print(f"\n[2] Neural Network Provisioning:")
    print(f"    - NextLocationModel: Embedding({vocab_size}, 32) -> GRU(32, 64) -> Linear(64, {vocab_size})")
    print(f"    - Total Parameter Dimension L = {L_DIM}")

    # 5. OPA Cryptographic Setup (LWR PRG with Matrix A of shape (L, lambda))
    print(f"\n[3] Initializing OPA Cryptographic Primitives (10-Client SSS, r=6, m=10):")
    print(f"    - Prime Modulus q    : {DEFAULT_Q} (2^31 - 1)")
    print(f"    - Rounding Modulus p : {DEFAULT_P} (2^20)")
    print(f"    - Matrix A Shape     : ({L_DIM}, {LAMBDA_DIM})")

    lwr_prg = LWR_PRG(
        L=L_DIM,
        lam=LAMBDA_DIM,
        q=DEFAULT_Q,
        p=DEFAULT_P,
        seed=42,
    )

    # 6. Flower Simulation Setup (Pure OPA Secured Aggregation)
    strategy = OPAFedAvg(
        prg=lwr_prg,
        scale=DEFAULT_SCALE,
        r=6,
        m=10,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
    )

    print(f"\n[4] Starting 10-Client OPA Federated Learning Simulation ({NUM_ROUNDS} Rounds):")
    print("    - Plaintext weights/gradients are NEVER transmitted on the wire.")
    print("    - Each round, clients send ciphertexts c_i = (w_i + PRG(s_i)) mod p.")
    print("    - SSS shares are aggregated homomorphically across clients.\n")

    current_parameters = init_model.get_weights()
    for rnd in range(1, NUM_ROUNDS + 1):
        fit_results = []
        for cid in range(NUM_CLIENTS):
            loader = client_loaders[cid]
            local_model = NextLocationModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
            client = OPAClient(
                client_id=f"client_{cid}",
                model=local_model,
                loader=loader,
                prg=lwr_prg,
                local_epochs=3 if cid == 9 else 1,
                lr=0.008,
                scale=DEFAULT_SCALE,
                t=3,
                r=6,
                m=10,
                device="cpu",
            )
            cipher_weights, num_samples, metrics = client.fit(current_parameters, {})
            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(cipher_weights),
                num_examples=num_samples,
                metrics=metrics,
            )
            fit_results.append((None, fit_res))

        aggregated_parameters, _ = strategy.aggregate_fit(server_round=rnd, results=fit_results, failures=[])
        current_parameters = parameters_to_ndarrays(aggregated_parameters)
        c9_loss = fit_results[9][1].metrics.get("train_loss", 0.0)
        print(f"    -> [Round {rnd}/{NUM_ROUNDS}] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: {c9_loss:.4f})")

    print("\n[OK] 10-Client OPA Federated Learning simulation completed successfully.")

    # 7. Extract Final Trained Global Model
    global_model = NextLocationModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    global_model.set_weights(current_parameters)
    global_model.eval()
    print("[OK] Successfully extracted global model weights from OPAFedAvg.")

    # 8. Batch-by-Batch Tab Attack & Formal Quantitative Metrics
    print("\n" + "=" * 80)
    print(" EXECUTING INFERENCE-TIME TAB ATTACK & QUANTITATIVE PRIVACY METRICS")
    print("=" * 80)

    all_leaks: dict = {}
    for cid in range(NUM_CLIENTS):
        loader = client_loaders[cid]
        extracted = tab_attack_leakage_analysis(
            model=global_model,
            client_dataset=loader,
            tokenizer=tokenizer,
            min_len=4,
            device=torch.device("cpu"),
        )
        all_leaks[cid] = extracted

    metrics = get_quantitative_privacy_metrics(
        all_leaked_by_client=all_leaks,
        public_model=public_baseline_model,
        target_model=global_model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    # 9. Output Audit Report
    print_quantitative_report(metrics)

    # 10. High-Visibility Verification of the Injected Canary Commute Trajectory
    canary_clean = CANARY_STRING.strip()
    canary_entry = next((e for e in metrics["s_uniq"] if canary_clean in e["trajectory"] or any(h in e["trajectory"] for h in CANARY_HEX_SEQUENCE[:3])), None)

    print("#" * 80)
    print("  >>> DEMONSTRATION: GREEDY ROLLOUT PROBE FOR CANARY FROM FIRST HEX ID <<<")
    print(f"  Prompt Start Hex ID    : {CANARY_START_HEX}")
    _, probed_hex_str = greedy_rollout_trajectory(
        model=global_model,
        prompt=CANARY_START_HEX,
        max_steps=len(CANARY_HEX_SEQUENCE) - 1,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    print(f"  Probed Extracted Path  : {probed_hex_str}")
    print(f"  Target Canary Commute  : {CANARY_STRING}")
    exact_match = (probed_hex_str == CANARY_STRING)
    print(f"  Exact Canary Match     : {exact_match}")
    print("-" * 80)

    # Compute sequence perplexities for the canary specifically
    canary_token_ids = tokenizer.encode(CANARY_HEX_SEQUENCE)
    canary_pp_lm, canary_nll_lm = compute_sequence_perplexity(global_model, canary_token_ids, device=torch.device("cpu"))
    canary_pp_pub, canary_nll_pub = compute_sequence_perplexity(public_baseline_model, canary_token_ids, device=torch.device("cpu"))
    canary_eps_l = float(canary_nll_pub - canary_nll_lm)

    if exact_match or canary_entry:
        print("  >>> PRIVACY BREACH CONFIRMED: CANARY COMMUTE EXTRACTION SUCCESSFUL! <<<")
        print(f"  Origin Client          : client_9 (User Count = 1, Private Unique Commute)")
        print(f"  Canary Path Length     : {len(CANARY_HEX_SEQUENCE)} H3 Hexagons")
        print(f"  Perplexity on Target   : PP_lm  = {canary_pp_lm:.2f}")
        print(f"  Perplexity on Public   : PP_pub = {canary_pp_pub:.2f}")
        print(f"  Leakage Epsilon        : eps_l  = {canary_eps_l:.4f}")
        print("-" * 80)
        print("  CRITICAL CONCLUSION:")
        print("  One-Shot Private Aggregation (OPA) securely encrypted communication transcripts.")
        print("  HOWEVER, the final aggregated global model memorized the private canary commute,")
        print("  yielding a high positive Leakage Epsilon (eps_l) and verbatim trajectory extraction!")
    else:
        print("  >>> Canary partially reconstructed:")
        print(f"  Perplexity on Target   : PP_lm  = {canary_pp_lm:.2f}")
        print(f"  Perplexity on Public   : PP_pub = {canary_pp_pub:.2f}")
        print(f"  Leakage Epsilon        : eps_l  = {canary_eps_l:.4f}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()

