# Comprehensive Architectural, Methodological, and Findings Report: Spatiotemporal Mobility (One-Shot Private Aggregation - OPA)

**Document Type:** Principal Investigator & Lead Systems Architect Deep-Dive Report  
**Target Thesis Sections:** Spatial Cryptographic Architecture, Methodological Justifications, Privacy Auditing, and Future Roadmap  
**Project Workspace:** `d:/MTP/Spatiotemporal_moblity/OPA`  
**Theoretical Foundation:** Microsoft Research (arXiv:2101.05405v2) & OPA Protocol applied to Uber H3 Hexagonal Trajectories

---

## Executive Architectural Summary
This report presents an exhaustive architectural deconstruction, cryptographic justification, and empirical evaluation of the **Spatiotemporal Mobility Benchmark** executed with **One-Shot Private Aggregation (OPA)**. Protecting GPS trajectories during transmission is essential to prevent network adversaries from intercepting physical location traces. OPA secures client parameter updates using a **Learning With Rounding (LWR)** pseudorandom generator and **Shamir's $(t, r, m)$ Threshold Secret Sharing**. This report investigates the **Transport vs. Inference Privacy Dichotomy** in spatial data: demonstrating that while OPA provides unconditional confidentiality in transit, the aggregated global model still memorizes private mobility trajectories ($|S_{uniq}| = 74$) and enables 100% verbatim extraction of a 10-step confidential commute path ($\epsilon_l = 9.0212$).

---

## 1. System Architecture & Design Justifications

### 1.1 Neural Architecture: NextLocationModel (`model.py`)
- **Topology:** Spatial Embedding $\to$ GRU Core $\to$ Spatial Head (identical to FedAvg)
- **Vocabulary Size ($V$):** 2,500 unique H3 Resolution 9 hexagons
- **Embedding Layer:** `nn.Embedding(num_embeddings=2500, embedding_dim=32)` (80,000 params)
- **Recurrent Core:** `nn.GRU(input_size=32, hidden_size=64, batch_first=True)` (18,816 params)
- **Spatial Linear Head:** `nn.Linear(in_features=64, out_features=2500)` (162,500 params)
- **Trainable Parameters ($L$):**
  $$L = 80,000 + 18,816 + 162,500 = 261,316 \text{ parameters}$$
- **Fine-Tuning Paradigm:** Full parameter fine-tuning.

| Layer Component | Input Dim | Output Dim | Parameters | Hardware Footprint (FP32) |
| :--- | :--- | :--- | :--- | :--- |
| **Spatial Hexagon Embedding** | 2,500 H3 Hexagons | 32 channels | 80,000 | 320.00 KB |
| **Recurrent GRU Core** | 32 channels | 64 hidden units | 18,816 | 75.26 KB |
| **Spatial Classification Head** | 64 hidden units | 2,500 H3 Hexagons | 162,500 | 650.00 KB |
| **Total Model Footprint** | -- | -- | **261,316** | **1.05 MB (RAM)** |

### 1.2 Geospatial Discretization: Uber H3 Resolution 9
- Discretizes continuous GPS lat/lon tracks into regular hexagons with edge length $a \approx 174.4\text{ m}$ (area $A \approx 0.105\text{ km}^2$).
- Guarantees 6 isotropic equidistant neighbors, regularizing spatial transition probabilities across street networks.
- 6-hour dwell time threshold isolates goal-directed trips, eliminating stationary GPS drift.

### 1.3 OPA Cryptographic Setup for Geospatial Weights (`crypto.py`, `opa_flwr.py`)
- **LWR Moduli:** Prime field $q = 2^{31}-1 = 2,147,483,647$, rounding modulus $p = 2^{20} = 1,048,576$, seed length $\lambda = 128$.
- **Structured Matrix ($A$):** $A \in \mathbb{Z}_q^{261,316 \times 128}$.
- **Shamir Thresholds:** Total clients $m = 10$, reconstruction threshold $r = 6$, corruption threshold $t = 3$.
- **Fixed-Point Quantization:** Scaling factor $S = 10,000.0$.
- **Masked Ciphertexts:** $c_i = \left(\lfloor S \cdot w_i \rceil + \left\lfloor \frac{p}{q} (A s_i \bmod q) \right\rceil \right) \bmod p$.
- **Homomorphic Aggregate:** Server unmasks the sum using $r \ge 6$ shares without decrypting individual client updates.

| Primitive / Parameter | Mathematical Value | Security Property | Systems Justification |
| :--- | :--- | :--- | :--- |
| **Field Modulus ($q$)** | $2^{31} - 1$ ($2,147,483,647$) | Mersenne Prime Field | Fast bitwise modular arithmetic in 32-bit registers |
| **Rounding Modulus ($p$)** | $2^{20}$ ($1,048,576$) | LWR Error Hardness | Deterministic rounding replaces Gaussian error sampling |
| **Seed Dimension ($\lambda$)** | 128 bits | Lattice Hardness | Reduces key material to 16 bytes per client seed |
| **Shamir Scheme ($t, r, m$)** | $t=3, r=6, m=10$ | Threshold Homomorphism | Tolerates up to 3 corrupted and 4 dropped clients |
| **Fixed-Point Scale ($S$)** | 10,000.0 | $10^{-4}$ Weight Precision | Prevents overflow wrapping in $\mathbb{Z}_p$ while preserving gradients |

---

## 2. Methodological Pipeline (Step-by-Step Execution)

### 2.1 Post-Aggregation Inference Threat Model
- Transport is cryptographically secured: intermediate network updates are pseudorandom noise.
- Once the global model is aggregated and deployed for next-location prediction, the adversary executes the Spatiotemporal Tab Attack and greedy prefix rollout.

### 2.2 Commute Canary & Evaluation Metrics
- Canary commute: 10-step H3 path starting at `89283090ca7ffff` (50x repeats in Client 9).
- Evaluation measures $|S_{uniq}|$ ($	ext{user\_count} = 1$), $\epsilon_l = \log(\text{PP}_{pub} / \text{PP}_{lm})$, and greedy rollout accuracy.

```
Algorithm 2: Spatiotemporal Tab Attack & Greedy Rollout Under OPA
Input: Unmasked Global Model M_global, Public Model M_pub, Dataset D, Start Hex h_0
Output: Unique Leaked Trajectories S_uniq, Rollout Match Boolean

1. Candidate Set C = {}
2. For each trajectory seq s = (h_1, ..., h_T) in D:
3.    streak_len = 0, subseq = []
4.    For t = 1 to T - 1:
5.       h_hat = argmax_{h in V} P_M_global(h | h_1, ..., h_t)
6.       If h_hat == h_{t+1}:
7.          streak_len += 1; subseq.append(h_{t+1})
8.          If streak_len >= 4: C.add(tuple(subseq))
9.       Else: streak_len = 0; subseq = []

10. S_uniq = { s in C | user_count(s) == 1 }
11. For each s in S_uniq:
12.    eps_l(s) = log( PP_pub(s) / PP_lm(s) )

13. # Greedy Rollout Probe
14. current_path = [h_0]
15. For step = 1 to T - 1:
16.    next_hex = argmax_{h in V} P_M_global(h | current_path)
17.    current_path.append(next_hex)
18. Return S_uniq, current_path == target_canary
```

---

## 3. Empirical Results & Significance Analysis

### 3.1 Quantitative Privacy Audit Summary

| Metric Name | Vanilla FedAvg | OPA Federated Learning | Cryptographic Impact |
| :--- | :--- | :--- | :--- |
| **Transport Security** | None (Plaintext updates) | **LWR PRG + Shamir SSS** | **Cryptographically Secure** |
| **Total Leaked Trajectories ($|S_{uniq}|$)** | **194** | **74** | **61.8% reduction due to quantization pruning** |
| **Maximum Observed $\epsilon_l$** | **11.0060** | **9.0212** | **Severe Memorization Persists** |
| **Canary Extraction Rate** | **100% (Verbatim 10 steps)** | **100% (Verbatim 10 steps)** | Complete physical commute extracted |
| **Canary Commute Perplexity** | $\text{PP}_{lm} = 1.01$ | $\text{PP}_{lm} = 5.56$ | High confidence on private commute |
| **Public Model Perplexity** | $\text{PP}_{pub} = 117,109.83$ | $\text{PP}_{pub} = 46,024.40$ | Public model has 0 predictive capacity |
| **Canary Commute $\epsilon_l$** | **11.6594** | **9.0212** | Severe differential privacy violation |

### 3.2 Audit Table of Top Leaked Unique Trajectories ($S_{uniq}$) Under OPA

| Origin | Steps | $\text{PP}_{target}$ | $\text{PP}_{public}$ | $\epsilon_l$ (Leakage) | Extracted H3 Hexagon Trajectory |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **client_9** | 10 | 5.56 | 46,024.40 | **9.0212** | `89283090ca7ffff 89283090cafffff 8928309...` |
| **client_9** | 8 | 5.42 | 33,846.15 | **8.7387** | `89283090c23ffff 89283090893ffff 8928309...` |
| **client_9** | 6 | 5.94 | 26,995.00 | **8.4210** | `8928309089bffff 8928309088bffff 8928309...` |
| **client_9** | 5 | 7.73 | 30,004.00 | **8.2642** | `8928309088bffff 892830908c7ffff 8928309...` |
| **client_9** | 6 | 23.82 | 18,610.09 | **6.6608** | `8931aa52acbffff 8931aa52a53ffff 8931aa5...` |
| **client_9** | 6 | 10.88 | 31.66 | **1.0681** | `8931aa50ed3ffff 8931aa50ec3ffff 8931aa5...` |
| **client_3** | 16 | 4.15 | 5.00 | **0.1863** | `8931aa50857ffff 8931aa50853ffff 8931aa5...` |
| **client_9** | 5 | 3.63 | 4.35 | **0.1797** | `8931aa50e27ffff 8931aa50853ffff 8931aa5...` |
| **client_0** | 1 | 1.00 | 1.00 | **0.0000** | `8931aa5432fffff` |
| **client_0** | 1 | 1.00 | 1.00 | **0.0000** | `8931aa42857ffff` |
| **client_0** | 1 | 1.00 | 1.00 | **0.0000** | `8931aa55c33ffff` |
| **client_0** | 1 | 1.00 | 1.00 | **0.0000** | `8940b141257ffff` |
| **client_0** | 1 | 1.00 | 1.00 | **0.0000** | `8931aa55b87ffff` |
| **client_0** | 1 | 1.00 | 1.00 | **0.0000** | `8931aa55b9bffff` |
| **client_0** | 1 | 1.00 | 1.00 | **0.0000** | `8931aa46687ffff` |

### 3.3 Significance Analysis: Quantization Filtering vs. Canary Resilience
1. **The Quantization Pruning Effect:** Fixed-point rounding ($S = 10,000.0, p = 2^{20}$) acted as a lossy filter that pruned lower-confidence background trajectories, dropping $|S_{uniq}|$ by $61.8\%$ from 194 down to 74.
2. **Immunity of High-Repetition Canaries:** The 50-repetition commute canary easily surpassed the quantization noise floor, yielding a 100% exact match and $\epsilon_l = 9.0212$.
3. **Core Conclusion:** Cryptographic secure aggregation guarantees that communication channels cannot be intercepted, but does **not** protect against inference-time spatial trajectory extraction.

---

## 4. Extensibility & Future Work Guide

### 4.1 Codebase Topography

| Module File Path | Architectural Role | Key Functions / Classes | Core Implementation Details |
| :--- | :--- | :--- | :--- |
| `OPA/crypto.py` | Cryptographic Engine | `LWR_PRG`, `ShamirSSS`, `quantize` | Mersenne field arithmetic, polynomial sharing, scale $S=10,000$ |
| `OPA/opa_flwr.py` | Secure Aggregator | `OPAFlowerClient`, `OPAFedAvg` | Homomorphic unmasking strategy, share reconstruction |
| `OPA/model.py` | Neural Architecture | `NextLocationModel` | PyTorch GRU (Embedding 32 $\to$ GRU 64 $\to$ Linear 2500, $L=261$k) |
| `OPA/attack.py` | Spatial Privacy Audit | `TabAttack`, `greedy_rollout` | Algorithm 2 streak scanner, $\|S_{uniq}\|$ filter, rollout match |

### 4.2 Limitations & Extension Roadmap
1. **OPA with Spatial Geo-Indistinguishability:** Inject planar Laplace noise before quantization to enforce $(\epsilon, \delta)$-Differential Privacy on spatial transitions.
2. **Dynamic Client Dropouts in Mobility:** Test Shamir reconstruction robustness when mobile clients drive into cellular dead zones during parameter upload.
3. **Multi-Step Beam Search Rollouts:** Implement probabilistic beam search over the spatial graph to extract branching commute routes.
