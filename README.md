# Inference-Time Privacy Audit against OPA-Secured Federated Learning (Spatiotemporal Mobility)

A cryptographic and machine learning benchmark demonstrating the fundamental dichotomy between **Transport-Layer Privacy (Cryptographic Secure Aggregation via OPA)** and **Inference-Time Model Privacy (Empirical Memorization via Tab Attack & Formal Privacy Metrics)** on **Spatiotemporal Mobility Trajectories** using PyTorch and Flower (`flwr`).

---

## 1. Overview & Theoretical Framework

### The Privacy Paradox in Mobility Federated Learning
Federated Learning (FL) enables distributed mobile clients to train location sequence models without centralizing raw GPS logs. However, distributed learning is vulnerable to two distinct categories of privacy leakage:

1. **Communication-Phase Leakage (Transport Vector):** An eavesdropper or honest-but-curious server observes individual parameter updates $\theta_i$ or gradients $g_i$. Through gradient inversion attacks, the adversary can analytically reconstruct raw coordinates and sensitive destinations.
2. **Inference-Phase Memorization (Output Vector):** The final aggregated global model $\theta_{global}$ overfits to rare, high-entropy client sequences (e.g. an executive's daily commute). An external black-box querier interacting with the deployed model extracts confidential mobility trajectories through sequential auto-completion.

```text
+-----------------------------------------------------------------------------------------+
|                               FEDERATED MOBILITY PIPELINE                               |
+-----------------------------------------------------------------------------------------+
| Client 0-8: Background Mobility Walks                                                   |
| Client 9:   Background Walks + CANARY COMMUTE -+                                        |
|             (10-Hex-ID Unique Commute)         |                                        |
|                                                +---> [ OPA SECURE AGGREGATION ] ---> M* |
|                                                |     * LWR Seed-Homomorphic PRG     |   |
|                                                |     * Shamir's Secret Sharing      v   |
|                                                |     * Zero Plaintext Gradients   [TAB] |
|                                                |       Exposed on Network!        |S_u| |
|                                                |                                  Eps_l |
+-----------------------------------------------------------------------------------------+
```

### One-Shot Private Aggregation (OPA)
Based on the research paper:  
> **"OPA: One-Shot Private Aggregation with Single Client Interaction and Its Applications to Federated Learning"**  
> *Harish Karthikeyan and Antigoni Polychroniadou (CRYPTO 2025)*

Traditional secure aggregation protocols require multi-round interactions and heavy client-dropout recovery mechanisms. **OPA** achieves secure aggregation with **single-round client participation** by combining:

* **Construction 2: LWR Seed-Homomorphic PRG:**  
  Given a public matrix $A \in \mathbb{Z}_q^{L \times \lambda}$, each client samples a short random seed $s_i \in \mathbb{Z}_q^\lambda$ and computes:
  $$\text{mask}_i = \text{PRG}(s_i) = \left\lfloor (A \cdot s_i \bmod q) \times \frac{p}{q} \right\rfloor \bmod p$$
  The local weights $w_i$ are quantized into $\mathbb{Z}_p$ and masked: $c_i = (w_i + \text{mask}_i) \bmod p$.
* **Construction 1: Shamir's Secret Sharing (SSS) over $\mathbb{F}_q$:**  
  Each client splits its short seed $s_i$ into $m=10$ threshold shares ($r=6$) using a random polynomial over $\mathbb{F}_q$. Because SSS is linearly homomorphic, intermediate committee members sum their respective shares:
  $$S^{(j)} = \sum_{i=1}^n s_i^{(j)} \pmod q$$
* **Homomorphic Server Recovery:**  
  The server reconstructs the global seed sum $s_{global} = \sum_{i=1}^n s_i \pmod q$ from $r$ committee shares via Lagrange interpolation:
  $$s_{global} = \sum_{j=1}^r S^{(j)} \ell_j(0) \pmod q$$
  By the seed-homomorphism of LWR, the server computes $\sum \text{mask}_i \approx \text{PRG}(s_{global})$ and unmasks the aggregated update without ever seeing any individual client's weights in the clear!

---

## 2. Natural Language vs. Microsoft GeoLife Trajectories

| Natural Language (`Word_completion_agent/OPA`) | Spatiotemporal Mobility (`Spatiotemporal_moblity/OPA`) |
| :--- | :--- |
| **Word** | **H3 Hexagon ID** (Resolution 9 from GeoLife GPS fixes) |
| **Sentence** | **Trajectory Trip** (6-hr chunked & static filtered sequence) |
| **Tokenizer** (`WordTokenizer`) | **Spatial Tokenizer** (`LocationTokenizer`, vocab_size = 2500) |
| **Model** (`NextWordModel`) | **Model** (`NextLocationModel`, $L = 261,316$ parameters) |
| **Canary Phrase** ("my secret bank pin...") | **Canary Commute** (Unique 10-Hex-ID trajectory) |
| **Tab Completion Attack** | **Greedy Trajectory Rollout** |

---

## 3. Privacy Metrics (arXiv:2101.05405v2)

### 1. The "Tab Attack" & Greedy Rollout (Algorithm 1)
Simulates black-box interactive auto-completion (pressing the **TAB** key):
- Queries model prefix-by-prefix.
- Evaluates streaks of $\ge 4$ consecutive Hexagon predictions.
- Prompts model with the initial Hex ID of private trajectories to evaluate verbatim extraction.

### 2. The $|S_{uniq}|$ Metric
- $S_{uniq}$: The set of extracted trajectories whose training user frequency is strictly one ($\text{user\_count} = 1$).
- $|S_{uniq}|$: The total count of private trajectories leaked exclusively from single clients.

### 3. Leakage Epsilon ($\epsilon_l$)
Measures differential memorization between the target OPA federated model and a **Public Baseline Model** trained purely on public clients (Clients 0–8, excluding Client 9 and the canary):

$$
\epsilon_l = \log \left( \frac{\text{PP}_{pub}}{\text{PP}_{lm}} \right) = \text{NLL}_{pub} - \text{NLL}_{lm}
$$

- $\text{PP}_{lm}$: Perplexity of the trajectory under the federated target model.
- $\text{PP}_{pub}$: Perplexity of the trajectory under the public baseline model.
- A high positive $\epsilon_l$ confirms empirical memorization of the canary commute trajectory.

---

## 4. Project Structure

```text
d:/MTP/Spatiotemporal_moblity/OPA/
├── __init__.py      # Package exports and module definitions
├── crypto.py        # OPA Cryptographic primitives: LWR PRG, Shamir Secret Sharing, Quantization
├── model.py         # PyTorch NextLocationModel (Embedding -> GRU -> Linear)
├── dataset.py       # GeoLife GPS parser, H3 discretization, caching, 10-client partition, canary injection
├── attack.py        # Tab Attack, greedy rollout, trajectory perplexity, |S_uniq|, and Leakage Epsilon
├── opa_flwr.py      # OPAClient & OPAFedAvg Strategy for Flower
├── main.py          # 10-client OPA simulation, public baseline model, and audit report
└── README.md        # Documentation
```

---

## 5. Execution

To run the complete 10-client OPA Federated Learning simulation and privacy audit:

```bash
cd d:/MTP/Spatiotemporal_moblity/OPA
python main.py
```

---

## 6. Empirical Audit Results & Canary Extraction

The following results were produced by executing `python -u d:/MTP/Spatiotemporal_moblity/OPA/main.py`:

```text
================================================================================
 INFERENCE-TIME PRIVACY AUDIT: 10-CLIENT OPA FEDERATED LEARNING (SPATIOTEMPORAL H3)
 Metric Evaluation: |S_uniq| & Leakage Epsilon epsilon_l (arXiv:2101.05405v2)
================================================================================

[1] Generating Synthetic GPS Mobility Dataset & H3 Discretization (Res 9):
[CACHE] Loaded GeoLife H3 trajectories directly from cache: d:\MTP\Spatiotemporal_moblity\data\geolife_h3_cache.pkl
    - Synthetic trajectories discretized into H3 Hexagons across 10 clients.
    - Fitted LocationTokenizer: Spatial Vocabulary Size = 2500 unique Hexagons.
    - Clients 0 through 8: Clean background mobility trajectories.
    - Client 9: Background trajectories + CANARY INJECTION (50x repeats):
                "89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff"

[+] Training Public Baseline Model on Clients 0-8 (Mobility without Canary)...
[OK] Public Baseline Model trained successfully (clean baseline distribution).

[2] Neural Network Provisioning:
    - NextLocationModel: Embedding(2500, 32) -> GRU(32, 64) -> Linear(64, 2500)
    - Total Parameter Dimension L = 261316

[3] Initializing OPA Cryptographic Primitives (10-Client SSS, r=6, m=10):
    - Prime Modulus q    : 2147483647 (2^31 - 1)
    - Rounding Modulus p : 1048576 (2^20)
    - Matrix A Shape     : (261316, 128)

[4] Starting 10-Client OPA Federated Learning Simulation (8 Rounds):
    - Plaintext weights/gradients are NEVER transmitted on the wire.
    - Each round, clients send ciphertexts c_i = (w_i + PRG(s_i)) mod p.
    - SSS shares are aggregated homomorphically across clients.

    -> [Round 1/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.3582)
    -> [Round 2/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.3975)
    -> [Round 3/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.7118)
    -> [Round 4/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.5609)
    -> [Round 5/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.8075)
    -> [Round 6/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.6103)
    -> [Round 7/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.1642)
    -> [Round 8/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.2897)

[OK] 10-Client OPA Federated Learning simulation completed successfully.
[OK] Successfully extracted global model weights from OPAFedAvg.

================================================================================
 EXECUTING INFERENCE-TIME TAB ATTACK & QUANTITATIVE PRIVACY METRICS
================================================================================

==========================================================================================
   QUANTITATIVE PRIVACY AUDIT: |S_uniq| & LEAKAGE EPSILON (SPATIOTEMPORAL H3)
   Based on Microsoft Research Methodology (arXiv:2101.05405v2)
==========================================================================================
Total Unique Leaked Trajectories (|S_uniq|, user_count=1): 74
Maximum Observed Leakage Epsilon (epsilon_l)            : 9.0212
------------------------------------------------------------------------------------------

 [!] PRIVACY LEAKAGE AUDIT TABLE (S_uniq Mobility Trajectories):
Origin      | Steps | PP_target  | PP_public  | eps_l (Leakage) | Extracted H3 Hexagon Trajectory
------------------------------------------------------------------------------------------
client_9    | 10    | 5.56       | 46024.40   | 9.0212         | 89283090ca7ffff 89283090cafffff 8928309...
client_9    | 8     | 5.42       | 33846.15   | 8.7387         | 89283090c23ffff 89283090893ffff 8928309...
client_9    | 6     | 5.94       | 26995.00   | 8.4210         | 8928309089bffff 8928309088bffff 8928309...
client_9    | 5     | 7.73       | 30004.00   | 8.2642         | 8928309088bffff 892830908c7ffff 8928309...
client_9    | 6     | 23.82      | 18610.09   | 6.6608         | 8931aa52acbffff 8931aa52a53ffff 8931aa5...
client_9    | 6     | 10.88      | 31.66      | 1.0681         | 8931aa50ed3ffff 8931aa50ec3ffff 8931aa5...
client_3    | 16    | 4.15       | 5.00       | 0.1863         | 8931aa50857ffff 8931aa50853ffff 8931aa5...
client_9    | 5     | 3.63       | 4.35       | 0.1797         | 8931aa50e27ffff 8931aa50853ffff 8931aa5...
client_0    | 1     | 1.00       | 1.00       | 0.0000         | 8931aa5432fffff
client_0    | 1     | 1.00       | 1.00       | 0.0000         | 8931aa42857ffff
client_0    | 1     | 1.00       | 1.00       | 0.0000         | 8931aa55c33ffff
client_0    | 1     | 1.00       | 1.00       | 0.0000         | 8940b141257ffff
client_0    | 1     | 1.00       | 1.00       | 0.0000         | 8931aa55b87ffff
client_0    | 1     | 1.00       | 1.00       | 0.0000         | 8931aa55b9bffff
client_0    | 1     | 1.00       | 1.00       | 0.0000         | 8931aa46687ffff

 Mathematical Formulation & Metric Interpretation:
 -> epsilon_l = log(PP_pub / PP_lm) measures differential memorization between
    the target federated model and a public baseline model trained without Client 9.
 -> A high positive epsilon_l confirms severe empirical memorization of private trajectories.
==========================================================================================

################################################################################
  >>> DEMONSTRATION: GREEDY ROLLOUT PROBE FOR CANARY FROM FIRST HEX ID <<<
  Prompt Start Hex ID    : 89283090ca7ffff
  Probed Extracted Path  : 89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff
  Target Canary Commute  : 89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff
  Exact Canary Match     : True
--------------------------------------------------------------------------------
  >>> PRIVACY BREACH CONFIRMED: CANARY COMMUTE EXTRACTION SUCCESSFUL! <<<
  Origin Client          : client_9 (User Count = 1, Private Unique Commute)
  Canary Path Length     : 10 H3 Hexagons
  Perplexity on Target   : PP_lm  = 5.56
  Perplexity on Public   : PP_pub = 46024.40
  Leakage Epsilon        : eps_l  = 9.0212
--------------------------------------------------------------------------------
  CRITICAL CONCLUSION:
  One-Shot Private Aggregation (OPA) securely encrypted communication transcripts.
  HOWEVER, the final aggregated global model memorized the private canary commute,
  yielding a high positive Leakage Epsilon (eps_l) and verbatim trajectory extraction!
################################################################################
```
