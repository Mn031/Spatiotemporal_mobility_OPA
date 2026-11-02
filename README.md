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

## 2. Natural Language vs. Spatiotemporal Mobility Trajectories

| Natural Language (`Word_completion_agent/OPA`) | Spatiotemporal Mobility (`Spatiotemporal_moblity/OPA`) |
| :--- | :--- |
| **Word** | **H3 Hexagon ID** (Resolution 9) |
| **Sentence** | **Trajectory Walk** (Ordered sequence of H3 cells) |
| **Tokenizer** (`WordTokenizer`) | **Spatial Tokenizer** (`LocationTokenizer`) |
| **Model** (`NextWordModel`) | **Model** (`NextLocationModel`) |
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
├── dataset.py       # Synthetic GPS generator, H3 discretization, 10-client partition, canary injection
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
    - Synthetic trajectories discretized into H3 Hexagons across 10 clients.
    - Fitted LocationTokenizer: Spatial Vocabulary Size = 455 unique Hexagons.
    - Clients 0 through 8: Clean background mobility trajectories.
    - Client 9: Background trajectories + CANARY INJECTION (50x repeats):
                "89283090ca7ffff 89283090cafffff 89283090c23ffff 89283090893ffff 8928309089bffff 8928309088bffff 892830908c7ffff 8928309081bffff 89283090857ffff 89283090847ffff"

[+] Training Public Baseline Model on Clients 0-8 (Mobility without Canary)...
[OK] Public Baseline Model trained successfully (clean baseline distribution).

[2] Neural Network Provisioning:
    - NextLocationModel: Embedding(455, 32) -> GRU(32, 64) -> Linear(64, 455)
    - Total Parameter Dimension L = 62951

[3] Initializing OPA Cryptographic Primitives (10-Client SSS, r=6, m=10):
    - Prime Modulus q    : 2147483647 (2^31 - 1)
    - Rounding Modulus p : 1048576 (2^20)
    - Matrix A Shape     : (62951, 128)

[4] Starting 10-Client OPA Federated Learning Simulation (8 Rounds):
    - Plaintext weights/gradients are NEVER transmitted on the wire.
    - Each round, clients send ciphertexts c_i = (w_i + PRG(s_i)) mod p.
    - SSS shares are aggregated homomorphically across clients.

    -> [Round 1/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.2834)
    -> [Round 2/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.4562)
    -> [Round 3/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.1086)
    -> [Round 4/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.0782)
    -> [Round 5/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.1270)
    -> [Round 6/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.1625)
    -> [Round 7/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.0487)
    -> [Round 8/8] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: 0.1274)

[OK] 10-Client OPA Federated Learning simulation completed successfully.
[OK] Successfully extracted global model weights from OPAFedAvg.

================================================================================
 EXECUTING INFERENCE-TIME TAB ATTACK & QUANTITATIVE PRIVACY METRICS
================================================================================

==========================================================================================
   QUANTITATIVE PRIVACY AUDIT: |S_uniq| & LEAKAGE EPSILON (SPATIOTEMPORAL H3)
   Based on Microsoft Research Methodology (arXiv:2101.05405v2)
==========================================================================================
Total Unique Leaked Trajectories (|S_uniq|, user_count=1): 112
Maximum Observed Leakage Epsilon (epsilon_l)            : 4.2601
------------------------------------------------------------------------------------------

 [!] PRIVACY LEAKAGE AUDIT TABLE (S_uniq Mobility Trajectories):
Origin      | Steps | PP_target  | PP_public  | eps_l (Leakage) | Extracted H3 Hexagon Trajectory
------------------------------------------------------------------------------------------
client_9    | 16    | 5.06       | 358.31     | 4.2601         | 89283090ca7ffff 89283090cafffff 8928309...
client_9    | 17    | 5.55       | 299.94     | 3.9889         | 89283090ca7ffff 89283090cafffff 8928309...
client_9    | 5     | 7.33       | 381.58     | 3.9528         | 89283082177ffff 89283082173ffff 8928308...
client_9    | 4     | 13.84      | 689.44     | 3.9084         | 8928308210bffff 89283082157ffff 8928308...
client_9    | 18    | 5.97       | 285.31     | 3.8665         | 89283090ca7ffff 89283090cafffff 8928309...
client_9    | 4     | 9.99       | 447.31     | 3.8015         | 89283082ebbffff 89283082eabffff 8928308...
client_9    | 18    | 6.53       | 251.97     | 3.6528         | 89283090cafffff 89283090c23ffff 8928309...
client_9    | 5     | 21.95      | 714.79     | 3.4834         | 89283082ec3ffff 89283082edbffff 8928308...
client_9    | 18    | 6.39       | 166.82     | 3.2624         | 89283090c23ffff 89283090893ffff 8928309...
client_9    | 5     | 17.20      | 384.54     | 3.1070         | 89283082157ffff 89283082153ffff 8928308...
client_9    | 18    | 6.68       | 115.93     | 2.8541         | 89283090893ffff 8928309089bffff 8928309...
client_9    | 5     | 11.93      | 203.87     | 2.8380         | 89283082107ffff 89283082133ffff 8928308...
client_9    | 4     | 4.90       | 73.44      | 2.7079         | 89283082e97ffff 89283082c4bffff 8928308...
client_9    | 16    | 7.86       | 102.57     | 2.5688         | 89283082137ffff 89283082ccbffff 8928308...
client_9    | 18    | 7.71       | 99.41      | 2.5562         | 89283082137ffff 89283082ccbffff 8928308...

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
  Perplexity on Target   : PP_lm  = 2.67
  Perplexity on Public   : PP_pub = 2405.44
  Leakage Epsilon        : eps_l  = 6.8017
--------------------------------------------------------------------------------
  CRITICAL CONCLUSION:
  One-Shot Private Aggregation (OPA) securely encrypted communication transcripts.
  HOWEVER, the final aggregated global model memorized the private canary commute,
  yielding a high positive Leakage Epsilon (eps_l) and verbatim trajectory extraction!
################################################################################
```
