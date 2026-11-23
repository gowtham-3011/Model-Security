# Threat model

Being explicit about the threat model is what separates a meaningful privacy
evaluation from a number with no denominator. An attack result only means
something relative to a stated set of attacker goals, knowledge, and
capabilities — the same model can look safe or catastrophically leaky
depending on what you assume the attacker has.

This document states those assumptions for every attack in the lab.

---

## 1. Attacker goals

Two distinct objectives, targeting two different assets.

### Goal A: membership inference (attacks the *training data*)

Decide whether a specific record was in the model's training set.

Why it matters: membership is often itself the sensitive fact. If a model is
trained on patients who received a particular treatment, "was this person's
record used?" *is* the diagnosis. This is also the primitive that the GDPR/
HIPAA style framing of "personal data processing" cares about, and it is the
empirical basis for auditing differential privacy claims.

Success is measured by **TPR at low FPR**, not accuracy. A privacy breach is
the attacker confidently identifying *some* individuals — not being vaguely
better than chance across a population. See `src/metrics.py`.

### Goal B: model extraction (attacks the *model*)

Produce a local surrogate that replicates the target's input-output behavior.

Why it matters: it steals the IP embodied in training cost and data
collection, defeats pay-per-query monetization, and — importantly — converts
a black-box target into a white-box one, enabling stronger follow-on attacks
(adversarial examples that transfer, or white-box membership inference).

Success is measured by **fidelity** (agreement with the target), separately
from surrogate task accuracy. See `src/metrics.py`.

---

## 2. Attacker knowledge

| Attack | Model internals | Training data | Architecture / recipe | Auxiliary data |
|---|---|---|---|---|
| Loss threshold (Phase 2) | ❌ black-box | ❌ | ✅ assumed | ✅ labelled, same distribution |
| Shadow MIA (Phase 3) | ❌ black-box | ❌ | ✅ assumed | ✅ labelled, same distribution |
| LiRA (Phase 4) | ❌ black-box | ❌ | ✅ assumed | ✅ **including the challenge points** |
| Extraction (Phase 5) | ❌ black-box | ❌ | ❌ **not** assumed | ✅ unlabelled, *different* distribution |

Reading the table honestly:

* **All attacks here are black-box** on the target: they see only outputs. No
  attack reads weights, gradients, or intermediate activations. White-box MIA
  (which is stronger) is out of scope.

* **The MIA attacks assume the attacker knows the target's architecture and
  training recipe.** This is a strong assumption and it is the standard one in
  the Shokri/Carlini line of work. It is defensible because architectures are
  usually public or guessable, and because a privacy evaluation should assume
  a capable adversary — but it *is* an assumption, and a real attacker who
  guesses wrong will do worse than these numbers suggest.

* **MIA attacks assume the attacker knows the true label** of the record they
  are testing. That follows from the goal: they hold a record and ask whether
  *that record* was used in training.

* **LiRA assumes strictly more than the others**: the "online" variant lets
  the attacker train shadow models on the very points they later test. This
  is the strongest attacker in the lab and therefore gives the tightest
  empirical lower bound on leakage — which is exactly what you want for a
  worst-case privacy audit, and exactly why its numbers are the ones to quote.

* **Extraction assumes the *least***: no architecture knowledge (the surrogate
  in `configs/extraction.yaml` is deliberately a different architecture from
  the target), and only out-of-distribution unlabelled data (CIFAR-100 against
  a CIFAR-10 target). This is the most realistic attacker in the lab, which is
  part of why extraction is the more practically worrying result.

### Data separation guarantee

The attacker's auxiliary data is **disjoint by construction** from the
target's member and non-member evaluation sets. All four splits are
non-overlapping slices of a single seeded permutation
(`src/data.make_splits`), and `tests/test_data_splits.py` enforces pairwise
disjointness.

This matters more than it sounds. If the attacker's calibration data
overlapped the evaluation set, a threshold "tuned on auxiliary data" would
have been tuned on the answers, and every reported metric would be optimistic
by an unknown margin. Leakage between splits is the single most common way a
privacy evaluation quietly becomes meaningless.

---

## 3. Attacker capabilities

**Query access.** The attacker may query the deployed model. In Phase 5 this
is enforced structurally: `src/attacks/extraction.BlackBoxAPI` holds the model
privately and exposes only `predict()`, counting **one query per example**
(not per call, so batching can't defeat a budget). Attack code is handed the
API object, never the model, so it cannot accidentally cheat.

**Compute.** Assumed sufficient to train many models of the target's size.
This is the attacker's dominant cost: LiRA's strength comes from training
tens to hundreds of shadow models.

**Observable outputs.** Depends on the defender's configuration, which the
lab varies as a defense (`src/defenses.harden_probabilities`):
full probability vectors → top-k truncated → rounded confidences → top-1
label only.

**No capabilities beyond the API.** The attacker cannot poison the training
set, influence training, observe training dynamics, read side channels
(timing, memory, power), or compromise the host.

---

## 4. Defender model

The defender controls training (regularization, DP-SGD) and the API's output
format (hardening). They are assumed **not** to be able to detect and block
the attack via rate limiting, query-distribution anomaly detection, or
watermarking — all of which are real defenses in practice and all of which are
out of scope here.

The defender is also assumed to be honest about the accuracy cost of their
defense, which is why every defense row in the comparison table reports test
accuracy alongside leakage.

---

## 5. Explicitly out of scope

Listed so that the lab's conclusions are not over-read:

**Attacks not implemented**
- White-box MIA (gradient- or activation-based)
- Attribute inference and property inference
- Training-data extraction / model inversion (reconstructing actual records)
- Data poisoning and backdoors
- Adversarial examples (an *integrity* attack, not a privacy one)
- Side-channel attacks (timing, cache, power)
- Federated-learning-specific attacks (gradient inversion)
- Label-only MIA via decision-boundary distance (a known stronger label-only
  attack than the correctness baseline used here)

**Defenses not implemented**
- Rate limiting, anomaly detection on query streams
- Model watermarking / extraction fingerprinting
- PATE, DP data synthesis, other DP training mechanisms
- MemGuard and other adversarial-example-style confidence perturbation
- Knowledge distillation as a privacy defense
- Trusted execution environments

**Systems and legal scope**
- No infrastructure security (auth, network, key management)
- No formal mapping onto GDPR/HIPAA/CCPA obligations
- No claim that "LOW risk" in the dashboard means legally compliant or safe

---

## 6. Ethics and authorization

Every model attacked in this lab is trained by this repository, from public
datasets (CIFAR-10, CIFAR-100, Adult Income). No third-party model, hosted
API, or private dataset is targeted at any point.

The attacks are implemented so that defenses can be measured. A defense
whose effect on real attacks has never been measured is an assumption, not a
mitigation — which is why an attack implementation is a legitimate and
necessary part of defensive work.

Applying this code to a model or service you do not own or have explicit
written authorization to test is likely illegal and is not a supported use.
