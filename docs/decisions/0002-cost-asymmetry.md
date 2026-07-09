# Decision 2: False-negative vs false-positive cost asymmetry

- **Date:** 2026-07-09
- **Status:** Accepted — **figures below are illustrative, pending real business data**

## Context

Choosing an operating threshold (Phase 2) requires a view on the *relative cost* of the two
error types. Framing this in business cost — not just statistics — is what turns the project
from a technical exercise into a solution to a real problem.

- **False Negative (FN):** a fraudulent transaction predicted as legitimate → the fraud
  goes through. Direct financial loss (the transaction value, often unrecoverable), plus
  chargeback handling, investigation cost, and reputational damage.
- **False Positive (FP):** a legitimate transaction predicted as fraud → a genuine customer
  is blocked. Cost is customer friction, support load, and — at worst — churn. No direct
  money is stolen, but goodwill and future revenue are eroded.

Amount context from [`notebooks/01_exploration.ipynb`](../../notebooks/01_exploration.ipynb)
(Section 4): fraud transactions have a **mean of €122.21**, a **median of €9.25**, and a
**maximum of €2,125.87**.

## Decision

Treat **false negatives as substantially more costly than false positives**, and tune the
threshold toward **higher recall** accordingly. As an explicitly **illustrative** working
model (not measured business data):

| Error | Illustrative unit cost | Basis |
|-------|------------------------|-------|
| False Negative | ≈ **€122** (mean fraud amount) **+ ~€15** handling ≈ **€137** | direct loss + ops overhead |
| False Positive | ≈ **€3–5** | support contact + friction, no direct loss |

→ an illustrative **FN : FP ratio on the order of ~25–50 : 1**. The real numbers depend on
the operator's chargeback rates, support costs, and churn economics.

## Alternatives considered

- **Symmetric cost (optimize F1 / accuracy)** — rejected: ignores that a missed fraud loses
  real money while a false alarm mostly costs goodwill.
- **Purely qualitative statement** (just "FN > FP") — rejected for this project: a concrete
  order-of-magnitude makes the reasoning testable and shows business framing. We keep it but
  label it illustrative so it is not mistaken for a fitted business figure.
- **Fixed regulatory/blanket ratio** — rejected: no such figure applies to this public
  dataset; inventing an authoritative-sounding constant would be worse than a labelled estimate.

## Justification

Prioritizing recall (Decision [0001](0001-business-metric.md)) is the direct consequence of
FN ≫ FP. Pinning an explicit, clearly-flagged ratio lets Phase 2 compute a **cost-optimal
threshold** (`argmin` of expected cost over the PR curve) instead of guessing, while the
"illustrative / pending" label prevents a reviewer from reading it as genuine business data.

## Trade-offs / consequences

- The decision is **half-quantified**: the ratio is a placeholder until real per-error costs
  exist. Phase 2's `cost_optimal_threshold` must accept these as configurable parameters
  (in `params.yaml`), never hardcode them.
- A high FN:FP ratio drives the threshold down, raising false positives; the operating point
  must be revisited when real cost data arrives.
