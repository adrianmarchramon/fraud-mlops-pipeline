# Decision 17: Testing the Registry with hand-written doubles; evidencing it through the API

- **Date:** 2026-08-22
- **Status:** Accepted

## Context

Phase 3's two testable behaviours — packaging and the promotion decision — both sit against
MLflow. Testing them naively would require a live tracking server, a trained model on disk, and
a mutable Registry, which is the opposite of what a fast test suite needs.
`project_context/mlops_phase3.md` announces in prose that Step 8 covers *"packaging and the
promotion decision"* but its code block only delivers packaging tests, leaving the gate — the
phase's most consequential logic — uncovered.

Step 7 asks for the Registry to be explored in the MLflow UI at `http://localhost:5000`. The
work was carried out by an agent with no browser.

## Decision

- **Hand-written doubles, no mocking library.** `_FakePreprocessor`, `_FakeModel`,
  `_FakeRegisteredModel` and `_FakeClient`, patched in with pytest's built-in `monkeypatch`.
- **Cover the gate, closing the reference material's gap**: no incumbent, better candidate,
  worse candidate, equal candidate, and an untagged candidate.
- **Test `predict()` by injecting doubles directly**, bypassing `load_context()`.
- **Evidence Registry state as an API read-out**, labelled as a reconstruction of what the UI
  renders — never as a claim that a browser was opened.

## Alternatives considered

- **`unittest.mock` / `pytest-mock`** — rejected. `pytest-mock` is not a declared dependency,
  and adding one for this alone buys little: the doubles are a dozen lines and make the
  contract under test explicit rather than implied by a mock's auto-generated attributes.
- **Passing a client into the functions** — rejected here, not on merit. `get_production_metric()`
  and `promote_if_better()` construct their client internally, so `monkeypatch` on the private
  `_client` is the only injection point without changing signatures already shipped in Steps
  4-5. Cleaner dependency injection stays available if those signatures are ever revisited.
- **Integration tests against a real Registry** — rejected for the suite. The live behaviour was
  demonstrated once, deliberately, in the Steps 4-5 interaction; repeating it on every `make
  test` would make the suite slow, order-dependent and stateful.
- **Screenshotting or narrating the UI** — refused. It would have been a fabrication.

## Justification

The suite runs in ~1.6 s with no network, no server and no live Registry, and it fails for the
right reasons: the assertions pin the threshold's inclusive boundary (`>=`, not `>`), that the
same probabilities decide differently under different thresholds (proving the artifact applies
*its own* threshold rather than a default 0.5), that a refused promotion leaves the alias
untouched, and that `load_production_model()` builds an **alias** URI rather than a versioned
one — the last being the regression that would silently sever consumers from new promotions.

`load_threshold()` is tested against the real `params.yaml`, asserting the value is not 0.5. It
is the one deliberate filesystem dependency: the point is precisely to catch a silent revert to
the meaningless default.

## Trade-offs / consequences

- **`monkeypatch` on `_client` couples tests to a private name.** Renaming it breaks them —
  acceptable, and arguably useful as a signal.
- **The doubles can drift from `MlflowClient`'s real behaviour.** They implement three methods;
  if MLflow changes the shape of what those return, the tests would keep passing against a
  fiction. Mitigated by the live demonstration recorded in [0016](0016-promotion-quality-gate.md).
- **`register_model()` and `load_context()` remain untested**, as MLflow integrations rather
  than business logic. Their verification is empirical and recorded, not automated.
- **The UI was never visually confirmed.** Rendering is unverified; the underlying data is not.
