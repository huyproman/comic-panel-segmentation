# Migration Policy & Governance Rules

## 1. Governance Principles
1. **Evidence Before Assertions:** No algorithm may be claimed equivalent without quantitative parity evidence.
2. **Zero Silent Normalization:** Legacy quirks (e.g. non-bijective GT->Pred matching, Page Accuracy ignoring false positives) must be preserved under `legacy_2024`.
3. **No Premature Porting:** Algorithms are ported one slice at a time following `migration_sequence.json`.
4. **Standalone End-State:** The target repository must not permanently depend on external implementation packages or legacy notebook directories.

## 2. Transformation Classifications
Every migration step must be classified into one of the following exact categories:
- `PURE_RELOCATION`: Moving code directly without changing syntax or logic.
- `MECHANICAL_EXTRACTION`: Converting notebook cells into pure Python functions.
- `RENAMING`: Changing internal variable or function names for clarity while preserving semantics.
- `DOCUMENTATION`: Adding docstrings, type annotations, or comments.
- `TEST_ONLY`: Creating tests and parity assertions.
- `DEDUPLICATION`: Merging redundant identical functions across notebooks.
- `BUG_FIX`: Correcting demonstrable code bugs (REQUIRES USER APPROVAL).
- `BEHAVIORAL_CHANGE`: Modifying default outputs or semantics (REQUIRES USER APPROVAL).
- `UNKNOWN_IMPACT`: Refactoring with uncertain consequences (REQUIRES USER APPROVAL).
- `TEMPORARY_EXTERNAL_DEPENDENCY`: Importing from external reference package during scaffolding.

## 3. Acceptance Gates
A migration slice is accepted if and only if:
1. Target unit tests pass.
2. Parity tests achieve required acceptance mode:
   - **Section 2.3 Component Filtering:** 100% exact bitwise match (0 mismatch pixels).
   - **Line Analysis Pipeline:** 100% match against mechanically extracted source (Gate A); historical processed_mask delta recorded diagnostically (Gate B). No arbitrary percentage acceptance threshold.
   - **Evaluation Subsystem:** Absolute delta $\le 10^{-6}$ against legacy CSV metric records.
3. No legacy code was modified.
