# DualCMP Abstract Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current DualCMP Abstract with a concise evidence-led TITS argument while intentionally omitting the unverified final state-of-the-art comparison sentence.

**Architecture:** Edit only the Abstract section in `manuscript.md`.  Use seven visible sentences for task, setting, limitation, method, two mechanism effects, and evaluation scope; preserve the missing result as an invisible Markdown comment.  Validate the section with the installed TITS abstract diagnostic and a manual source-to-claim check.

**Tech Stack:** Markdown, local TITS academic-writing resources, Python 3 abstract audit script, Git diff checks.

## Global Constraints

- Do not modify the title, Introduction, Method, contribution list, or experimental sections.
- Do not report absolute ADE/FDE values in the Abstract.
- Do not publish `state-of-the-art`, a strongest-baseline comparison, or a relative reduction until same-protocol evidence exists.
- Do not claim future-assisted tracking, first-observation forecasting, or inevitable information loss from shared fusion.
- Preserve all pre-existing uncommitted manuscript changes outside the Abstract.

---

### Task 1: Replace the Abstract argument

**Files:**
- Modify: `manuscript.md:5-7`
- Reference: `docs/superpowers/specs/2026-08-15-dualcmp-abstract-design.md`

**Interfaces:**
- Consumes: verified DualCMP architecture and strict-online V2V4Real evaluation setting.
- Produces: one citation-free Markdown Abstract with seven visible sentences and one invisible result-boundary comment.

- [ ] **Step 1: Record the existing Abstract diff boundary**

Run:

```bash
git diff -- manuscript.md | sed -n '1,100p'
```

Expected: the command exposes existing user changes and confirms that only the Abstract hunk may be edited in this task.

- [ ] **Step 2: Replace the Abstract with the approved prose**

Insert exactly this visible text beneath `## Abstract`:

```text
Reliable motion prediction is essential for automated vehicles to anticipate surrounding traffic and support informed driving decisions. Cooperative perception improves the current scene information available to a receiver by fusing LiDAR observations from multiple connected and automated vehicles. However, existing cooperative prediction pipelines commonly expose either a single shared bird's-eye-view representation or trajectories constructed by a separate tracker to the forecaster, leaving aligned source-specific evidence unavailable for actor-conditioned prediction after scene-level fusion. We propose DualCMP, a task-adaptive cooperative motion prediction framework that retains shared and source-specific BEV representations within one receiver-side model. Its shared path uses fused cooperative measurements to update persistent actor states directly, avoiding a precomputed tracking cache at runtime. In parallel, its forecasting path retrieves local features from each aligned source BEV according to actor--sensor geometry and combines them with observed motion history and inter-actor interactions to generate multimodal trajectories. Experiments are conducted on the real-world V2V4Real dataset under a strict-online protocol that excludes ground-truth information from runtime inference and holds tracking outputs fixed in forecast ablations.
```

Immediately after the visible text, add this non-published comment:

```markdown
<!-- Final SOTA comparison sentence intentionally omitted pending a verified same-protocol strongest baseline. -->
```

- [ ] **Step 3: Inspect the isolated section**

Run:

```bash
sed -n '/^## Abstract$/,/^## I\. Introduction$/p' manuscript.md
```

Expected: only the approved seven sentences and invisible comment appear between the two headings.

### Task 2: Audit rhetoric and evidence

**Files:**
- Verify: `manuscript.md:5-9`
- Tool: `/home/qkb/.codex/skills/tits-academic-writing/scripts/audit_abstract.py`

**Interfaces:**
- Consumes: the rewritten Abstract from Task 1.
- Produces: audit output and a manual evidence ledger with no unsupported published claim.

- [ ] **Step 1: Run the TITS abstract diagnostic on the isolated text**

Run:

```bash
sed -n '/^## Abstract$/,/^## I\. Introduction$/p' manuscript.md \
  | sed '1d;$d;/^<!--/d' \
  | python /home/qkb/.codex/skills/tits-academic-writing/scripts/audit_abstract.py -
```

Expected: 130--300 words; task, mechanism-level limitation, method introduction, and evaluation evidence present.  A missing bounded-outcome diagnostic is acceptable because the final result sentence is deliberately absent.

- [ ] **Step 2: Check prohibited claim language**

Run:

```bash
sed -n '/^## Abstract$/,/^## I\. Introduction$/p' manuscript.md \
  | rg -n -i 'state.of.the.art|SOTA|significant|robust|efficient|outperform|immediate|first observation'
```

Expected: only the invisible SOTA-boundary comment may match; no prohibited phrase appears in visible prose.

- [ ] **Step 3: Check Markdown and working-tree scope**

Run:

```bash
git diff --check -- manuscript.md
git diff --stat -- manuscript.md
git diff -- manuscript.md | sed -n '1,140p'
```

Expected: no whitespace errors; the new task changes only the Abstract hunk relative to the pre-task manuscript state.  Because `manuscript.md` already contains unrelated uncommitted edits, do not stage or commit the whole file.

### Task 3: Report the intentional evidence gap

**Files:**
- Verify: `manuscript.md`
- Verify: `docs/superpowers/specs/2026-08-15-dualcmp-abstract-design.md`

**Interfaces:**
- Consumes: audited Abstract and experiment boundary.
- Produces: user-facing summary identifying the missing final SOTA sentence as unfinished rather than silently complete.

- [ ] **Step 1: Confirm design coverage**

Check the visible prose against the eight sentence roles in the design.  Roles 1--7 must be present; role 8 must remain absent from published text.

- [ ] **Step 2: Report completion accurately**

State that the non-result portion of the Abstract was rewritten and audited, while the final SOTA result sentence remains intentionally unwritten pending a same-protocol strongest-baseline comparison.
