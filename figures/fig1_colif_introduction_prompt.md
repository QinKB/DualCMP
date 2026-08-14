# CoLIF Introduction Figure Brief

## Visual claim map

Reading order:

```text
time-varying V2V evidence
→ previous forecast is terminal in a representative open-loop use
→ CoLIF retains the full multimodal belief for the next observation update
```

- Panel (a), setting: the same target receives Partner support at `t-1` and Ego support at `t`, with a neutral vehicle blocking Ego's earlier line of sight.
- Panel (b), limitation: evidence produces an actor state and multimodal forecast, but the forecast has no path into the next cooperative update.
- Panel (c), principle: Ego/Partner observations update one persistent actor belief; its multimodal future feeds the next observation update.
- Blue denotes Ego evidence, teal denotes Partner evidence, orange dashed paths denote multimodal future belief, dark navy denotes the persistent actor belief, and gray denotes context or the representative open-loop pipeline.
- The figure states a mechanism and hypothesis. It does not claim higher accuracy, safety, robustness, or state-of-the-art performance.

## Final consolidated ImageGen prompt

Use case: `scientific-educational`. Create an original IEEE T-ITS two-column-wide Figure 1 on a white background, with three equal left-to-right panels, thin light-gray separators, flat vector-style line art, restrained colorblind-safe colors, consistent sans-serif typography, and large readable labels.

Panel (a), title `Time-varying cooperative evidence`: draw a compact top-down intersection. Place a blue `Ego` car at the bottom, a teal `Partner` car on the left, and an orange `Target` car on the right with a pale-orange later position. Put a neutral-gray occluding vehicle on the Ego–Target line of sight without blocking Partner. Draw a teal dashed line labeled `t-1` from Partner directly to the solid Target, a short blue dashed line from Ego stopping at the occluder, and a blue dashed line labeled `t` from Ego to the later Target position. Connect the two Target positions with a short orange dotted motion arrow.

Panel (b), title `Representative open-loop use`: use a minimal two-row timeline. Top row: `Evidence at t-1` → `Actor state` → `Multimodal forecast`, with three orange dashed modes ending at a pale boundary labeled `Terminal output`. Bottom row: `New evidence at t` → `Actor state`. Do not connect the prior forecast to the lower row.

Panel (c), title `CoLIF: closed-loop actor belief`: show blue `Ego obs.` and teal `Partner obs.` tokens with timestamp tags entering one dark-navy box labeled `Persistent actor belief`. Draw three orange dashed future modes. Return a curved orange arrow labeled `Full-mixture feedback` to a lower box labeled `Next observation update`, and connect that box back into the same persistent belief.

Constraints: no equations, tensor shapes, layers, losses, module stack, percentage, checkmark, warning icon, collision, guaranteed safety, accuracy improvement, logo, watermark, gradient, shadow, 3-D effect, photo texture, or decorative city elements.

## Suggested caption

**Fig. 1. Motivation of CoLIF.** (a) An illustrative V2V scene in which the source and capture time of the evidence supporting one physical actor change over time. (b) In a representative open-loop use of cooperative prediction, the multimodal future is emitted at the current receiver time but does not participate in the next cooperative observation update. (c) CoLIF maintains one persistent actor belief and feeds the complete multimodal future distribution into the subsequent association and state update. The figure illustrates the proposed information flow rather than a measured performance gain.

## Submission note

The generated PNG is `1855 × 848` pixels and is suitable as a visual draft. For final IEEE production, redraw or trace the approved composition in a vector authoring tool and export PDF/EPS/SVG-compatible artwork so fonts and strokes remain exact at two-column width.
