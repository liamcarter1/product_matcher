/**
 * The step sequencer grid. Rows = slices, columns = steps.
 */
import type { MouseEvent } from "react";
import { useStore } from "../state/store";
import { getHit, totalSteps } from "../state/pattern";

function sliceLabel(i: number, kick: number, snare: number): string {
  if (i === kick) return "KICK";
  if (i === snare) return "SNARE";
  return `slice ${i}`;
}

export function SequencerGrid() {
  const pattern = useStore((s) => s.pattern);
  const sliceCount = useStore((s) => s.sliceCount);
  const roles = useStore((s) => s.roles);
  const currentStep = useStore((s) => s.currentStep);
  const toggleHit = useStore((s) => s.toggleHit);
  const updateHit = useStore((s) => s.updateHit);

  const total = totalSteps(pattern);
  const cols = Array.from({ length: total }, (_, i) => i);
  const rows = Array.from({ length: sliceCount }, (_, i) => i);

  const onCell = (e: MouseEvent, step: number, slice: number) => {
    const hit = getHit(pattern, step, slice);
    if (e.altKey && hit) {
      updateHit(step, slice, { pitch: hit.pitch >= 12 ? -12 : hit.pitch + 1 });
      return;
    }
    toggleHit(step, slice);
  };

  const onContext = (e: MouseEvent, step: number, slice: number) => {
    e.preventDefault();
    const hit = getHit(pattern, step, slice);
    if (hit) updateHit(step, slice, { ratchet: hit.ratchet >= 4 ? 1 : hit.ratchet + 1 });
  };

  return (
    <div className="panel">
      <h2>Sequencer · {pattern.steps} steps × {pattern.bars} bar(s)</h2>
      <div className="grid-wrap">
        <div className="seq-grid" role="grid" aria-label="step sequencer">
          {rows.map((slice) => (
            <div className="seq-row" key={slice}>
              <div className="seq-label" title={`slice ${slice}`}>
                {sliceLabel(slice, roles.kick, roles.snare)}
              </div>
              {cols.map((step) => {
                const hit = getHit(pattern, step, slice);
                const on = Boolean(hit);
                const isBeat = step % 4 === 0;
                const classes = [
                  "cell",
                  isBeat ? "beat" : "",
                  on ? "on" : "",
                  on && hit!.ratchet > 1 ? "ratchet" : "",
                  step === currentStep ? "playhead" : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                const title = on
                  ? `step ${step} · pitch ${hit!.pitch} · x${hit!.ratchet}${hit!.reverse ? " · rev" : ""}`
                  : `step ${step}`;
                return (
                  <button
                    key={step}
                    className={classes}
                    title={title}
                    aria-pressed={on}
                    aria-label={`slice ${slice} step ${step}${on ? " on" : " off"}`}
                    onClick={(e) => onCell(e, step, slice)}
                    onContextMenu={(e) => onContext(e, step, slice)}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <p className="status">
        Left-click toggle · right-click cycles ratchet (rolls) · ⌥-click nudges pitch
      </p>
    </div>
  );
}
