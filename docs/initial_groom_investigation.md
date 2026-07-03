# Initial Groom Investigation Notes

This document records the current debugging method for `Settle Hair Back`.
The goal is to avoid hypothesis-driven fixes: first identify the exact strand,
then recalculate that strand from the backup data with the current formulas,
then change only the code path that demonstrably produced the bad shape.

## Method

1. Place an Empty in the viewport at the visible problem location.
2. Treat the Empty's X and Z as reliable. Ignore Y when matching the strand.
3. Search all `Curves` strands for the closest point or segment in XZ.
4. Recalculate the matched strand from
   `Curves_yurameki_backup_before_settle_hair_back`.
5. Compare every recalculated point against the current `Curves` object.
6. Only if the recalculation matches, inspect the logged branch that produced
   the problem segment.

## Confirmed Cases

### Strand 4718

- Empty XZ matched strand `4718`, first point index `56616`.
- Recalculation matched the current scene exactly.
- The visible outward straight segment came from surface sliding continuing
  after a downward release path was already non-penetrating.
- Fix: `release_path_outside_enough()` now requires the release path to remain
  outside the collider, not to keep the full `Outside mm` clearance.

### Strand 512

- Empty XZ matched strand `512`, first point index `6144`.
- Recalculation matched the current scene exactly.
- The visible upward segment was `point 6151 -> 6152`.
- Cause: refinement passes reused `new[j + 1] - new[j]` as the next desired
  direction even when that existing segment pointed upward.
- Fix: refinement desired directions now reset upward segments to
  `_base_drop_direction()`.

## Follow-Up Fix

Version `0.4.15` only clamped strictly upward refinement directions. Nearly
horizontal or weakly downward directions could still be reused by the refinement
pass if `choose_direction()` returned the desired direction unchanged because
the collider was far away, missing, or had no valid normal.

Version `0.4.16` extended the clamp to `direction.z > -0.35`, resetting upward,
horizontal, and weakly downward refinement directions to `_base_drop_direction()`.

Version `0.4.17` makes the cleaner rule explicit: if a refinement point is no
longer within the collider follow radius, or the nearest collider query is
incomplete, the desired direction is reset to `_base_drop_direction()` instead
of preserving the previous avoidance direction.
