# Packed animation playback controls

`createAnimationController(pose)` controls an existing packed animation player.
It is an explicit, opt-in API for imported TRS and morph tracks, not a drop-in
Three.js `AnimationMixer` or arbitrary `PropertyBinding` implementation.
The controller borrows its player; disposing the controller does not dispose
that player. All times below are seconds.

## Speed ramps and smooth stops

```js
const action = controller.createAction(0).play();
action.warp(1, 2, 0.5);  // Ramp from 1x to 2x over half a second.
action.warpTo(0.5, 1);   // Replace the ramp, starting at the current speed.
action.halt(0.25);       // Ramp to zero, retaining the action's pose and weight.
```

`warp(from, to, seconds)` changes the speed immediately to `from`, then linearly
ramps toward `to`. `warpTo(to, seconds)` starts at the current speed. A zero
ramp duration sets the target speed immediately. Playback integrates the ramp's
area rather than multiplying the whole frame by its final speed. Direction
reversals are split so forward and reverse loops, or a finish before the turn,
are not lost. Calculations use JavaScript f64 arithmetic, not fixed-point time.

`stopWarping()` cancels at the current speed. `setTimeScale(value)` also cancels
a ramp. `halt(seconds)` does not deactivate the action or change its weight;
resume with `setTimeScale` or another warp. `timeScale` reports the instantaneous
speed, `warping` reports an unfinished ramp, and `effectiveTimeScale` is zero
for inactive, paused, finished or pending-start actions.

Pause freezes the playhead, **not** fades or speed ramps. Natural completion
stops the playhead but lets an outstanding ramp reach its target. Explicit
`stop()` and `reset()` cancel ramps without restoring the previous speed or
weight. A fade configured with `stopWhenDone` stops playback at its fade boundary,
even when one update spans far beyond that boundary.

## Scheduled starts

```js
const action = controller.createAction(0)
  .startAt(controller.time + 1)
  .play()
  .warpTo(2, 0.5);
```

`startAt(when)` accepts a finite, nonnegative absolute controller time. It does
not implicitly play the action. Before the deadline an armed action contributes
no pose, and its playhead, fade and ramp remain frozen. An update crossing the
deadline advances only the elapsed post-start interval. A deadline already in
the past catches up on the next update, including `update(0)`.

`scheduled` and `startTime` expose the pending state. `cancelStart()` cancels the
deadline without catch-up; a playing action resumes from its current phase.
`stop()` and `reset()` cancel the deadline. `seek()` preserves it. Assigning a new
deadline before replaying a finished action preserves that deadline through
`play()`. Pause does not postpone a scheduled start: the action starts on time,
its fades and ramps run, and only its playhead stays paused.

## Duration control and one-shot synchronization

`action.duration` is the imported clip duration, not its current wall-time
traversal duration. `setDuration(seconds)` sets the speed needed to traverse the
clip in the requested positive number of seconds. It retains the current phase
and direction, cancels a speed ramp, and does not activate the action. A stopped
clock with speed zero resumes at a positive speed. The current ping-pong leg
remains unchanged.

`target.syncWith(source)` copies normalized phase and normalized local speed
from another live action in the same controller. For a two-second source and a
four-second target, source time `0.5` maps to target time `1`, and source speed
`1` maps to target speed `2`. Ping-pong direction is preserved; different loop
modes receive the same local direction at the instant of synchronization and
subsequently follow their own loop behavior.

Synchronization is a snapshot, not a permanent link. It cancels the target's
speed ramp, resets its traversal count and clears its finished state. It does
not copy activation, pause, schedule, fade, weight, mask, blend mode or loop
configuration. Call `play()` separately when needed. Both clip durations must
be positive. Unrepresentable nonzero scaled rates are rejected instead of
silently stopping playback. Invalid controls leave both actions unchanged.

## Phase-aligned, duration-aware crossfades

```js
const walk = controller.createAction(walkClip, { loop: "repeat" }).play();
const run = controller.createAction(runClip, { loop: "repeat" });
controller.crossFade(walk, run, 0.35, { sync: true, warp: true });
```

The original three-argument `crossFade(from, to, seconds)` remains a weight-only
transition. `sync: true` aligns the target's normalized phase and ping-pong leg
and resets its traversal budget. Alone, it does not change the target's speed.
`warp: true` creates paired speed ramps, beginning at the source's normalized
local rate and ending at the target's currently configured rate. Combining
both options aligns phase and rate for unequal-duration clips. Warp-only
transitions preserve phase offsets and do not guarantee coincident ping-pong
reflections. A zero-duration transition is immediate, without a residual ramp.

Advanced transitions require positive clip durations and matching loop modes.
Warped transitions additionally require a playing, unpaused source. These
admission errors are checked before either action changes. A target starts
immediately and loses a pending start deadline. An inactive target begins at
zero weight; an already active target retains its current weight, so interrupted
fades continue from the current weights. The outgoing action stops at the end
of the fade. Finite traversal budgets still apply; independent later controls
on either action can break phase/rate alignment.

## Updates, events and publication

Call `controller.update(nonnegativeDelta)` explicitly. There are no timers or
listener callbacks. The controller evaluates all contributing layers through
one `pose.blend()` call, retaining the player's output-array identities.

Read `controller.events` after each successful update. It is a frozen snapshot,
not an accumulating queue. A scheduled action emits
`{ type: "started", action, time: deadline }` once, before its own loop/finish
records. Actions retain creation-order event grouping, not a global sort by
start timestamp. Consecutive same-direction loops are aggregated into a count;
a reversal can produce two directional loop records. Event storage is bounded
by the action count, not by how many traversals a large update crosses.

Invalid input, reentrant mutation, an overflowing clock, or failed pose evaluation
cannot publish half an update: action clocks, fades, ramps, scheduled starts,
events and the player pose remain at the last successfully committed update.
Configuration calls intentionally take effect immediately on control state;
the rendered pose changes on the next successful update.

## Regression tests

From the repository root:

```sh
node --test tools/ingest/animation_controller*.test.mjs
```

The suites use the real CPU pose player, including a scheduled skin/morph
publication check. They cover existing playback, warps, scheduled starts,
synchronization, interrupted crossfades, rollback, numerical edge cases and
large-versus-subdivided updates. They do not establish Three.js mixer parity,
browser pixel equivalence, GPU behavior or an acceleration claim.
