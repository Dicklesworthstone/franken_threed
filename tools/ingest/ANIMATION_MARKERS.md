# Timed animation cues

Packed animation actions can emit named clip-time markers for footsteps, audio,
particles, gameplay handoffs or application state transitions. Markers follow the
**existing action clock**, including reverse playback, repeat/ping-pong loops,
rate warps, scheduled starts and crossfades. They do not add a scheduler, sample
another pose or call application code during pose evaluation.

```js
const walk = model.controller.createAction(walkClip, {
  loop: 'repeat',
  markers: [
    {name: 'foot-left', time: 0.18},
    {name: 'foot-right', time: 0.68},
  ],
}).play();

// In the application's existing frame loop:
model.update(deltaSeconds); // keep the normal model update/upload path
for (const event of model.controller.markerEvents) {
  // Dispatch application effects here, AFTER the successful update.
  // event.action identifies walk, even when several actions share walkClip.
  dispatchAnimationCue(event);
}
```

For a standalone CPU player, call `controller.update(deltaSeconds)` instead of
`model.update`. Do not call both for the same frame. Marker events are generated
synchronously by that update; they are not audio-device timestamps or wall-clock
alarms. Rendering/GPU upload after a CPU controller commit is a separate boundary.
A later GPU failure does not roll back the committed action clocks or cues.

## Track configuration and event records

`createAction(clip, {markers})` accepts up to 4,096 `{name,time}` records. `time`
is finite local clip time in seconds, in the inclusive range `[0, action.duration]`.
Names contain 1..256 characters. There is no arbitrary callback or payload field;
the application can associate its own data with the name and action handle.
Duplicate names and duplicate times are allowed and represent distinct cues.

Records are validated and copied before creating the action. Unsorted input is
accepted; the caller's array is not sorted or retained. `action.markers` is a
frozen time-sorted snapshot of frozen records. Cues at the same time retain their
original declaration order in **both** playback directions.

`action.setMarkers(records)` replaces the track transactionally and returns the
action. An invalid later record leaves the old track intact. Replacement neither
moves the playhead nor emits cues at its current position. Passing `[]` disables
markers. Previously retained metadata and event snapshots remain usable.

Each entry in `controller.markerEvents` is frozen:

```js
{
  type: 'marker',
  action: walk,            // the existing live action handle
  name: 'foot-left',
  time: 0.18,              // authored local clip time, not controller wall time
  direction: 1,           // +1 forward or -1 reverse LOCAL clip travel
  traversal: 3,           // zero-based leg index before this arrival
}
```

Traversal numbering follows the action's `completedTraversals` convention: a
ping-pong leg is one traversal, not a round trip. Reset, seek and loop
reconfiguration retain their existing effects on that counter. Event names,
times, directions and traversal numbers are snapshots; `event.action` remains
the live action object, just like existing loop/finish events.

`markerEvents` is a **separate stream** from `controller.events`. Existing
`started`, aggregated `loop`, and `finished` records keep their original shape,
ordering and aggregation behavior. This prevents marker insertion from changing
legacy loop consumers. Within `markerEvents`, actions appear in creation order;
each action's cues appear in actual traversal order, including both sides of a
reversing warp. Events across different actions are not globally sorted by wall
time, and the two streams are not a single interleaved timestamped timeline.

## Arrival, wrap and reversal rules

A forward movement reports markers in `(oldTime, newTime]`; reverse movement
reports markers in `[newTime, oldTime)`. Departure is excluded and arrival is
included. Reaching a marker exactly at a frame boundary therefore reports it
once, not again when the next frame departs from that point. No epsilon expands
these intervals: comparisons use the action clock's computed local endpoints.
Existing floating-point timing behavior is unchanged.

For a repeating clip, arrival at the final endpoint and entry at the wrapped
endpoint are separate cue locations. A forward wrap can report a marker at
`duration` on the outgoing traversal, followed by a marker at `0` on the next
traversal. Reverse wraps behave symmetrically. A finite action's **terminal**
arrival does not report a wrap into a traversal that never starts.

A ping-pong reflection reports the endpoint only on arrival. Its departure on
the opposite leg does not duplicate that endpoint marker. Event `direction`
reflects actual local travel, not merely the sign of `timeScale`; the two differ
on the returning leg. A speed warp that reverses within one update reports the
outward and returning paths separately, even when its net displacement is zero.
A natural finish before the turn does not produce cues from a nonexistent return.

Playing, resetting, seeking, syncing phase or replacing markers does not
synthesize a traversal. In particular, a marker at time zero is not an automatic
"play" event. Scheduled starts still use the existing `started` event. A
zero-duration clip is the explicit exception: its time-zero markers fire once
with its first natural completion, on a nonzero playback advance, and can fire
again after an explicit replay.

Paused, stopped and zero-speed actions emit no movement cues. A clamped finished
action contributes its held pose but produces no repeated markers. Fading to a
stop reports only cues reached before the fade deadline, not cues from the whole
host delta. Zero weight does **not** stop an otherwise running clock, so its cues
still report just as its loop events do. During a crossfade both running actions
can emit cues. Effect suppression, foot-contact choice and audio gain remain
application decisions, not implicit marker-weight heuristics.

Before a scheduled deadline, markers remain inactive along with the playhead,
fade and warp. Late starts report the intervening traversals using the existing
catch-up rules. Even `update(0)` can report catch-up cues for an already-past
deadline; it does not report cues when there is no actual action-clock advance.

## Bounded updates and rollback

The default limit is **4,096 marker events per controller update**, aggregated
across every action and every segment of a reversing warp. Configure it at
construction, or on an existing model/controller:

```js
const controller = createAnimationController(pose, {maxMarkerEvents: 8192});
// A model already owns a controller; configure that one instead of replacing it.
model.controller.setMarkerEventLimit(8192);
console.log(model.controller.maxMarkerEvents);
```

The limit must be an integer in `[1,65536]`. Changing it affects subsequent
updates, not previously published snapshots. The traversal engine computes the
required number of events before enumerating full loops. An enormous seek-like
update cannot silently allocate or walk through an unbounded number of markers.
Marker-free actions retain the existing constant-size loop summaries.

An overflow throws `AnimationMarkerError` with code `ANIMATION_MARKER_LIMIT`.
Nothing is silently dropped, collapsed or deferred to a hidden queue. The update
leaves all committed action clocks, schedules, fades, warps, pose bytes/versions,
legacy events and marker events unchanged. Retry with smaller update steps,
raise the limit, or disable unneeded tracks explicitly. A pose-evaluation error
(such as a singular skinned mesh) likewise publishes none of its pending cues.
All pending event storage is private until the real pose blend succeeds.

`markerEvents` represents the last **successful** update, not an accumulating
queue. Consume it once after each successful update. Do not redispatch the prior
snapshot after a failed update. Retained old snapshots are caller-owned; an
application retaining every historical frame is outside the per-update budget.
Controller disposal clears its published/pending marker lists without disposing
the borrowed pose. Existing reentrancy and disposed-action guards apply to track
and limit changes. External effects of caller getters are not a rollback sandbox.

## Live clips and storage boundaries

Markers also work with clips installed later through `pose.addClips`:

```js
const [clip] = model.pose.addClips([take.clip]);
const action = model.controller.createAction(clip, {
  markers: [{name: 'handoff', time: 0.5}],
}).play();
```

Times must fit the installed clip's duration. Existing actions and their marker
tracks keep their identities when other clips are appended. Markers are explicit
**action metadata**, not pose channels or glTF animation-pointer targets. They
are not automatically imported from source extras, captured by the pose
recorder, included in `pose.snapshotClips()`, or written into exported GLBs.
Save `action.markers` separately with the corresponding motion/application data
when persistence is required. Copying the same metadata to another action still
validates it against that destination clip's duration.

## Validation

```sh
node --test tools/ingest/animation_markers.test.mjs \
  tools/ingest/animation_controller_markers.test.mjs \
  tools/ingest/animation_controller.test.mjs
```

The traversal suite includes an independent stepwise walking oracle over 12,000
deterministic randomized paths, exact budget boundaries, endpoints, ties and
huge-jump refusal. Controller integration uses the production sampler/blender
with morphs and distinct mesh-local skin palettes, including late-installed
clips, scheduled/reversing playback, synchronized warped crossfades, partitioned
updates, singular-pose/detached-output failures and successful retries. These
are CPU semantics tests, not rendered-image, native GPU, audio-timing, Three.js
mixer-equivalence or performance benchmarks.
