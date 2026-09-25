# Install motion into an existing animation player

`pose.addClips(clips)` installs decoded animation clips into a live packed pose
player. Recorded, baked or explicitly retargeted motion can be sampled on that
same player without reconstructing the skeleton, world matrices, skin palettes
or mesh consumers. It returns a frozen array of the new numeric clip indices.

```js
// take.clip is the result of createAnimationRecorder(...).finish().
// Its target node indices must already match this pose's rig.
const [recordedClip] = pose.addClips([take.clip]);

// Installation itself does not change the current frame or advance any clock.
pose.sample(0.5, {clip: recordedClip});
// Existing consumers still borrow the same public arrays and pose object.
```

A clip uses the existing decoded channel contract: `{name, channels}`, where each
channel has `node`, `path`, `times`, `values`, and optional `interpolation`.
Translation, rotation, scale and morph-weight channels support the existing
LINEAR, STEP and CUBICSPLINE sampling rules. `addClips` neither quantizes keys nor
resamples them; numerical evaluation uses the same implementation as initial
imports. Recorded/baked input still has its original sampling approximation.

## Stable identities and explicit publication

Installation is append-only. Existing clip IDs, names, durations, sampler cursors
and action bindings are not replaced or renumbered. Duplicate names are allowed;
use the returned indices rather than assuming a name is unique. An unnamed clip
uses `animation_<index>`, with its final index in this player's table.

`pose.clips` returns a frozen metadata snapshot of `{name, duration}` entries.
A successful nonempty batch replaces this snapshot; previously retained snapshots
remain unchanged, and the old entries retain identity in the new snapshot.
`pose.clipVersion` starts at zero and increments once per successful nonempty
installation, independently of `pose.version`. An empty batch does not change
either version or metadata identity. Reading metadata does not expose keyframes.

Installing clips does not sample or reset anything. The current local edits,
external root placement, sampled clip/time/mode, pose version and all published
array identities and bytes remain unchanged. New bindings and sufficiently wide
morph-blend scratch are prepared before publication. Existing players with no
authored clips can receive their first animation through the same operation.

Input channels and keyframe arrays are copied. Editing, detaching or discarding
the caller's input after success cannot change the installed motion. The source
definition used to create the player is not modified, nor are other players made
from that definition. This API does not mutate an original GLB/source snapshot;
retain the input clips when exporting them through the asset exporter.

## All-or-nothing batches and lifetime bounds

Construction and late installation use the same clip validator. The complete
batch must have valid target indices, channel widths, unique bindings within
each clip, ordered nonnegative finite times, finite values, admitted interpolation
and valid quaternion keys. Matrix nodes may receive morph-weight channels but
not TRS animation. A later failure in a batch publishes none of its earlier clips
and consumes no IDs, component budget or metadata revision.

The existing maximum of 4096 clips and 16,777,216 keyframe numeric components
applies to the player's **entire** table, including initial imports and every
successful batch. Components include each channel's time array and all value or
cubic-tangent components. Per-channel key and per-clip channel limits still apply.
An empty-channel clip is legal. There is no removal/eviction or replacement API;
keyframe storage remains owned by the player for its lifetime.

Pose operations and installation cannot reenter each other through input
callbacks. Exceptions propagate and leave owned registry state unchanged.
Detached published outputs are rejected before publication. Caller code, getters
and iterators are not a sandbox: their external effects cannot be rolled back.
The call is synchronous; there is no device operation, asynchronous loader,
background clock or implicit renderer initialization.

Successful admission is not a promise that every sample is evaluable. For
example, animating a skinned mesh to zero scale can make its world matrix singular.
The existing transactional sampling/blending error rules still apply, and earlier
clips remain available after an unsuccessful evaluation.

## Boundaries and validation

Tracks must already address this player's node order and morph widths. The API
does not identify another asset's skeleton by matching numeric indices, perform
retargeting, map synthetic instance nodes back to source nodes, or import Three.js
PropertyBinding tracks. Source model/asset export and live installation are
separate operations. GPU consumers need their usual update/upload after the
caller actually samples or blends a new pose, not after a registry-only change.

`node --test tools/ingest/animation_runtime_clips.test.mjs` exercises the real
runtime, including late first clips, all three interpolation modes, normal and
additive blending, masks, wider morph channels, shared-skeleton mesh-local
palettes, cumulative limits and failed-batch recovery. These CPU checks do not
claim native GPU pixels, Three.js mixer parity, full-repository validation or
measured speed improvements.
