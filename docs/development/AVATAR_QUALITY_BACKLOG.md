# Avatar quality — known problems

**Opened:** 2026-08-13, after the first real look at the landmark-warp engine on
the owner's own portrait and camera.

Every item here was **observed**, not predicted. Each records what was seen, why
it happens, and what would actually fix it — because for several of these the
obvious fix is not the real one, and knowing that in advance is the point of
writing them down.

The engine works and holds 30 fps. This is the list of what stands between
"working" and the quality bar: *believable on a video call*.

---

## 1. The forehead is not covered, and the skin tone steps at the edge — FIXED 2026-08-13

**Seen:** a visible difference in skin colour where the avatar stops and the
real forehead begins.

**Why:** the mask deliberately stops just above the brows. That was done to
avoid a different artifact — MediaPipe's face outline runs along the hairline,
and no two people's hairlines match, so covering up to it left a hard
horizontal edge. Trading the hairline seam for a mid-forehead seam is not a
fix; it moves it.

**What would actually fix it:** the seam exists because the two images have
different colour *gradients*, not just different means. Matching the mean —
which the engine does now — cannot remove that. Gradient-domain blending
(`cv2.seamlessClone`, Poisson) solves it properly by construction: the edge is
made to match by definition, whatever the two sources look like.

The cost has to be measured before committing: seamless cloning at this face
size plausibly costs 10–30 ms, against 15.4 ms for the entire current warp. If
it is too slow, the fallback is a much wider feather at the top plus local
colour matching in bands rather than one global mean.

**Owner's requirement: the forehead must be covered.** Both routes have to
satisfy that.

**Fixed, and not the way this predicted.** `cv2.seamlessClone` was measured
first, as planned: 11.3 ms for a 200 px face, 21.6 at 300, 29.0 at 400. Against
a 33 ms budget with 15 already spent, that is not affordable at the size a face
actually occupies.

What was affordable is the same idea done cheaply. The difference between scene
and avatar is measured at 1/8 scale, blurred, and added back — low frequencies
carry lighting, high frequencies carry identity, so correcting only the low ones
relights the avatar without touching its features, and the edge matches because
the field is continuous. The blur is weighted by the mask so the room behind the
head never contributes.

That removed the tone step, which then allowed the mask to be extended back over
the whole forehead as the owner required, stopping just short of the hairline —
hairlines are structure rather than tone, and no blending fixes structure.

Cost went *down*: 15.4 ms to 12.65 ms mean, because the mean-matching it
replaced used boolean indexing over the full frame.

---

## 2. No teeth when smiling

**Seen:** smiling opens the avatar's mouth shape, but the mouth interior is
wrong — there are no teeth.

**Why:** this is a hard limit of the method, not a bug. The engine reshapes
pixels that exist in the portrait. The portrait's mouth is closed, so there are
no teeth pixels anywhere in the source; stretching the lips apart can only
reveal more of the lips.

**What would actually fix it:** a second photograph of the same face with the
mouth open, used as the source for the mouth region when the tracked jaw is
open. This is the owner's own suggestion and it is the correct one.

---

## 3. Eyes do not close when blinking

**Seen:** blinking closes the user's eyes; the avatar's stay open.

**Why:** two causes stacked. The mesh carries only six points per eye, which is
too few to describe an eyelid closing. And, as with the teeth, the portrait has
open eyes — there are no closed-eyelid pixels to move into place.

**What would actually fix it:** more eye landmarks (MediaPipe provides the full
eyelid contour, it is simply not in the current subset), plus either a
closed-eye source photograph or a synthesised lid. The landmark part is cheap
and worth doing first, to see how far it gets on its own.

---

## 4. Eyes grow when the brows are raised or the eyes opened wide

**Seen:** raising the brows or opening the eyes wide makes the avatar's eyes
grow larger than they should.

**Why:** the warp maps the avatar's eye region onto the user's eye *geometry*
directly. A wider eye opening therefore stretches the avatar's eye rather than
opening it. The avatar inherits the user's proportions where it should be
inheriting only their expression.

**What would actually fix it:** drive the eye and brow regions from the tracked
**blendshapes** — which are already measured every frame, and are exactly
"how open is this eye" as a number — with clamping, rather than from raw
landmark positions. The identity's proportions then stay the avatar's while the
motion comes from the user.

---

## 5. Head rotation past roughly 15 degrees breaks the alignment

**Seen:** turning the head about 15° or more and the avatar stops following
convincingly.

**Why:** the warp is two-dimensional. A rotated head reveals parts of the face
the portrait never photographed, and a flat mesh cannot invent them — it
stretches the pixels it has across a surface that has turned away.

**What would actually fix it:** source photographs at several yaw angles, chosen
or blended by the yaw already measured every frame (B003 verified that
measurement). This is again the owner's suggestion, and it is the standard
answer for image-based avatars.

The alternative is a genuinely 3-D-aware method — which is what LivePortrait
is, and it remains blocked on its weight licence.

---

## The common thread: more than one photograph

Three of the five (teeth, eyes, rotation) come down to the same limit: **the
engine can only show what the source photograph contains.** The owner's
suggestion — several versions of the same face — addresses all three, and is
cheap compared with any alternative.

What that needs from the avatar format:

- the manifest already supports multiple assets with roles, so this is an
  additive change, not a redesign;
- a rule for choosing between them per frame (yaw for rotation, jaw-open for
  the mouth, eye-blink for the eyes);
- blending between sources so switching is not visible as a jump.

**Photographs to request, all of the same person, same lighting, same distance:**

| Purpose | Photograph |
|---|---|
| Base | Front, neutral, mouth closed, eyes open (**already have**) |
| Teeth | Front, smiling with teeth visible |
| Blink | Front, eyes closed |
| Rotation | Turned ~20–30° left; the same to the right |

---

## The 3-D question

Raised by the owner: would building a 3-D model of the face work better?

For robustness, yes — and it is already this project's documented backup engine
(`docs/OPEN_SOURCE_EVALUATION.md` B4). Real geometry solves rotation, eyelids,
and jaw opening by construction rather than by collecting more photographs,
because the shape exists whether or not a camera ever saw that angle.

**But there is something cheaper to do first, and it is already paid for.**
MediaPipe returns 478 landmarks with a **z** coordinate and a 4×4 head-pose
matrix, both verified in B003. The warp engine currently uses **x and y only**
and discards the rest. Rotating the avatar's landmark mesh by the measured head
pose before projecting it — a 3-D-aware warp of the same photograph — should
recover a good part of item 5 with no new dependency, no new model, no new
photographs, and no new licence question. That is the first thing to try.

What it still cannot do is invent the far cheek when the head turns. Nothing
that reshapes one photograph can.

Beyond that, in rough order of what they cost:

| Approach | Solves | Costs |
|---|---|---|
| 3-D-aware warp of the current photo | rotation alignment | a day; nothing else |
| Morphable face model (3DMM) fitted to the photo | rotation, lids, jaw, teeth | a real component, and **licence checks** — the well-known face models are research-only or need a separate commercial agreement, the same trap InsightFace already set |
| Per-identity neural head avatar | best quality of all | training per avatar from *video*, which does not exist for a person who was generated |
| LivePortrait | rotation and expression from one photo | unresolved weight licence; already the documented primary |

The honest limit of every 3-D route: a head built from a single photograph
looks like a game character rather than a person, and hair is the hardest part
of it. Robustness and photorealism pull in opposite directions here, which is
why the evaluation kept both a neural and a 3-D path alive.

---

## Order of work

Cheapest and most visible first; each is independently useful.

1. **Forehead and the seam** (item 1). It is the most visible flaw and affects
   every frame. Measure seamless cloning first, since it may make the mask
   question disappear entirely.
1b. **Use the depth already being measured** — the 3-D-aware warp described
   above. Cheapest attack on item 5, and it needs nothing from anyone.
2. **Eye landmarks** (item 3, first half). Cheap, no new photographs needed.
3. **Blendshape-driven eyes and brows** (item 4). Removes a distortion rather
   than adding a feature.
4. **Multi-photo sources** (items 2, 3, 5). The largest piece, and it needs the
   owner to shoot the photographs first.

---

## Scheduling note

The 14-day plan has **day 5 as the virtual camera** — the part that makes any of
this usable in Zoom or Meet at all. Today's quality work is not on that plan.

Both matter, and the order is the owner's call. Worth knowing when deciding:
until the virtual camera exists, none of this quality work can be seen by anyone
outside this application.

---

# Update — end of day 4

This file was **overwritten** by a copy of `PROJECT_STATUS.md` in commit
`5e7a788`, and everything above is recovered from `4ec86f8`. Nothing was lost
permanently; it was invisible for a day, which is its own lesson about a
document nobody reads until they need it.

What has happened to the five items since they were written:

| # | Problem | State |
|---|---|---|
| 1 | Forehead seam | **Fixed** — low-frequency relighting, see above |
| 2 | No teeth when smiling | **Engine done, waiting on a photograph** |
| 3 | Eyes do not close | **Engine done, waiting on a photograph** |
| 4 | Eyes grow when opened wide | **Engine done** — expression drives the blend, not the mesh |
| 5 | Rotation past ~15° | **Engine done, waiting on photographs** |

The common thread the original diagnosis identified — *the engine can only show
what the source photograph contains* — was addressed by building the
multi-source path (`b4b7d7c`). The source set grew from 6 roles to **22**: mouth
shapes, gaze in four directions, wide eyes, squint, brow positions, far turns
and head tilts. Each is selected and blended from a blendshape the tracker
already measures or from the head pose B003 verified — no thresholds, no state
machine — and every role is optional, so an avatar uses whatever photographs it
has.

Which means items 2, 3 and 5 are no longer engineering problems. They are
**supply** problems: the photographs decide the quality now, and the prompts for
all twenty-two are written out in `AVATAR_PROMPTS.txt`.

Cost, at a face the size one occupies at desk distance: 13.8 ms mean, 17.7 ms
p95 in ordinary use; 15.2 / 19.2 in the worst case where the expression changes
every frame.

## 6. Periodic flicker — three causes fixed, a fourth not ruled out

Reported by the owner after the multi-source work. Three independent causes were
addressed in `85cc6d6` because none could be excluded without watching it:

- **The camera was passed through whenever tracking dropped**, showing the
  owner's real face. There is no length of tracking loss at which that is the
  right answer — the product exists to keep that face off the call. The avatar
  now holds its last position indefinitely.
- **The relighting chased the camera's exposure.** A webcam hunting exposure
  brightens and dims the whole scene, and a correction recomputed each frame
  followed it instantly, so the face pulsed against a merely drifting
  background. The field now carries 80% of the previous frame forward.
- **Landmark smoothing was off** at 0.35 in configuration but zero in effect
  since day 3. Jitter moves the mask edge a pixel or two per frame, which reads
  as a shimmer.

**Still open: the owner is verifying whether it is gone.** If it persists it is
a fourth cause, and the three above are then known-good rather than suspected.

## 8. The avatar stopped looking like its own portrait — FIXED 2026-08-14

**Seen:** the owner, on the first real look at the 20-source avatar: *"the
quality is catastrophic — compared to the photograph it looks like 144p. The
eyes are completely different and do not look real."*

**Why, measured on a motionless subject.** A resting face was scoring **eleven
simultaneous expressions**:

| role | weight | % of frames active |
|---|---|---|
| `source_turn_left` | 0.95 | 100% |
| `source_tilt_down` | 0.95 | 100% |
| `source_squint` | 0.95 | 100% |
| `source_look_down` | 0.90 | 100% |
| `source_mouth_pressed` | 0.59 | 100% |
| `source_brows_up` | 0.51 | 100% |
| `source_look_left` | 0.50 | 100% |
| `source_brows_down` | 0.40 | 100% |
| `source_look_up` | 0.36 | 100% |

Two independent mistakes, and each is worth stating plainly because both look
reasonable in the code:

1. **Expression was measured absolutely, not against rest.** MediaPipe's
   blendshapes are not zero on a relaxed face, and its head pose is not zero
   unless the camera is exactly on axis — which no webcam above a monitor is.
   The person was sitting still and the engine read "squinting hard, looking
   down and left, turned 19°, tilted 18°".
2. **Opposing photographs were blended together.** Look-left and look-right
   were both active on **98%** of frames, look-up and look-down on **100%**,
   brows-up and brows-down on **100%**. The engine averaged four gaze
   directions into eyes that pointed nowhere. *That is the eyes complaint,
   exactly.*

The result was a weighted average of eleven photographs in which the base
portrait's share had been renormalised to nearly zero. Sharpness of the blended
source measured **17 against the portrait's 62** — a literal multiple exposure,
which is what reads as "144p".

**Fixed.**

- `_Neutral` learns the resting face and every expression is measured above it.
  Blendshapes get a floor that follows a new low in about a quarter of a second
  and creeps up over twenty, so a relaxed face is learned quickly but a held
  expression cannot become the new normal. Head pose gets a slow mean. Both are
  seeded from the first frame, so frame one is already correct.
- Opposing photographs are one signed quantity: the stronger side wins and the
  other is zero by construction.

**Measured after**, driven with the same readings taken off the real camera:

| | before | after |
|---|---|---|
| Sources active, at rest | 11 | **0** |
| Blended source | sharpness 17 | **byte-identical to the portrait** (62) |
| Sources active, smiling and turned 22° | ~12 | **3** |
| Blend cost, at rest | 6.75 ms | **0.02 ms** |
| Blend cost, in motion | 6.75 ms | **2.04 ms** |

A still face now gets its own photograph back, unmodified. That also removes
most of item 7 — the blend is no longer rebuilt on nearly every frame because
at rest there is nothing to rebuild.

**Still to confirm by eye:** whether this also removes the fourth flicker
cause. The weights were swinging violently frame to frame — `source_look_left`
ranged 0.05 to 0.92 on a still subject — which changed the face's appearance
every frame. That is a flicker by any definition, and it appeared when
multi-source landed. Plausible, not proven; the owner has to look.

## 9. Every photograph changes the whole face — opened day 6

**The one that decides whether more photographs help or hurt**, so it is the
next piece of work.

`_blend_sources` produces **one blended source image for the entire face**. A
smile photograph does not change only the mouth; it changes the forehead, the
ears, the jaw, the hair. So does every other. Twelve active sources meant twelve
photographs averaged across every pixel — a multiple exposure — which is what
the owner saw as "144p" and what the sharpness measurement showed as 62 → 17.

Day 6's neutral-baseline fix removed the *unnecessary* sources: a resting face
now uses none. But the moment the user genuinely smiles, turns and raises an
eyebrow at once — which is most of talking — three or four photographs are
legitimately active and they still fight over the whole face.

**The owner asked whether 100–150 photographs would solve the quality problem.
With this architecture they would make it worse**, and that is measured rather
than argued: going from 6 to 20 took animation from 13.8 ms to 22.1 ms and
sharpness from 62 to 17. More sources means more of them active at once, and
every one of them repaints the entire face.

There is also a ceiling no number of photographs reaches. Expressions
**combine**: smiling *and* looking left *and* brows raised *and* head turned.
Photographs cover axes; a face occupies their product. Four axes at five steps
is 625 combinations, six axes is over 15,000. Twenty-two photographs cover each
axis once. A hundred and fifty cover each axis more finely and still cannot
cover a combination.

**The fix: each photograph contributes only to its own region.** Mouth
photographs to the mouth, eye photographs to the eyes, brow photographs to the
brows, turn photographs to the whole face because that is genuinely what they
describe. Regions come from landmarks the tracker already provides, feathered so
no boundary shows.

Three things follow, and the third is the point:

1. **Ghosting stops**, because each region is drawn from one or two photographs
   instead of twelve.
2. **Cost falls**, because a mouth blend touches the mouth's pixels rather than
   the whole face — and the mouth is a small fraction of it.
3. **More photographs finally help.** Combinations stop being a problem, because
   the mouth axis and the eye axis no longer interfere with each other: they are
   composed rather than averaged. *This* is the change that has to land before
   generating more photographs is worth anyone's time.

Worth doing alongside it: the source is warped down by about **3×** (773 px face
to 255 px on screen) with `INTER_LINEAR`, which samples 2×2 and discards the
rest. Downscaling the blended source with `INTER_AREA` before the mesh warp is
the correct filter for a reduction, and it makes the warp cheaper at the same
time.

## 7. The source blend outgrew its cache — opened day 5

The owner supplied the full photograph set on day 5 and 20 sources are wired in.
That is the outcome items 2, 3 and 5 were waiting for, and it cost the frame
budget.

**Seen:** the animation stage costs 22.1 ms with 20 sources against 13.8 ms with
six. Headroom fell from ~11.5 ms to ~2.5 ms. Throughput still holds at 29.80 fps
with nothing dropped, so nothing looks wrong yet — which is exactly why it is
worth writing down now rather than after the first call that stutters.

**Why**, measured over 596 real frames rather than reasoned about:

- **12.1 sources are active per frame** (min 8, max 15). Each costs one
  full-image pass.
- **The blend cache hits 2.3% of the time.** It is keyed on the weights
  quantised to 0.02. Six sources changed slowly enough that the same key came
  up again and again; twelve independent dials means at least one crosses a
  step nearly every frame, so 97.7% of frames rebuild the whole blend.
- The blend alone is **6.75 ms mean / 8.04 ms p95**, where it used to be
  effectively free.

**What will not fix it, measured before proposing it.** Keeping only the
strongest N sources is the obvious move and it is a bad trade: the weight beyond
the top six averages **0.59** of the total, so it is genuinely spread across
photographs rather than concentrated. Keeping 6 saves 2.6 ms and moves pixels by
up to 29 levels out of 255; keeping 10 is invisible but saves only 0.6 ms.

**What probably will**, in the order worth trying:

1. **Smooth the weights over time** the way landmarks are already smoothed. The
   blendshapes jitter frame to frame, and that jitter is what breaks the cache.
   Smoothing should raise the hit rate *and* steady the picture — the same fix
   for a performance problem and a quality one.
2. **Coarsen the quantum** for roles whose weight is small, since a 0.02 step
   matters at weight 0.6 and not at 0.05.
3. **Fuse the blend into one pass.** Twelve `addWeighted` calls each read and
   write the whole accumulator; one weighted sum over twelve sources does not.
   Roughly 22 million pixel-operations in 6.75 ms is well short of what this
   machine can do, so there is real room here.

None of these changes what the avatar can express. They change what it costs to
express it.

---

# Update — day 5

The virtual camera exists (B004). Everything in this file is now visible in
Meet, Zoom, Discord and OBS rather than only in our own preview window, which
is the condition the original scheduling note said all of this quality work was
waiting on.
