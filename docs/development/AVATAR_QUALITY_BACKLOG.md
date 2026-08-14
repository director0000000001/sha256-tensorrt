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

## 15. The wedge above a blinking eye — FIXED 2026-08-15

**Seen:** on every blink, a hard-edged light triangle between the brow and the
closed lid. Found by replaying the owner's recorded clip, which is what that
harness is for.

**Where it is not.** Rendering the blended source at the same frame shows a
clean, correct half-closed eye. The blend is right. The defect is created by the
mesh warp that follows it — which is why three plausible fixes all failed:

| tried | result |
|---|---|
| Iris collapsing to a degenerate point | no change |
| `_BROW_FOLLOW` lowered to match `_EYE_FOLLOW` | no change |
| Dropping triangles *inside* the eye opening, as the mouth does | no change |

**Where it is.** Drawing the target mesh over a blinking frame answered it in
one look, after three guesses had not: the triangulation anchors **long, thin
triangles on the eye landmarks and runs them out across the cheek and brow**. A
fan of them converges on each inner corner. When the lid closes, that anchor
point travels, every triangle in the fan sweeps with it, and the sweep is the
wedge.

Six landmarks per eye is the cause, and not for the reason recorded in item 3.
The problem is not that six points cannot describe a lid — the photographs carry
the lid now. It is that six points leave the eye **connected to the rest of the
face**, with nothing local for the triangulation to attach to.

**The fix, and why it is not a constant.** The eye needs the full eyelid ring —
sixteen points per eye, which MediaPipe already provides and this mesh has never
used. Then the long triangles terminate on the ring instead of reaching through
the eye.

Tried, and it is the right direction: the wedge shrinks markedly. It is not
finished — some of it survives, and changing the mesh invalidates ten tests that
encode the six-point layout, including the whole iris-placement group. That is a
piece of work with its own verification, not a tweak, so it is written down here
rather than half-landed.

**Method note, which is the transferable part.** Four attempts at this cost
about an hour; the mesh drawing cost five minutes and ended it. When a defect
has survived two hypotheses, stop proposing a third and render the thing that is
actually being blamed.

### Fixed, and the ring was only half of it

The ring landed as described — sixteen points per eye, exactly ten tests
rewritten — and the wedge shrank and **did not go away**. What remained was a
second, smaller fan living *inside* the eye, and it had a cause of its own.

**MediaPipe reports the whole iris circle, including the part behind the
eyelid.** That is not a fault in the tracker: it fits a circle to the visible
arc, and the circle is a circle whether or not a lid is over it. But the mesh
warps *pixels*, and the pixels above a shut lid are eyelid rather than iris.
Measured on the avatar's own photographs, in pixels of its own frame:

| photograph | lid height | iris height | iris outside the lid |
|---|---|---|---|
| base portrait | 29.0 | 37.3 | 8.3 |
| eyes wide open | 38.2 | 39.8 | 1.6 |
| **eyes closed** | **12.9** | **30.0** | **17.0** |

So it is not only a blink: on **every** photograph the iris stands proud of the
lid ring that should contain it. A blink merely makes it absurd — a
thirty-pixel iris inside a thirteen-pixel eye. The triangles joining the lid
ring to the iris are then inside-out, and warping an inside-out triangle drags
the brow and the socket down into the eye. That is the wedge.

Two things were wrong, and both had to be fixed:

1. **The irises were not part of the eye region.** An eye photograph moved the
   lids and left the pupil where the open eye had put it, so on a blink the lid
   closed *above* its own iris. They now move together — the region owns the
   lids and the irises inside them. Gaze is still geometry, not photographs;
   nothing here chooses where the pupil points.
2. **Nothing kept the iris inside the eye.** It now is, by one rule applied to
   the source mesh and the target alike: the iris ring is squashed vertically
   until it fits within the nearer lid. Vertically only — a lid occludes from
   above and below and leaves the width alone. Squashing stands in for
   occlusion, which a triangulated mesh cannot express.

The rule replaces `_EYE_OPEN_RATIO`, which shrank the *target* iris by the
socket's aspect ratio and left the source mesh untouched. What a warp cannot
tolerate is the two disagreeing, and they did.

**Rendered, at the deepest blink in the owner's recorded clip** (frame 860,
`eyeBlink` 0.70): the wedge is gone. The lid, the crease and the brow read as
skin, with lashes and a sliver of iris. The same defect appeared as a hard dark
M above each eye on the brows-raised frame — looking up narrows the opening the
same way — and that is gone too. At rest the eye is unchanged, which it should
be: source and target get the same treatment, so the warp between them does not
move.

The mesh is also better conditioned than before any of this. Worst triangle
area ratio between source and target on the blink frame: **2.31 before, 2.04
after**, with the inside-out ones removed rather than merely reduced.

**What it costs.** RTX 5070 Ti, i7-14700KF; 1817 frames of the recorded clip,
the same clip both ways.

| | before | after |
|---|---|---|
| mesh points | 119 | 139 |
| triangles | 189 | 229 |
| animate mean | 19.24 ms | **20.42 ms** |
| animate p95 | 22.05 ms | **22.51 ms** |
| animate p99 | 25.11 ms | 25.16 ms |

Forty more triangles for 1.18 ms of mean and 0.46 ms of p95 — about 30 µs each,
which matches the per-call-overhead model this warp was written around. The
budget had roughly 2.5 ms of headroom and this spends about half of it. That is
a real price and it is worth it for a defect on every blink, but it means the
next mesh point is not free either.

**What is still not right, and is not this.** At a 40° turn and at extremes of
gaze the eye is poor in both versions — a dark spur off the iris rather than a
wedge above it. That is item 13's known limit (an iris dragged through a mesh
instead of translated as a rigid disc) meeting item 5's (no usable far-turn
photographs). Neither was made worse; neither was fixed here.

## 14. Direct light and a camera in front of the face — the owner's finding

Not a code defect, and worth recording because it changes what "good" costs.

**Observed by the owner:** with a shadow falling across the face, the tracker put
brow landmarks along the *shadow edge* rather than the brow, and the avatar
answered with a long eyebrow running across the face. Moving the light to face
them directly, and the camera to sit squarely in front, largely fixed it.

That is the tracker being fooled, not the engine. MediaPipe finds brows by
contrast, and a hard shadow edge is the strongest contrast on the face.
`_BROW_FOLLOW` at 0.5 then lets half of the error through by design.

**What follows from it:**

- **Lighting and camera placement belong in the product's instructions**, not
  in a developer's head. A user with a window to one side will get worse results
  than one lit from the front, and there is no amount of engine work that
  removes that entirely.
- It also explains part of why quality assessments have been inconsistent
  between sessions: the room was not a constant.
- Worth considering later: detect a strong lighting gradient or an off-axis
  camera and say so in the UI, rather than silently producing a worse avatar.

## 13. Gaze as geometry rather than as photographs — LARGELY FIXED 2026-08-14

**Seen:** *"when I move my eye to the side it looks like there are two eyes, one
at full opacity and one very faint, like a ghost."* Rendered across the weight
range and it was exactly that: at 0.5 a grey wash of two overlapping irises, at
0.75 a solid pupil with a faint second beside it.

**Why no amount of tuning could fix it.** Every other photograph changes how a
feature *looks*, and a cross-fade of two appearances is a third plausible
appearance. A gaze photograph changes where the pupil *is*, and a cross-fade of
two positions is not a position in between — it is both of them, at half
opacity. Blending one gaze photograph with the base does the same thing against
a forward-looking iris.

**Fixed by moving the pupil instead of swapping it.** MediaPipe's 478-landmark
model carries the irises — a centre and four points per eye — and the mesh
simply did not include them. It does now, so gaze is geometry: the avatar's own
iris, at its own size, positioned inside its own socket by the offset measured
on the user. Continuous, exact at any angle, and it needs no photograph.

The four gaze photographs are no longer used. They can stay in the manifest
harmlessly, but they are not what makes gaze work any more.

**What is not solved.** The mesh has four points around each iris and a handful
around the socket, so the triangles between them shear as the pupil travels.
Rendered across the range: horizontal movement is clean, upward is clean,
**downward notches the top of the iris into a slight heart shape**, and extremes
in any direction distort it further. The travel is therefore scaled to 0.55 and
clamped — ±0.065 of eye width horizontally, 0.035 up, 0.012 down — which are the
limits at which the render stops being clean.

Little is lost by that: gaze reads as a *direction* long before it reads as a
magnitude, and nobody watches a video call from the corner of their eye. But it
is a compromise and it should be recorded as one.

**The proper fix, when it is worth the time:** translate the iris as a **rigid
disc in image space** before the warp — erase it to sclera, stamp it at the
offset — instead of dragging it through the mesh. An iris is a rigid circular
object that slides; the mesh is the wrong tool for it, and this is the mesh
being asked to do something it is not shaped for.

## 12. The avatar was wearing the user's eyes — FIXED 2026-08-14

**Seen**, on a live frame the owner captured: pale, flat, staring eyes with too
much white showing, in a face that otherwise reads as plausible. *"The eyes are
the easiest way to notice this is a deepfake-type program."* They were right,
and it was the eyes doing nearly all of that work.

**Why, and it is not the blend.** Rendering the blended source showed correct
eyes — the earlier fixes had done their job. The damage happens *after*, in the
mesh warp: it maps the avatar's landmarks onto the user's, so at full strength
the avatar's eye aperture becomes **the user's eye aperture**, and the avatar's
iris is stretched to fill someone else's eye. Two people's eyes differ in size
and shape as a matter of course; 40% is unremarkable. Stretched that far, the
iris flattens and the sclera takes over, which is exactly the pale stare.

The mechanism for this already existed — `_BROW_FOLLOW`, added when raising the
brows was found to stretch the eyes beneath them — and it had simply never been
applied to the eyes themselves.

**And a second cause on top of it.** Gaze is the one photograph family where
blending averages a *position* rather than an appearance. A look-left and a
look-up both active put the pupil in two places and the blend leaves a ghost of
it in neither. Opposing pairs were already exclusive after item 8, but left and
up are not opposites.

**Fixed, both:**

- **`_EYE_FOLLOW = 0.2`.** The eyes barely follow the user's landmarks at all.
  They no longer need to: blinking, squinting, widening and gaze all arrive as
  photographs, carrying their own geometry in the blended mesh since item 11.
  The mesh's remaining job there is to place the avatar's own eyes on the live
  head, which is a rigid transform, not a reshaping.
- **At most one gaze photograph at a time**, the strongest. A pupil somewhere
  definite and roughly right beats a precise average of two wrong places.

Rendered both ways with a simulated user whose eyes are 40% wider — an ordinary
difference between two people. Before: stretched, staring, lid pulled clear of
the iris. After: the avatar's own eye shape, correct lid coverage, normal iris.

**Why this took so long to find.** Every previous fix was upstream of it and
every one was measured on the *blend*, which was never wrong. The defect lived
in the geometry the blend was warped through. Looking at the blend could not
have found it; rendering the warp did, immediately.

## 11. Aligning the photographs destroyed the very thing they were for — FIXED 2026-08-14

**The one that actually mattered**, and the reason three measured fixes in a row
produced no visible change. It was found by rendering the mouth and the eyes and
*looking at them*, after measuring proxies stopped explaining anything.

**Seen**, at 2× magnification on the composed source:

- **Blinking rendered as vertical brown streaks** across the eye socket. Not a
  soft eyelid, not a half-closed eye — streaks. This is what the owner meant by
  "the eyes make it obvious this is a deepfake-type program".
- **The teeth were a sliver.** They *were* present, crushed into a horizontal
  line perhaps two pixels tall between closed lips, which the live warp then
  stretched into a smear.

**Why.** Every source photograph was mesh-warped onto the base portrait's
landmark layout at load, so that blending them would be a straight average. The
base portrait has **a closed mouth and open eyes**. So:

- the open-mouth photograph was warped *shut*, its teeth compressed to nothing;
- the closed-eye photograph had its eyelid pulled *across an open eye socket*,
  and the few triangles spanning that socket each stretched their own strip of
  eyelid independently — which is exactly the "fangs" failure already documented
  for the mouth, appearing again in the eyes.

The alignment was destroying precisely the geometry each photograph had been
taken to supply. It worked for gaze — where the difference is texture, not shape
— which is why *some* sources always looked right and the geometric ones never
did.

**Fixed: two alignments, because two different things are being borrowed.**

- **Expression photographs are placed by a similarity transform** — rotate,
  scale, translate — fitted on the landmarks an expression does not move (the
  face outline and the nose bridge). Their own geometry survives intact. Their
  landmarks are carried through the same transform and stored.
- **Pose photographs are still mesh-warped.** For a turn the geometry is exactly
  what should *not* be borrowed: the live face supplies the angle, and the
  photograph is wanted for shading and for the cheek a frontal shot cannot show.

And the mesh now travels with the pixels: `_blend_sources` blends **landmarks**
with the same per-region weights it blends pixels, and the warp is driven from
that blended mesh rather than the base portrait's. A photograph of an open mouth
is only useful if the mesh also says the mouth is open.

**Result, rendered and looked at:** blinks are closed eyelids with eyelashes; a
smile shows a full row of teeth; a wide-open mouth shows teeth and tongue. The
comparison images are not kept in the repository — they are the owner's avatar —
but the change is not subtle and does not need a number.

**The lesson worth keeping.** Sharpness of the blended source rose from 20 to 52
and the relighting's identity leakage halved, and neither was visible, because
this defect sat downstream of both. Proxy measurements can be right and useless
at the same time. Render the thing and look at it.

## 10. The relighting was printing the user's face onto the avatar — FIXED 2026-08-14

**The biggest one, and it was hiding behind all the others.**

**Seen:** the owner, after items 8 and 9 both landed and both measured well:
*"almost the same. nothing new or special."* Two fixes that doubled and tripled
the source blend's sharpness produced no visible change, which is itself the
clue: something downstream was undoing them.

**Why.** `_match_colour` corrects the avatar's illumination by measuring the
difference between the scene and the avatar, blurring it, and adding **all of
it** back. Whatever that blur passes is therefore taken from what lies under the
mask — and what lies under the mask is *the user's face*, not the room.

The blur was a fixed 21-pixel kernel at 1/8 scale, which works out to about 11%
of face width. Measured by projecting the correction onto known patterns:

| what the scene contained | transferred to the avatar |
|---|---|
| overall brightness | 100% — lighting, and correct |
| a left-to-right light ramp | 92% — lighting, and correct |
| **a brow shadow** | **52%** — the user's face |
| **eye sockets** | **32%** — the user's face |
| **a beard edge** | **38%** — the user's face |
| skin texture | 1% |

So half of the user's brow shadow and better than a third of their beard was
being printed onto the avatar, every frame. That is the thing the product exists
to prevent, and it explains why the avatar never quite looked like its own
photograph however good the blend feeding it became.

**Fixed by changing what the correction is allowed to be.** It is now a *plane*
— one brightness and two gradients per colour channel — fitted to the masked
difference by weighted least squares. This is not a tuning change: overall
brightness and a directional gradient are exactly what a plane represents, so
they come through whole, while a brow shadow is curvature and a plane cannot
represent curvature at all.

| what the scene contained | before | after |
|---|---|---|
| overall brightness | 100% | **100%** |
| left-to-right ramp | 92% | **100%** |
| top-to-bottom ramp | 92% | **100%** |
| brow shadow | 52% | **26%** |
| eye sockets | 32% | **12%** |
| beard edge | 38% | **23%** |
| skin texture | 1% | **0%** |

Better on both axes at once: lighting is followed *more* completely, and the
user's face leaks through at less than half the rate. It is also cheaper than
the blur it replaced — three sums and a 3×3 solve over a 1/8-scale image.

**Why it is not zero.** An off-centre band has a linear component, so a plane
fits part of any asymmetric feature. Driving it to zero needs a correction that
is constrained to be *smooth in a way faces are not*, and the honest next step
if 26% still shows is to fit the plane on the **avatar's own** rendered
illumination rather than on the scene — but that is a larger change and this one
should be looked at first.

Widening the blur was tried first and rejected on measurement: at 35% of face
width the brow shadow fell to 27%, but the light ramp fell to 56%, so real
lighting stopped being corrected. The plane has no such trade.

## 9. Every photograph changes the whole face — FIXED 2026-08-14

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
time. **Still to do** — it is the remaining softness after this fix.

### Fixed, and measured against the version it replaced

Each photograph is now confined to a feathered region built at load from the
avatar's own landmarks: mouth photographs to the mouth, eye photographs to the
eyes, brow photographs to the brows. Turns and tilts stay global, because a
turned head genuinely does change every pixel.

On the owner's avatar, driven with readings taken from the real camera. The same
weights, blended both ways:

| what the face is doing | sources | **before**<br>whole-face average | **after**<br>by region |
|---|---|---|---|
| resting | 0 | 51 | 52 |
| smiling | 2 | **28** | **53** |
| smiling + glancing + brow raised | 4 | **20** | **52** |
| all of that, plus a 22° turn | 6 | **20** | **92** |

Sharpness is variance of the Laplacian on the blended source; the base portrait
measures 62. **The whole-face average lost two-thirds of the detail the moment
the face did more than one thing at once.** Composing by region keeps it.

The regions are small, which is the other half of the point:

| region | size | share of the face | photographs |
|---|---|---|---|
| mouth | 243×149 | 5.5% | 7 |
| eyes | 388×130 | 7.6% | 7 |
| brows | 439×109 | 7.2% | 2 |

Cost is roughly unchanged — 1.0 to 3.7 ms depending on how much is happening,
against 1.0 to 3.8 before. **This change bought quality, not speed**, and it
should be reported that way. The speed came from day 6's neutral baseline, which
removed the sources that had no business being active.

**What this unlocks.** More photographs are now worth generating. A mouth
photograph and an eye photograph no longer interfere, so the set can grow along
each axis without the combinations problem that made 150 photographs a worse
idea than 22.

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

---

## 16. A 4K source renders *softer* than a small one — OPEN, measured

**Seen:** the owner generated a 4K clip. Frames from it carry far more detail
than any photograph in the library, and the rendered avatar came out worse.

**Measured, both directions.** Every face normalised to 455 px wide, the base
portrait's scale, so the comparison is detail per pixel of face rather than file
size:

| source | native face | detail at 455 px |
|---|---|---|
| base portrait, PNG still | 455 px | 84 |
| smile, PNG still | 391 px | 116 |
| **smile, 4K video frame** | **969 px** | **475** |
| 720p video frame | 570 px | 36 |

Four times the detail going in. And rendered, driven by a frame belonging to
neither source set so that neither wins by being copied through a near-identity
warp:

| sources | rendered face detail |
|---|---|
| PNG stills | 113 |
| 4K video frames | 91 |

**Not diagnosed.** Reducing oversized sources with `INTER_AREA` before alignment
was the obvious candidate — the queued backlog item 9 — and it is now done, and
it moved the number by one part in three hundred. The loss is somewhere else in
the alignment or the runtime warp.

**What it means for the plan:** more 4K clips will not improve sharpness until
this is found. They are still worth generating for *coverage* — poses and
expressions the library does not have — but the resolution argument for them is
suspended until the rendered number moves.

**A caution for whoever picks this up:** the first render comparison I ran was
rigged and I did not notice for two rounds. The driving frame was the same
photograph as one of the sources, which turns that source's path into a copy and
the other's into a warp. Any comparison of two source sets has to be driven by a
frame belonging to neither.
