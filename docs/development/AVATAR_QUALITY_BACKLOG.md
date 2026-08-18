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

### Two things ruled out, 2026-08-15

**It is not the filter in the runtime warp.** That was the obvious next
candidate — `warp_mesh` reduces with `INTER_LINEAR`, and backlog 9's closing
paragraph proposed `INTER_AREA` before the mesh warp and was never done *there*.
Measured over 900 frames of the recorded clip, the runtime reduction is **1.47×
mean, 1.55× worst**. Bilinear samples 2×2, which is very nearly right at 1.5×.
The 3× figure in backlog 9 was real when it was written and is not any more:
reducing oversized sources at load moved the runtime factor from about 3× to
1.47×. So that commit did fix the reduction it was aimed at — it simply was not
where the loss is.

**And the metric cannot settle it.** Sharpness here is the variance of the
Laplacian, and *aliasing raises it*. Point-sampling a reduction throws detail
away and adds high-frequency noise, and the noise scores higher than the detail
did. Measured on the inner face, the rendered output scores **181% of the
blended source it came from** — which is not a gain, it is the proxy failing.
Every number in this item rests on that measure, including "4K sources carry
four times the detail", so the contradiction it records may be the metric rather
than the engine.

**What that means for the next attempt:** do not chase this with another
Laplacian number. Render the two source sets, driven by a frame belonging to
neither, and look at them side by side at the size a call actually shows.

---

## 17. A turn was averaged when it should have been chosen — FIXED 2026-08-15

**The owner's instruction, and they were right.** *"Every 0.1 second it should
screenshot each moment and we save the photographs that way. I think that would
be better."*

It is better, and it could not be done while the engine **blended** its sources.
Item 9 measured what happens when it is: going from 6 photographs to 20 took
animation from 13.8 ms to 22.1 ms and sharpness from 62 to 17, because every
active source repaints the whole face. So a hundred turned frames would have
made the picture worse, which is why the answer for months was "not yet".

**A turn is not a role, it is an axis.** There is one open mouth and one smile;
there is no such thing as *the* turn. The engine held four turned photographs —
near and far, each side — and averaged them against the frontal one, so every
angle between them was a blend of two angles that were neither, and past the
outermost photograph it was extrapolating from a shot that did not go that far.
That is item 5, and no number of photographs fixes it while they are averaged.

**Fixed by selecting instead of blending.** An asset may now carry the yaw it
was measured at. Those form a *ladder*, sorted by angle, with the base portrait
as the rung at zero. Per frame the engine takes the two rungs that bracket the
head's angle and weights them by distance — and past the outermost rung it
clamps rather than extrapolating, because a photograph cannot show more of the
cheek than was pointed at the camera.

Two properties fall out of it, and the second is the reason this unlocks the
owner's idea:

1. **A rest is still byte-identical to the portrait.** The base portrait is a
   rung, and it is the accumulator everything else mixes into, so at zero the
   bracket collapses onto it and nothing is mixed at all.
2. **More photographs are now free.** Only ever one or two are active, whatever
   the library holds. Thirty rungs cost what four roles did.

**The ladder, from the owner's head-turn clip:** 30 frames from −47° to +56°,
one per three degrees of yaw, each 945–982 px across the face against the base
portrait's 455. Three degrees because that is finer than the alignment error the
mesh warp leaves behind — below it the ladder would be measuring its own noise.

**Measured**, RTX 5070 Ti, i7-14700KF, 1817 frames of the recorded clip:

| | four turn roles | thirty-rung ladder |
|---|---|---|
| turn photographs | 4 | **30** |
| animate mean | 20.42 ms | **20.72 ms** |
| animate p95 | 22.51 ms | **22.78 ms** |
| avatar load | 2.6 s | 7.4 s |
| sources in memory | 42 MB | 95 MB |

Three-tenths of a millisecond for seven times the coverage. Load time and memory
are where a dense set is actually paid for, and both are one-off.

Rendered at +39.8° of yaw — the extreme in the clip, and past anything the
engine could previously hold together: a properly turned head with correct cheek
shading, a real jawline, a plausible nose. The far eye is still poor at that
angle, which is item 13's limit rather than this one.

**What this makes worth doing:** the same treatment for pitch. Tilt is the other
axis carrying exactly two photographs, and clip 5 in `AVATAR_VIDEO_PROMPTS.txt`
is a tilt clip. The ladder is one-dimensional today; a second axis is a real
change, not a parameter.

---

## 18. Three of MediaPipe's blendshapes do not measure what they are named — FIXED 2026-08-15

The owner generated a clip that holds each eye expression in turn, which makes
this measurable rather than arguable. Held over that clip:

| what the face is doing | lid opening | `eyeWide` | `eyeSquint` |
|---|---|---|---|
| resting | +0.00 | 0.00 | **0.51** |
| eyes held wide open | **+0.53** | 0.13 | 0.07 |
| eyes held shut | **−0.66** | 0.00 | **0.89** |

`eyeWide` never rises above **0.13** on eyes visibly at their widest, so the
wide-eye photograph scored 0.23 against a 0.45 floor and **was never once
selected from three clips generated to contain it**. `eyeSquint` reads *highest*
on a shut eye: it is a second, worse closure signal wearing another
expression's name, which is why it had been taking half of every blink.

The lid opening separates all three states cleanly. So the whole eye family —
wide, narrowed, shut — now comes off one geometric measurement, the same way the
teeth have come off the lip gap since day 6 and for the same reason.

**Two things this cost, both found by rendering and worth remembering.**

*A ratio of a height to a width carries the frame's aspect ratio.* Live that
cancels, because every frame is the same shape. The extraction tool measures a
2160×3840 clip against a 1024×1024 portrait, where the same eye reads 0.56 of
what it should, so every frame of every clip scored as a hard blink. Two
vertical distances have no such factor.

*A slow mean of the eye opening is dragged down by every blink and every
glance*, until it sits below the eye's actual resting opening — and then a face
doing nothing reads as wide. Rendered: a resting avatar staring with the whites
showing. It follows an opening eye slowly and a closing one ten times slower
still, so what it estimates is "how open this eye is when it is open".

## 19. The clearest example of a role is not the highest-scoring frame — FIXED 2026-08-15

The extractor took the frame that scored highest for each role. For "eyes
closed" that was the frame from the **frown**: a hard scowl squeezes the lids
further shut than resting them does. Every region reaches a little past its own
landmarks — the eye region has to carry the lid and the socket shadow, so it
takes in the lower brow — and the avatar therefore blinked with a glare.

Frames are now ranked by their score for the role *minus half of everything else
they are also doing*. The chosen closed-eye frame moved from the frown to the
three-second closure the clip was built around.

---

## 20. Every rung of the turn ladder was torn in half — FIXED 2026-08-15

Item 17 gave the engine thirty photographs of the owner's face between −47° and
+56° and a rule for choosing between them. It did not fix the turn, because
**each of those photographs was folded flat before it was ever used**.

`recordings/turn-sources-broken.png` is what the engine actually held: a
vertical seam down the face, the far eye and brow drawn twice, the mouth smeared
sideways. Every rung looked like this, and at any yaw past a few degrees one of
them was applied at full weight.

**One cause, three complaints.** The owner's *"a slight turn nearly loses the
face"*, their *"two photographs on top of each other"*, and their *"only good at
five to ten degrees"* were all this.

### Why

A rung was mesh-warped onto the **frontal** portrait's landmark layout, on the
argument that the live face supplies the angle so the photograph is wanted only
for its shading. That argument is true of the result and false of the method.
The live warp maps a source mesh onto the tracked face, and it can only do that
if the mesh describes the source. Given the frontal layout for a 35-degree
picture, the fold happens at load, before a frame is ever composed.

Measured on the owner's own ladder, against a 455 px face:

| rung | mean landmark distance from the frontal layout | worst |
|---|---|---|
| ±16° | 30 px | 69 px |
| ±26° | 41 px | 105 px |
| ±36° | 61 px | 156 px |
| ±47° | 77 px | 200 px |

That distance *is* the fold. It is what the mesh warp had to close, by dragging
triangles across the face.

### Fixed by the change item 11 already made for expressions

Every photograph is now placed by a **similarity transform** — rotate, uniform
scale, translate — and keeps its own geometry, pose photographs included. Its
landmarks travel with it, and the blend carries them into `blend_points`, so the
live warp runs from a turned mesh onto a turned face and is nearly an identity.

**Two frames of reference then meet in the blend, and that is the whole of the
new complexity.** Expression photographs are still cut from a face looking
straight ahead. Pasted at the frontal mouth's coordinates over a head that has
turned, a mouth photograph lands 35 px beside the mouth at 17° and 70 px beside
it at 36° — the same double exposure, in a new place. So a region is carried
onto the moved head, pixels and mask and landmarks together, before it is
composed. Below a pixel of travel it is left alone: carrying costs a resampling,
and a resampling softens.

### Measured, on the recorded clip — 1847 frames

The number is **how hard the live warp has to work**: after a rigid fit of the
avatar's blended mesh onto the tracked face, the mean distance left over. That
residual is what the warp closes by deforming, so it is a direct measure of how
much the picture is being stretched.

| | before | after |
|---|---|---|
| warp residual, all frames | 7.12 px | **6.83 px** |
| warp residual, frontal | 6.82 px | 6.82 px |
| warp residual, turned past 10° | 22.43 px | **7.35 px** |
| rendered sharpness, all frames | 140 | **185** |
| animate mean | 21.55 ms | 21.37 ms |
| animate p95 | 24.14 ms | 24.03 ms |

A turned face now costs the warp what a frontal one does — 7.35 px against
6.82 — where before it cost three times as much. The frame budget is unchanged
within run-to-run noise, which is ±0.4 ms on this machine.

Sharpness over the 35 turned frames alone reads 58 → 50, and it is the one
number here that should not be taken at face value: a Laplacian counts the
fold's own seam as detail. Rendered side by side at +39° — the extreme in the
clip — the two frames measure 87.3 and 87.4, and the beard, the brow and the eye
are plainly better in the second. See `recordings/turn-before-after.png` and
`recordings/turn-sources-fixed.png`.

### What it cost: the tilt photographs, dropped

The third family to go, after the four gaze photographs (item 13) and the squint
(item 18), and for the same reason each time — *a photograph that is wrong more
often than it is right costs more than it gives.*

Measured over the whole recording, a tilt photograph was active on **99.2% of
frames at a median weight of 0.38**. Nearly 40% of every frame of an ordinary
conversation was a different photograph laid over the portrait. That was
survivable while everything was folded onto the frontal layout, where it read as
shading; once a pose photograph brought its own geometry, the mesh followed it —
and the geometry is wrong. `clip_tilt_down.png` sits **47.5 px** from the base's
landmarks, a tenth of a face, for a photograph about six degrees further down
than the portrait. It is not a pitch axis; it is another frame of the same
person, differing in more than pitch.

With the tilts in, the mesh sat **11.0 px** from the tracked face on frontal
frames against **6.8 px** without them. Rendered, they are a face pulled
downward with hollows under the eyes, and everything softer, because
cross-fading two pictures that do not agree is a blur. Dropping them is most of
the 140 → 185 sharpness above.

**What would make pitch usable is what made yaw usable:** a measured angle on
the asset and a ladder of them from a nod clip, so the weight is chosen by the
angle the photograph was shot at rather than by a guessed full-scale number.
`extract_avatar_frames.py` already records yaw per frame; pitch is the same
measurement, and clip 5 in `AVATAR_VIDEO_PROMPTS.txt` is a tilt clip. This is
the second axis item 17 said would be a real change rather than a parameter —
and it is now the cheapest quality win left in the photograph set.

---

## 21. The mouth was an empty hole — FIXED 2026-08-15

**The owner, having looked at the renders:** *"when smiling the teeth do not
show. And when the mouth opens, no tongue and no teeth either."*

They were describing a flat maroon void where the inside of a mouth should be,
and there were **three separate causes** behind it, each hiding the next.

### One: the interior was cut from the wrong photograph

`_paint_mouth_interior` fills the live opening with cavity colour and then
copies back whatever the source picture has inside its own lips. Which pixels
those are was a mask cut **from the base portrait at load** — and the base
portrait's lips are together, a sliver four pixels tall on a 455 px face. So
however wide the blend's mouth was, only that sliver was ever copied back and
the rest stayed flat colour.

It is asked of the **blended mesh** now, which is the thing that actually
describes the blend, and drawn as a polygon rather than warped as a mask — the
same answer without a second resampling. The precomputed mask is gone.

### Two: the mouth region was sized to a closed mouth

With the interior copying again, a wide open mouth showed **stubble under the
tongue**. Each region's box and mask were cut from the *base's* landmarks, and
the base's lower lip sits 79 px above the open-mouth photograph's. Everything
below the cut was still base portrait — a chin — and the interior copy dutifully
carried that chin into the opening.

Regions are now sized to hold every photograph that acts on them. The mask is
the base's hull grown by the usual margin, **plus each photograph's own hull
with no margin at all**, and that second half is not a detail: a photograph may
act where the mesh follows it and no further. Adding the margin to a
photograph's hull as well pushed the mouth's mask 124 px down the chin, where
the face outline lives and no region moves it — which rendered as a second lower
lip below the first.

### Three: the teeth were an average of photographs that disagree

That left teeth, but translucent ones, with a closed mouth showing through. At
the owner's widest smile the mouth region was:

| photograph | weight | its own lip gap |
|---|---|---|
| `source_smile` | 0.60 | 0.075 — **teeth** |
| `source_smile_closed` | 0.40 | 0.015 |
| `source_mouth_pressed` | 0.28 | 0.007 |
| `source_mouth_o` | 0.05 | 0.010 |
| `source_mouth_wide` | 0.05 | 0.083 |

Normalised, 43% of the mouth was a photograph with teeth in it and 57% was
photographs with their lips together. 43% of a tooth is a translucent tooth.
This is the fourth appearance of the same defect — the gaze photographs, the
squint, the averaged turns, and now the mouth: **an average of pictures that
disagree is a picture of neither.**

The lips may cross-fade; an opening is a continuum and the photographs either
side of it agree about where the mouth is. What is *behind* them may not. So
the interior is now taken whole, from whichever photograph with its lips
genuinely parted carries the most weight in the frame, placed by a fit of its
own lips onto the live ones. Its own inner lips are both the fit and the limit
of what is copied, so teeth arrive opaque and no lip ever does.

Which photographs have an interior is measured at load rather than listed. The
owner's set separates cleanly at a fifth of the way open — `source_mouth_open`
0.199, `source_mouth_wide` 0.083, `source_smile` 0.075, then a gap down to
`source_smile_closed` 0.015 and everything else below it.

**Rendered:** a smile with a full row of opaque teeth, and an open mouth with
teeth, tongue and lower teeth. What is left is a dark crescent at the bottom of
a very wide opening, where the photograph's own mouth does not reach — which is
the intended behaviour: opening wider than any photograph shows more darkness.

**Still open in the same family:** the lips themselves are still an average of
five photographs, and `source_mouth_o` is active on 84% of the recording. A
mouth shape is not an axis the way yaw is, but a rounded-lip photograph present
in five frames out of six of ordinary speech is the tilt photographs' problem
wearing another face.

---

## 22. Every turned photograph in the library is motion blurred — the material, not the engine

**The owner, on the renders after the turn fix:** *"the head turned to the side
is horrible."* They are right, and after item 20 the reason is no longer the
geometry. It is that **the photographs at those angles are smears**.

### The measurement, and why it took three attempts

The obvious measure says nothing here. Laplacian variance over a face counts
the detail the face is *showing*, and a turned face shows far less of it — one
eye goes behind the nose and a smooth cheek fills the frame. This file already
records the effect: a median of 179 at 0-8 degrees of yaw against 9 at 20-35,
**on frames sharp enough to count beard hairs**. So a blurred turn and a sharp
turn score alike, and every filter built on that number is blind to exactly the
defect it was meant to catch.

The eye nearer the camera does not have the problem. It keeps its lashes, its
brow hairs and its iris edge to well past 40 degrees, so it can be compared
across a whole sweep. Measured that way on the owner's own ladder, each rung
against the sharpest rung in it:

| rung | near-eye detail | of the best |
|---|---|---|
| −1.6° | 645.7 | 100% |
| +16.9° | 13.9 | **2.1%** |
| +26.2° | 9.5 | **1.5%** |
| +38.9° | 9.6 | **1.5%** |
| +55.1° | 11.0 | **1.7%** |

One frame in the clip was taken while the head was still, and it is the only
sharp one. Every other rung was taken while the head was moving.

### Why nothing caught it

The extractor's blur floor is deliberately *local* — a frame is judged against
the third of a second around it, because that is the only fair comparison for a
measure that responds to what the face is doing (item 19's neighbours). In a
clip where the head turns continuously, **every frame in that window is blurred
alike**, so the local maximum is itself a smear and passes. The floor was built
for hold-and-release expression clips and is exactly wrong for a sweep.

### What changed here

Nothing in the engine, because there is nothing in the engine to change: a warp
cannot put back detail that was never photographed. Two things instead —

1. **`extract_avatar_frames.py` now measures the near eye** and reports each
   rung as a percentage of the sharpest one, with a warning naming the cause
   and the remedy when rungs fall below a quarter. The next blurred ladder gets
   caught before an avatar is built from it.
2. **The clip 4 prompt now asks for a stepped turn**, not a slow sweep: turn a
   little, hold still, turn a little, hold still. The old prompt asked for a
   continuous movement with three pauses per side, which cannot fill a ladder
   that wants a rung every three degrees however slowly it is taken.

**This is a supply problem now, and it is the largest one open.** The engine's
geometry at a turn is correct and measured — 22.4 px of warp residual down to
7.3 — and the picture will stay soft at every angle until the clip is
regenerated.

---

## 23. At a turn the mask left the user's own nose and mouth on screen — FIXED 2026-08-15

**The owner, testing the application:** *"when I turn my head, the face is lost
and my own face appears, half of it."*

They were describing the one failure this product cannot have, and they were
describing it accurately: at 39 degrees of yaw, **their real nose, their real
lips and their real chin were on screen**, beside an avatar's cheek.

### Why

The composite mask is the face oval, filled as a polygon — deliberately not the
convex hull, because a hull bridges the gaps beside the chin and in front of the
ears and fills them with whatever was behind the head in the portrait.

But the face oval is a ring of *anatomical* points around the head, and a head
is only convex from the front. Turned far enough, the nose and the lips stand
**outside** that ring in the picture: the ring has swung round behind them. The
mask therefore stopped short of the profile, and everything in front of it was
never replaced.

Rendered with the mask outlined, the gap is unmistakable — the whole nose and
mouth sit outside the green line.

### Measured rather than eyeballed, after the first attempt failed

The first fix added the nose and the lips, and the owner tested it: *"still
there, almost the same. When I turn left, slowly my left eye and part of my face
appear; turning further, the nose too."*

So it was counted instead. Every landmark is a point on the user's real face, so
any of them outside the mask is a piece of that face on screen. At 39° of yaw,
before any of this:

| group | outside the mask | weakest coverage |
|---|---|---|
| eyes | **16 of 32** | 0.00 |
| brows | **6 of 12** | 0.00 |
| irises | **5 of 10** | 0.00 |
| nose | 6 of 11 | 0.02 |
| cheeks | 4 of 8 | 0.12 |
| lips | 7 of 32 | 0.02 |

Half of each group, and always the same half: **the far side of the face is
outside the mask entirely**. Not a rim — a face. The owner's "part of my face"
was one whole side of it, and their eye at zero coverage.

### Fixed

The convex hull of the brows, eyes, nose, lips and cheeks is added to the mask.
Their hull rather than their outline, and only theirs: a blob over the middle of
the face cannot bridge to an ear or to the background, which is the whole reason
the outline rule exists.

It goes on **after** the erosion, at full strength. The erosion pulls the outer
boundary inside the real face so the blend happens over skin on both sides;
these are not a boundary but the middle of the face, and eroding them would
leave a rim of a real eye showing. Only the feather crosses them.

| group | outside before | after |
|---|---|---|
| eyes | 16 of 32, at 0.00 | **0 of 32, weakest 1.00** |
| irises | 5 of 10, at 0.00 | **0 of 10, weakest 1.00** |
| brows | 6 of 12, at 0.00 | 2 of 12, weakest 0.38 |
| nose | 6 of 11, at 0.02 | 1 of 11, weakest 0.41 |
| cheeks | 4 of 8, at 0.12 | 1 of 8, weakest 0.34 |

| frame | yaw | mask area before | after | |
|---|---|---|---|---|
| 300 | +0.4° | 93357 | 93462 | **+0.1%** |
| 960 | −0.7° | 92331 | 92440 | **+0.1%** |
| 1442 | −2.1° | 92881 | 92983 | **+0.1%** |
| 1533 | −1.9° | 102480 | 102598 | **+0.1%** |
| 1846 | +39.3° | 65236 | 75921 | **+16.4%** |

Frontal it is the same mask to a tenth of a percent. At a turn it covers a sixth
more of the face, and what it covers is an eye.

**What remains.** The points still outside are the outermost ones — the end of a
brow, a mouth corner — and they sit at 0.34-0.41 rather than at zero, which is
the feather rather than a hole. The oval's own points are outside by
construction: the oval *is* the boundary, and it is eroded on purpose.

### And the eyes stopped following at a turn

The owner, on the same run: *"when I look sideways it closes the eye by itself
while mine is open, and it almost never follows my eyes opening and closing
then."* Two separate things, and only one of them is the engine.

**The closing is the blur.** Two candidates were measured and one survives. The
engine does not ask for a closed eye at a turn — over the whole recording, the
closed-eye weight is 0.082 within 2° of rest and **0.000 on every frame past
5°** — and the lid measurement does not drift either, reading 0.058 at a turn
against 0.054 at rest, slightly *more* open. Nor are the ladder photographs
narrow-eyed: measured with the engine's own lid ratio, every rung sits between
95% and 109% of the portrait. What is left is item 22: a motion-blurred eye
reads as a shut one, which is the same sentence as the owner's other complaint
in the same breath, *"the eyes look smeared."*

**The not-following is real, and now fixed.** Every eye photograph is of a face
looking straight ahead, so at a turn the carry fades it out (item 20) and the
blend has nothing left to say about the eyes — they freeze at whatever the
turned photograph caught. The mesh could say something, but it deliberately does
not: `_EYE_FOLLOW` is 0.2 because item 12 found that following the user's eye
aperture gives the avatar somebody else's eyes, pale and flat.

So the mesh now takes over **exactly as much as the photograph loses**: the
follow rises from 0.2 toward 0.6 in proportion to what was asked for and could
not be placed. At rest, where nothing is asked for, nothing changes and the
avatar keeps its own eyes. At a turn, a blink reads as a blink again.

The owner's own diagnosis was *"we don't have photographs of a turned face with
the eyes closed"*, and that is the right reading of the cause — but such
photographs cannot be used yet: the ladder is one-dimensional, and yaw × eye
state is the second axis item 17 flagged as a real change rather than a
parameter. Generating them now would waste the clip.

### And the mouth interior is anchored to the upper lip now

Same test, second complaint: *"when the teeth show, the lips are lost."* The
interior was fitted to the whole inner-lip ring, so the photograph's teeth were
scaled to whatever the live opening happened to be — open wider and the teeth
grow with it, and the thin shadow above them grows into a band the size of a
lip, sitting exactly where the upper lip should be.

The upper teeth hang from the skull and the jaw drops away from them, so the fit
is now on the inner lip's **upper arc** alone. The teeth arrive at the size the
mouth's width implies, stay under the lip they belong to, and opening wider
reveals darkness below them — which is what opening a jaw does.

---

## 24. A sliver of iris through a closed eyelid — FIXED 2026-08-18

**Seen**, by the owner, and it was the one defect they stopped the night on:
*"at a blink, a small piece of the iris shows through the closed eyelid."*
Screenshot 2026-08-17, eyes closed, the sliver at the outer corner of one eye.

### Two candidates were written down the night before. Both were wrong

The note left for the morning named the iris geometry: gaze is placed by
`_place_irises` and held inside the lid ring by `_contain_irises`, so at full
closure the iris is *fitted into a slit rather than hidden by it*, and what
shows is the portrait's own iris squeezed but not covered. The second candidate
was the eye region's mask failing to reach the outer corner.

Both were measured on the recorded clip before anything was changed, and neither
is the defect:

| measured | result |
|---|---|
| eye mask over the closed rung's own lid | **1.000** mean, 0.996 worst, 1.000 at the outer corner |
| composite at full closure | 100% the rung, no portrait underneath |
| iris points outside the lid ring after containment | 3–4 of 5, worst 4.7 px |

The mask is not the cause. The containment *is* wrong — it holds the iris inside
the ring's **bounding box** rather than the ring itself, and a shut lid is a lens
whose box is far taller than the lid is anywhere near the iris — but at full
closure the pixels under those triangles are eyelid either way, so it costs a
one-pixel nick in the lash line and not an iris. **Recorded here, not fixed:
see the note at the end.**

### What it actually is: the arrival fade, and a hole in the ladder

Rendered frame by frame through one blink, with the share that produced each:

```
f691  lid +0.10  share 0.00  (no photograph)
f692  lid +0.33  share 0.33  eyes@shut   <- 67% wide-open portrait underneath
f693  lid +0.60  share 0.67  eyes@shut   <- 33%
f694  lid +0.64  share 1.00  eyes@shut       clean
...
f702  lid +0.13  share 0.67  eyes@shut   <- and again on the way back up
f703  lid +0.12  share 0.33  eyes@shut
```

Those four frames are a genuine average of two photographs that disagree — a
shut lid over an open eye — which is a translucent iris under a closed eyelid,
up to **131 grey levels** from the frames either side of it. Over the whole
recording, **107 frames of 1847** were such a mix.

Day 10's eighth pass had already removed this fade from the *amount* — "a rung
is shown whole" — and left it in the *arrival*, where `_eased` ramps a region in
over a tenth of a second. A tenth of a second is three frames and a blink is
three or four, so for the lids the fade lasted the whole blink.

**Why it appeared on 2026-08-17 and not before:** that afternoon's own fix. The
two shallow rungs the ladder had, `eyes_c014` and `eyes_c017`, were refused at
load for carrying their brows down inside the portrait's eye — correctly; the
owner had picked both off the screen. That left the closing side of the ladder
running from a wide open eye at 0.0 straight to a **fully shut** photograph at
0.32, so the fade had the largest possible pair of pictures to average.

### Fixed, in two places

1. **The lids arrive whole.** `_ARRIVES_WHOLE` — the eyes region only, and only
   for rungs. A rung already says how far shut the eye is, so there is nothing
   left for an opacity to express. The mouth and the brows keep their fade;
   nothing else on a face moves inside the time a lid takes to fall.
2. **A band is no longer lost to a frame the engine will refuse.**
   `refused_as_rung` is the load-time check, offered to the builders, and
   `build_eye_ladder` now takes the sharpest frame in each band *that the engine
   will accept*. Both holes filled, from the same clip:

   | band | was | now |
   |---|---|---|
   | +0.1 | `video-08/0107` — refused, brows 16 px inside the eye | `video-07/0050` at +0.122 |
   | +0.2 | `video-08/0110` — refused, brows 16 px inside the eye | `video-07/0051` at +0.221 |

   The cost is sharpness: 99 against 101, and 76 against 102. A softer rung that
   reaches the screen is worth more than a sharp one that is thrown away.

### Measured after

| | before | after |
|---|---|---|
| frames that are a mix of two eye photographs | **107 of 1847** | **0** |
| eye rungs on the closing side | 4, none of them partly shut | 6 |
| `animate` mean / p95 | 21.20 / 23.93 ms | 20.91 / 23.91 ms |

The frame cost is unchanged. **Those two figures replace a pair first
reported at 34 ms**, which were taken while this machine was running six
other measurements at once; re-run on a quiet one they are the 20.4 / 22.5
the budget section already records. No frame-rate problem was found.

### Two things measured and deliberately not done

**A hold at the foot of the ladder.** Once the lids arrive whole, a reading
sitting on the start threshold turns the region on and off rather than fading
it, so a hold there looked obviously right. Measured, it is not: releasing at
three fifths of the threshold **more than doubles how long a closed eye is
held** — the median run goes from 18 frames to 41 — to buy eight fewer changes a
minute. The shortest run without it is three frames, which is a blink and not a
flicker, so there was nothing to buy.

**Containing the iris by the lid outline** instead of its bounding box. Correct
in principle, and it does remove the nick: rendered both ways, the change is a
thin line along the lash line, mean 1.1 grey levels with local peaks near 100.
But collapsing the iris into a shut lid makes those triangles near-degenerate,
and the affine fitted to them pulls in a **bright speck** where the nick was.
One artefact for another. Left open — the honest version clips the iris out
rather than shrinking it, and a triangulated mesh cannot express that without a
second triangulation for the closed eye.

### Still open, and it is material rather than code

The closing side of the ladder has no genuinely *half* closed rung, and the clip
does not contain one. Measured on `video-07`, the eyes clip:

```
0048  amount -0.004   open
0049  amount +0.037   half closed   <- the only frame in the gap
0050  amount +0.130   nearly shut
```

The lid crosses from open to nearly shut in **two frames**, the dump is one
frame per 50 ms, and `DEFAULT_MIN_MOVED = 0.08` discards frame 0049 before the
bands are formed — and `_EYES_LADDER_START = 0.12` would refuse to choose it
anyway. A blink is therefore a two-state cut, open to nearly shut. At 30 fps a
real lid closes in one or two frames, so this is close to honest; but if the
half-closed eye is wanted it needs a clip that closes *slowly*, and a finer band
near zero to file it under.

---

## 25. A shut eye's iris is tucked out of the way — FIXED 2026-08-19

**Seen**, by the owner, after item 24 removed the cross-fade: *"still slightly
visible. It is the pupils, but very thin — low in height, a bit long in width. A
flat, oval shape, for a very short time."*

Item 24's note said the containment was wrong but that it cost only a nick.
That was the right diagnosis and the wrong size.

### Found by removal, not by argument

Three renders of the same frames, side by side: as shipped, with the target lid
ring and iris ring drawn on, and with **every triangle touching an iris taken
out of the mesh**.

* The dark flat shape is on the lash line, toward the inner corner.
* The iris ring is drawn exactly over it.
* With the fan removed it is gone, and the lash line is continuous. No hole
  appears — the area is a few pixels tall and the rigid backdrop layer under the
  mesh is the composite's own shut lid.

Drawn on the frame, the lid ring at full closure **crosses itself**: the upper
and lower contours swap over near the corners. So "inside the lid" is not even
well defined there, which is why containment against the ring's *bounding box*
let three or four of the five iris points sit outside it.

**What the fan draws, once there is no eye to draw in, is the lash line squeezed
flat** — the darkest thing in the picture, compressed into a wide, low ellipse.
That is the owner's flat oval, and it lasts exactly as long as the lid is down.

### The line, measured on this face

The lid ring's height over its own width — a ratio, so it says the same thing in
the avatar's pixels and in the live frame's, at any distance from the camera:

| the eye | height / width |
|---|---|
| base portrait | 0.323 |
| the four wide rungs | 0.403–0.448 |
| every frame of the recording with the eye open | 0.298–0.450 |
| every shut rung | 0.106–0.128 |
| every frame with the eye shut | 0.106–0.123 |

**Nothing lies between 0.128 and 0.298**, on either side of the warp, so the
line goes in the middle of the empty band — written as **three fifths of this
portrait's own resting lid**, not as a number, because a fraction of *this* face
is the only kind of threshold this file has been able to trust.

Over 1847 frames it never misfires:

* dropped while the lid reading was below 0.10: **0 frames**
* kept while the lid reading was above 0.30: **0 frames**
* closest an open frame came to the line: **0.298**, against a line at 0.194
* closest a shut frame came: **0.191**

### Taking the fan out was the first answer, and it was wrong

That is what shipped first, and the owner came back with an arrow on each eye:
*"the small thin blackness above the red arrows, that is the problem."*

**The iris fan's triangles were the only ones covering the inside of the lid.**
`warp_mesh` lays a rigid placement of the whole composite down before the mesh
paints, so that no pixel is ever black — and with the fan gone, that backdrop
was all there was inside a shut lid. It carries the photograph's *own* lash line,
placed by a single similarity transform rather than by the mesh, so it lands a
few pixels off: **a second, thinner, darker lash line inside the shut lid.**

Measured by painting the backdrop a flat colour and counting what stayed:

| frame | pixels in the eye crop painted by no triangle |
|---|---|
| 694 | 27 |
| 695 | 37 |
| 696 | 36 |
| 702 | **252** |

Every one of them along the lash line. That is the arrow.

### So the ring is tucked, not removed

`_tuck_iris` pulls a shut eye's iris ring in to **0.35 of its distance from the
lid ring's own centroid**. The fan stays, the hole never opens, and what those
triangles paint is the lid's own pixels.

Scaling about the lid's centroid is what makes the two meshes agree: it is a
linear function of the ring, so wherever the destination ring is near an affine
image of the source's — which two descriptions of one shut eye are — the fan is
near an identity and paints the lid back onto itself. And the triangles between
the tucked ring and the lid *grow* rather than shrink, which is the other half:
the earlier attempt that clipped the iris to the lid outline drove them to no
area at all, and an affine fitted to three nearly-collinear points put a
**bright** speck where the dark one had been.

Rendered three ways on the same frames — containment only, fan dropped, ring
tucked — the third is the only one with neither the dark hook nor the thin line.

**Each eye is decided on its own.** A wink is a face doing two different things
at once, and the tuck is per eye by construction.

### Measured after

| | containment only | fan dropped | ring tucked |
|---|---|---|---|
| the flat dark oval | present | gone | gone |
| the thin line inside the lid | — | **present** | gone |
| pixels painted by no triangle | 0 | 27–252 | **0** |
| `animate` mean | 21.20 ms | 21.48–21.51 ms | 21.79–21.91 ms |
| p95 | 23.93 ms | 24.58–24.60 ms | 24.97–25.15 ms |

Three runs each, same machine, same recording. The tuck costs about **0.3 ms of
mean and 0.5 of p95** against dropping the fan, because the forty triangles it
gives back are drawn again. That is the price of not having a hole, and there is
room for it: the budget is 33.3 ms and the frame is at 22.

**And a correction to item 24.** That entry reported `animate` at 34 ms and
called it the largest open number in the project. It was not the engine — it was
this machine, running six other measurements while the figure was taken. Re-run
on a quiet one, before and after, the engine is at the 20.4 / 22.5 the budget
section already records. There was no frame-rate regression to find. The lesson
is the one the project already had on its list and had not yet done: **a quality
measure has to be a committed tool that reports its own conditions**, or a
number taken in passing ends up in a commit message.

---

## 26. An open eye stayed behind the closed one — FIXED 2026-08-19

**The owner's own diagnosis, and it was right:** *"during the closed-eye
photograph, I think the open-eye photograph stays behind it. That is my guess."*

It was not the base portrait, and it was not the brow rungs either — removing
the brows region entirely moved the number by 0.1 of a grey level. It is
**everything else in the library at once**, because of the order the composite
is built in.

### Why every photograph but one has an open eye in it

`_blend_sources` lays the whole-head photographs down first, across the entire
image: `_weighted_sum(avatar.image, avatar.sources, whole_head)`. Those are the
turn rungs, and **one of them is mixed in on nearly every frame** — a head is
almost never at exactly its resting angle, so the ladder's nearest rung is
almost always carrying some weight. A turn rung is this face at an angle with
its **eyes open**. The regions then compose on top.

So whatever share of the eye's opening the eyes region does not hold is an open
eye — sclera, iris and all — and the mesh squeezes it into the slit that a shut
lid has become. Measured on the owner's avatar, over the portrait's own eye:

| | eyes region | every other region |
|---|---|---|
| eye 0, the opening | mean 0.985, **min 0.608** | max 0.349 |
| eye 0, the iris | mean 0.996, **min 0.878** | max 0.118 |
| eye 1, the opening | mean 0.961, **min 0.333** | max 0.549 |
| eye 1, the iris | mean 0.978, **min 0.608** | max 0.349 |

Two thirds of an open eye at the worst pixel of it. The deficit is the brows'
claim: their mask is cut at the top of the eye *before* it is blurred, and a
feather is 5% of a face — 23 px here — so the blur puts it back over the top
eleven pixels of the eye. The iris starts four pixels below that line.

### Fixed: nobody but the eyes paints inside an eye

A fence around the portrait's own openings, lids and irises together, subtracted
from every other region's mask after its blur.

**Feathered at 1.5% of face width rather than 5%**, and that is the whole
difficulty of the fix rather than a detail. The gap between the top of an eye
and the brow above it is smaller than a region's feather, so a fence faded over
the usual distance reaches up the ridge and clips the brow rungs that hang
lowest — which is day 10's second brow, arrived at from the other side.
Measured, at 1.5%:

| | before | after |
|---|---|---|
| eyes over their own iris, eye 0 | min 0.878 | **min 0.996** |
| eyes over their own iris, eye 1 | min 0.608 | **min 0.946** |
| anything else over the iris | max 0.349 | **max 0.011** |
| the deepest brow rung over its own brow | mean 0.639 | 0.625 |
| the next two | 0.823, 0.880 | 0.817, 0.877 |
| no pixel claimed twice | 1.0000 | 1.0000 |

The brow rungs lose **0.014 at worst**, and they were already losing that much
to their own floor before this — a brow that has come down past the top of the
eye is clipped by `brow_floor`, which is a separate item and not this one.

Frame cost is unchanged: this is all at load.

### What is not yet known

The owner sees this live and the recorded clip only shows the mild half of it —
`driving.mkv` is frontal through its blinks, so the turn rungs are barely
weighted there, and the visible difference before and after is confined to the
upper lid arc. **Whether this is the whole of what they are seeing is not
established**, and the way to establish it is a driving clip recorded in the
pose that shows it, not another guess.
