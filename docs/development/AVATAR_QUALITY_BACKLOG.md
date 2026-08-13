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
