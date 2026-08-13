# Generating the avatar photograph set

The warp engine can only show what the source photograph contains — that is the
cause of three of the five problems in `AVATAR_QUALITY_BACKLOG.md`. This file is
the recipe for the extra photographs that fix them.

**Verify every generated photo with `scripts/check_avatar_photos.py` before
using it.** The thing that ruins this set is not image quality, it is *identity
drift*: a variant that is subtly a different person makes the avatar change face
when the engine switches source, which is worse than the problem being solved.

---

## The rules that matter more than the prompt

1. **Attach the original `portrait.png` every single time.** Never generate a
   variant from a previous variant — the drift compounds.
2. **One change per photo.** Asking for a smile *and* a head turn gives a worse
   result than two photos.
3. **Everything else must be identical**: lighting direction and colour, camera
   distance, framing, head size in frame, background, hairstyle, facial hair.
4. **Square, 1024×1024**, same as the base.
5. Left or right does not need to be exact — the checker measures the actual
   yaw and reports which way each photo turned.

---

## The prompts

Each is written to be pasted as-is, with the base photo attached.

### Shared preamble

Every prompt below starts with this. Paste it as one message together with the
variation that follows it.

```
Use the attached photograph as the exact reference for this person's identity.
Produce a new photorealistic photograph of the SAME person.

Keep these completely unchanged: face shape and bone structure, eye colour,
eye spacing, nose shape and size, lip shape, skin tone and texture, facial
hair, hairstyle and hairline, lighting direction and colour temperature,
background, camera distance, framing, and how large the head is within the
frame.

Do not add glasses, makeup, filters, or stylisation. Photorealistic, sharp
focus, plain background. Square 1:1 aspect ratio, 1024x1024.

The only change is:
```

## The complete set

Twenty-two photographs, in the order they are worth generating. Each is a role
the engine already knows how to drive; supply one and it is used, leave it out
and the avatar does without it.

**Tier 1 — the mouth.** Speech is where the eye goes, and the mouth is where a
warped photograph has the least to work with.

| # | Role → file | The only change is |
|---|---|---|
| 1 | `source_smile` → `portrait_smile.png` | ✅ have |
| 2 | `source_mouth_open` → `portrait_mouth_open.png` | ✅ have |
| 3 | `source_smile_closed` → `portrait_smile_closed.png` | a warm closed-mouth smile, lips together, no teeth showing at all |
| 4 | `source_mouth_o` → `portrait_mouth_o.png` | lips rounded and pushed forward as when saying "oh" or "you", a small round opening |
| 5 | `source_mouth_wide` → `portrait_mouth_wide.png` | lips stretched wide and flat as when saying "ee", teeth close together and visible |
| 6 | `source_mouth_pressed` → `portrait_mouth_pressed.png` | lips pressed firmly together as when saying "m" or "b", a flat closed line |
| 7 | `source_frown` → `portrait_frown.png` | corners of the mouth pulled down in a frown, mouth closed |

**Tier 2 — eyes and brows.** Gaze is the difference between a face that is
present and one that is staring.

| # | Role → file | The only change is |
|---|---|---|
| 8 | `source_eyes_closed` → `portrait_eyes_closed.png` | ✅ have |
| 9 | `source_brows_up` → `portrait_brows_up.png` | ✅ have |
| 10 | `source_look_left` → `portrait_look_left.png` | eyes looking to one side without the head turning, pupils toward the corner |
| 11 | `source_look_right` → `portrait_look_right.png` | eyes looking to the other side, head still facing forward |
| 12 | `source_look_up` → `portrait_look_up.png` | eyes looking upward, head still facing forward |
| 13 | `source_look_down` → `portrait_look_down.png` | eyes looking downward, head still facing forward |
| 14 | `source_eyes_wide` → `portrait_eyes_wide.png` | eyes opened wide as in alertness, brows not raised |
| 15 | `source_squint` → `portrait_squint.png` | eyes narrowed in a squint, lower lids raised |
| 16 | `source_brows_down` → `portrait_brows_down.png` | brows drawn down and together in a frown of concentration |

**Tier 3 — head pose.** These extend how far you can turn before the avatar
starts guessing.

| # | Role → file | The only change is |
|---|---|---|
| 17 | `source_turn_left` → `portrait_turn_left.png` | ✅ have (~20°) |
| 18 | `source_turn_right` → `portrait_turn_right.png` | ✅ have (~20°) |
| 19 | `source_turn_left_far` → `portrait_turn_left_far.png` | head turned about 40 degrees to the same side as photograph 17, close to profile but both eyes still visible |
| 20 | `source_turn_right_far` → `portrait_turn_right_far.png` | head turned about 40 degrees to the other side, mirroring 19 |
| 21 | `source_tilt_up` → `portrait_tilt_up.png` | chin raised, head tilted back about 15 degrees, eyes still looking at the camera |
| 22 | `source_tilt_down` → `portrait_tilt_down.png` | chin lowered, head tilted forward about 15 degrees, eyes still looking at the camera up through the brows |

### Building each prompt

Take the shared preamble above, and end it with the sentence from the table.
For example, number 4:

```
Use the attached photograph as the exact reference for this person's identity.
Produce a new photorealistic photograph of the SAME person.

Keep these completely unchanged: face shape and bone structure, eye colour,
eye spacing, nose shape and size, lip shape, skin tone and texture, facial
hair, hairstyle and hairline, lighting direction and colour temperature,
background, camera distance, framing, and how large the head is within the
frame.

Do not add glasses, makeup, filters, or stylisation. Photorealistic, sharp
focus, plain background. Square 1:1 aspect ratio, 1024x1024.

The only change is: lips rounded and pushed forward as when saying "oh" or
"you", a small round opening. Head stays facing forward, eyes look at the
camera.
```

End every prompt in tiers 1 and 2 with **"Head stays facing forward, eyes look
at the camera"**, so the only thing that varies is the one thing asked for. For
tier 3, end with **"Neutral expression, mouth closed"** instead — the head is
the variable there.

### What each one actually fixes

| Photograph | Without it |
|---|---|
| closed smile | a warm smile borrows the teeth photograph and looks toothy |
| "oh" and "ee" | every vowel looks like the same open mouth |
| pressed lips | "m" and "b" do not close the mouth |
| gaze, four ways | the avatar stares straight ahead while you read your screen |
| squint, wide eyes | the eyes have one size |
| brows down | only surprise is available, never concentration |
| far turns | past about 25 degrees the face stretches |
| tilts | nodding stretches the face vertically |

### Priority if the list is too long

3, 4, 6 (mouth shapes), then 10–13 (gaze), then 19–20 (far turns). Those nine
cover most of what is visible in an ordinary conversation.

---

## Checking what comes back

```powershell
python scripts/check_avatar_photos.py avatars/my_avatar/portrait.png <new-photo> ...
```

It reports, for every photo:

- whether a face was found at all;
- **identity**: proportions that should not change between photographs of one
  person — eye spacing against face width, nose length against face height,
  mouth width, jaw width;
- **framing**: how large the face is in the frame and where it sits, since a
  variant shot "closer" makes the avatar jump when the engine switches;
- **pose**: measured yaw, pitch and roll, which is how the turned photos are
  identified;
- **lighting**: mean skin colour, since a variant lit differently shows as a
  flicker on switch.

One honest limitation, stated by the tool itself: geometric ratios are a proxy
for identity, not face recognition. Under a large head turn, foreshortening
changes horizontal ratios legitimately — so for the turned photos the horizontal
identity numbers are expected to move, and the vertical ones are the ones to
read.
