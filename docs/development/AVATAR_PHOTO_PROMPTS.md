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

### 1 — Smile with teeth

```
The person is smiling naturally with the mouth open, showing the upper teeth
clearly and the lower teeth slightly. A genuine relaxed smile, not forced. The
eyes stay open and look directly at the camera. Head stays facing forward.
```

*Fixes: no teeth when smiling.*

### 2 — Eyes closed

```
Both eyes are fully and naturally closed, eyelids relaxed as in a normal blink.
The mouth stays closed with a neutral expression. Head stays facing forward.
```

*Fixes: the avatar's eyes not closing when you blink.*

### 3 — Head turned one way

```
The head is turned about 25 degrees to one side, as in a three-quarter portrait,
so that one cheek is more visible and the nose points slightly away from centre.
The eyes still look directly at the camera. Neutral expression, mouth closed.
The head stays upright — no tilt, no nod.
```

*Fixes: alignment breaking past ~15° of rotation.*

### 4 — Head turned the other way

```
The head is turned about 25 degrees to the opposite side from a three-quarter
portrait, mirroring the other three-quarter view, so the other cheek is more
visible. The eyes still look directly at the camera. Neutral expression, mouth
closed. The head stays upright — no tilt, no nod.
```

### 5 — Mouth open, speaking (optional but useful)

```
The mouth is open as if saying "ah" in the middle of a sentence, jaw dropped,
upper and lower teeth partly visible, tongue not prominent. Relaxed, natural,
not exaggerated. The eyes stay open and look directly at the camera. Head stays
facing forward.
```

*Fixes: the mouth interior during speech, which the smile photo alone does not
cover.*

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
