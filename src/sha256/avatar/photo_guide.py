"""What "Add avatar" needs from a photograph, and the prompts that produce
one (day 16, his ask: the Add avatar window shows the instruction, in
English, with a prompt for each photograph).

One photograph makes a neural avatar. Everything the importer checks
(``sha256.avatar.importer``) and everything fifteen days of his eye found
about portraits is in :data:`PHOTO_REQUIREMENTS`; the prompts are written
to be pasted into an image generator as they are. The second prompt is
the one optional variant worth making -- the same person with the lips
barely parted, day 6's finding that a closed-lip portrait has to invent
teeth on a smile.
"""

from __future__ import annotations

__all__ = ["PARTED_LIPS_PROMPT", "PHOTO_REQUIREMENTS", "PORTRAIT_PROMPT", "photo_guide_text"]

PHOTO_REQUIREMENTS = """\
WHAT THE PHOTOGRAPH MUST BE

One person, one photograph. The application makes the avatar from it and
pads, scales and checks the picture itself; a good one has:

  1. The face looking STRAIGHT at the camera -- not turned, not tilted
     (up to about 5 degrees is fine; past 12 the picture is refused).
  2. Eyes OPEN, looking at the camera.
  3. Lips CLOSED or barely parted, a neutral, relaxed face. A smile or an
     open mouth becomes the avatar's resting face and cannot be undone.
  4. Even, soft light on the whole face, from the front. No hard shadows,
     no half-lit face, no strong colour cast.
  5. Hair off the forehead, no hat, no glasses, no hand near the face.
  6. The face large in the picture: at least 350 px from forehead to chin
     (a 1024 x 1024 picture with the face filling about half its height).
  7. Room around the head: some picture above the hair and, above all,
     BELOW THE CHIN -- the neck and the top of the shoulders visible.
  8. A plain, quiet background. Sharp focus, no filters, no stylisation.

Square, 1024 x 1024 or larger, PNG or JPG.
"""

PORTRAIT_PROMPT = """\
A photorealistic studio portrait photograph of a person, head and
shoulders, looking straight into the camera. Neutral, relaxed expression:
lips closed, eyes open, eyebrows at rest. Soft, even, front lighting on the
whole face, no hard shadows. Hair off the forehead. No glasses, no hat, no
jewellery, no hands. The head fills about half of the picture's height,
with clear room above the hair and below the chin: the neck and the top of
the shoulders are visible. Plain, uniform light-grey background. Sharp
focus on the face, natural skin texture, no beauty filter, no stylisation.
Square 1:1 aspect ratio, 1024 x 1024.

[Describe the person here: age, gender, skin tone, hair, facial hair,
face shape -- one or two sentences. Keep everything above unchanged.]
"""

PARTED_LIPS_PROMPT = """\
Use the attached photograph as the exact reference for this person's
identity. Produce a new photorealistic photograph of the SAME person.

Keep these completely unchanged: face shape, eye colour and spacing, nose,
skin tone and texture, facial hair, hairstyle and hairline, lighting
direction and colour, background, camera distance, framing, and how large
the head is within the frame. No glasses, no filters, no stylisation.
Square 1:1, 1024 x 1024.

The only change is: the lips very slightly parted, relaxed, a thin dark
line between them, the teeth barely visible -- not a smile.
"""


def photo_guide_text() -> str:
    """The whole guide as one text: the requirements, then the prompts."""
    return (
        PHOTO_REQUIREMENTS
        + "\nPROMPT 1 -- the portrait (paste into the image generator, fill in the last lines)\n\n"
        + PORTRAIT_PROMPT
        + "\nPROMPT 2 -- optional: the same person with the lips barely parted "
        "(attach the portrait from prompt 1)\n\n"
        + PARTED_LIPS_PROMPT
        + "\nEach avatar is ONE photograph. Add the portrait first; the parted-lips "
        "variant can be added as a second avatar of the same person and compared "
        "on a call.\n"
    )
