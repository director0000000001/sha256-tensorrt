"""What "Add avatar" needs from a photograph, and the prompts that produce
one (day 16, his ask: the Add avatar window shows the instruction, in
English, with a prompt for each photograph; day 23, his ideas: the
prompts carry the person's own skin colour, and the avatar is SHAVED).

One photograph makes a neural avatar. Everything the importer checks
(``sha256.avatar.importer``) and everything the days of his eye found
about portraits is in :data:`PHOTO_REQUIREMENTS`; the prompts are written
to be pasted into an image generator (Gemini, ChatGPT) as they are.

Three prompts, in order. The first makes the face, with hair, from words
alone. The second takes that photograph and shaves the head: the avatar
the application uses. The application joins the avatar's face to the
person's own forehead and keeps the person's own hair (day 21), so the
avatar's hair is never wanted -- and every hair fault of days 21-22 (the
pale patch of the roots on the forehead, the smear over the cheek at a
turn, hair showing above the forehead) came from hair the portrait had.
The face is made first and shaved after so that it stays the same face.
The third, optional, is the shaved one with the lips barely parted (day
6's finding that a closed-lip portrait has to invent teeth on a smile).

Every prompt says ``[your skin color]`` (:data:`SKIN_PLACEHOLDER`). The
window's "Read my skin from the camera" button replaces it with the
reading's words and code (``sha256.avatar.skin_tone``); without the
reading the person writes their own words there.
"""

from __future__ import annotations

from sha256.avatar.skin_tone import SkinReading, fill_prompt

__all__ = [
    "BALD_PROMPT",
    "PARTED_LIPS_PROMPT",
    "PHOTO_REQUIREMENTS",
    "PORTRAIT_PROMPT",
    "PROMPTS",
    "photo_guide_text",
]

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
  5. Best of all NO HAIR: a shaved head (prompt 2 below). Otherwise hair
     off the forehead. No hat, no glasses, no hand near the face.
  6. The face large in the picture: at least 350 px from forehead to chin
     (a 1024 x 1024 picture with the face filling about half its height).
  7. Room around the head: some picture above the head and, above all,
     BELOW THE CHIN -- the neck and the top of the shoulders visible.
  8. A plain, quiet background. Sharp focus, no filters, no stylisation.
  9. The skin in YOUR colour, as the camera sees it: every prompt says
     "[your skin color]" -- the button below fills it from the camera.

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

Skin colour: [your skin color]. Render the skin in exactly this colour,
under neutral white balance -- no warm or cool colour cast, no tan, no
retouching that changes the tone.

[Describe the person here: age, gender, hair, facial hair, face shape --
one or two sentences. Keep everything above unchanged.]
"""

BALD_PROMPT = """\
Use the attached photograph as the exact reference for this person's
identity. Produce a new photorealistic photograph of the SAME person.

Keep these completely unchanged: face shape, eye colour and spacing, nose,
lips, eyebrows, facial hair, lighting direction and colour, background,
camera distance, framing, and how large the head is within the frame.
The skin colour and texture stay exactly as they are: [your skin color].
No glasses, no filters, no stylisation. Square 1:1, 1024 x 1024.

The only change is: the head completely shaved -- no hair at all on the
scalp, no stubble on the scalp, no hairline anywhere; the skin of the
scalp the same colour as the forehead, smooth and natural, lit as evenly
as the face. Eyebrows, eyelashes and any facial hair stay exactly as they
are.
"""

PARTED_LIPS_PROMPT = """\
Use the attached photograph as the exact reference for this person's
identity. Produce a new photorealistic photograph of the SAME person.

Keep these completely unchanged: face shape, eye colour and spacing, nose,
the shaved head, facial hair, lighting direction and colour, background,
camera distance, framing, and how large the head is within the frame.
The skin colour and texture stay exactly as they are: [your skin color].
No glasses, no filters, no stylisation. Square 1:1, 1024 x 1024.

The only change is: the lips very slightly parted, relaxed, a thin dark
line between them, the teeth barely visible -- not a smile.
"""

# In the order they are made; the titles are the window's.
PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "PROMPT 1 -- the face, with hair (paste into Gemini or ChatGPT; fill in the last lines)",
        PORTRAIT_PROMPT,
    ),
    (
        "PROMPT 2 -- the SAME face, head shaved: THE AVATAR (attach the photograph from prompt 1)",
        BALD_PROMPT,
    ),
    (
        "PROMPT 3 -- optional: the shaved one with the lips barely parted (attach "
        "the photograph from prompt 2)",
        PARTED_LIPS_PROMPT,
    ),
)

_WHICH_ONE = """\
WHICH PHOTOGRAPH BECOMES THE AVATAR

Add the SHAVED one (prompt 2). The application joins the avatar's face to
your own forehead and keeps your own hair, so the avatar's hair is never
wanted: with none in the picture, none can show at the forehead or at a
turn. The face is made with hair first (prompt 1) and shaved after so
that it stays the same face. Each avatar is ONE photograph; the
parted-lips variant (prompt 3) can be added as a second avatar of the
same person and compared on a call.

YOUR SKIN COLOUR

Every prompt says "[your skin color]". With the camera running (Start),
press "Read my skin from the camera", look at the camera in the light
your calls will have, and hold still for a second: the reading's words
and colour code take the placeholder's place in all three prompts, and
the Copy buttons copy them filled. Without the reading, write your own
words there. When the photograph is added, the application reads its
skin colour too and says how far it is from yours.
"""


def photo_guide_text(reading: SkinReading | None = None) -> str:
    """The whole guide as one text: the requirements, the three prompts
    (with ``reading``'s words in place of the placeholder when given),
    then which photograph to add and how the skin colour is read."""
    parts = [PHOTO_REQUIREMENTS]
    for title, prompt in PROMPTS:
        parts.append(f"\n{title}\n\n{fill_prompt(prompt, reading)}")
    parts.append(f"\n{_WHICH_ONE}")
    return "".join(parts)
