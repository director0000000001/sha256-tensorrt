# sha256 — the one-page guide

You get a folder (or a USB drive) with `sha256.exe` and three folders beside
it: `avatars/`, `models/`, `configs/`. Keep them together. Nothing is
installed, nothing phones home, nothing leaves your computer.

## What you need

| | |
|---|---|
| Windows | 10 or 11, 64-bit |
| Graphics card | an NVIDIA card with about 6 GB free (RTX 3060 or better is comfortable). Without one the app still opens and runs the plain photograph avatar. |
| Camera | any webcam at 720p 30 fps. A Sony ZV-E10 works over USB in **USB Streaming** mode: set the camera to **NTSC** (for 30 fps), turn **Smartphone Connect off**, put USB Connection on **Auto**, then USB Streaming → Execute, and plug the cable in *after* that. Use a data cable (a charge-only cable shows no camera). |
| Virtual camera | install **OBS Studio** once (free). It registers the "OBS Virtual Camera" device the app publishes to; OBS itself does not have to be open — and must not have *its own* virtual camera started at the same time. |
| Light | on your face, from the front. A shadow across the face puts the tracker's brows along the shadow. |

Close Discord and anything else that holds the camera before you start.

## The first time

1. Start `sha256.exe`. If Windows warns about an unknown publisher, choose
   *More info → Run anyway* — the build is not signed, on purpose. The bar
   along the bottom has what a call needs: **Start**, **Stop**, the
   **Camera**, the **Avatar**, **Virtual camera** and the red button.
   Everything else is under **Menu**, at the bar's right end.
2. Pick your **Camera** and an **Avatar**, press **Start**. Sit still and
   look at the camera for a second: the status line counts the frames and
   says when the avatar has centred on you.
3. Open **Menu → Calibrate me**. Say yes, follow the large instructions for
   about a minute and a half (sit still, blink, open your mouth, talk,
   smile, raise your eyebrows, glance left and right, turn your head).
   Each one is read aloud and shown with a few seconds to get ready, then
   a countdown while you do it. At the end the app asks your name, keeps
   your profile under it (`configs/drivers/`), reloads the avatar with it
   and shows what it measured. This is what makes the avatar follow
   *your* face rather than the owner's. The session is recorded under
   `recordings/` on this computer only — delete that folder whenever you
   like.
4. Every time you sit down: **Menu → Person → your name**. The status line
   reads "Profile: *your name*" while nothing else needs saying. Someone
   else at the same computer chooses theirs; **Nobody** runs the avatar
   with its own values.
5. Tick **Virtual camera**. In Zoom, Meet, Teams or Discord choose
   **OBS Virtual Camera** as the camera.

## On a call

- **Menu → Re-centre** if you moved your chair: look at the camera for a second.
- **Camera battery died** (the red button, or **Ctrl+Alt+B** from any
  window) blacks the camera with that text, as if the battery went. Press
  again to come back. The app does the same on its own when it loses your
  face for more than a moment.
- **Menu → Quality**: 480p makes the picture look like an ordinary webcam and
  hides small defects; 360p looks like a poor line.
- **Menu → That looked wrong**: choose it *during* anything that looks wrong. It
  saves the last ten seconds of tracking (numbers, no picture) plus the
  frame on screen and the camera frame at that instant, under `captures/`,
  so the moment can be studied.

## Adding an avatar (the owner only)

**Menu → Login** takes the owner's passphrase; then **Add avatar** appears in
the menu. A
good photograph: one face looking straight at the camera, eyes open, lips
closed or barely parted (a grin becomes the avatar's resting face), even
light, no glasses, hair off the forehead, and room below the chin. The app
checks all of these and says what it found; it pads and scales the picture
itself and keeps the original beside it.

## If something does not work

| You see | Do |
|---|---|
| "delivered no frames" at Start | another program holds the camera (Discord, Zoom's preview, the camera app). Close it. |
| The camera is not in the list | a charge-only cable, or Smartphone Connect is on (Sony). |
| "no virtual camera device" | install OBS Studio once; or OBS is running with its own virtual camera on — stop that one. |
| "no usable NVIDIA GPU" at start | the photograph avatar is running instead; the neural avatar needs an NVIDIA card. |
| The avatar's mouth or brows do not follow you | check **Menu → Person** shows *your* name; if it does, run **Menu → Calibrate me** again, sitting where you sit on calls, with the light you use. |
| The avatar stays a still picture after Start | it is waiting to centre on you: sit still and look at the camera for a second (the status line counts to 15). Talking, nodding or reading the screen keeps it waiting. |
| The app says it has been switched off by its owner | the owner's courtesy switch; nothing is wrong with your machine. |

## Privacy

Everything runs on your computer. The recordings and captures are your
face; they are yours to delete (`recordings/`, `captures/`). The app makes
no network request unless the owner has set the courtesy switch's address
in `configs/default.toml` — then it fetches one small text file at start
and once an hour, and nothing else.
