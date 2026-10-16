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
All of this page, and every button explained, is inside the app too:
**Menu → Instructions**.

## The first time

1. Start `sha256.exe`. If Windows warns about an unknown publisher, choose
   *More info → Run anyway* — the build is not signed, on purpose. A small
   window says what the start is doing. The very first start on a computer
   also prepares the fast engine for its graphics card: one to three
   minutes, once, the seconds counted on that window. Every later start
   takes about twenty seconds. The bar
   along the bottom has what a call needs: **Start**, **Stop**, the
   **Camera**, the **Avatar**, **Virtual camera** and the red button.
   Everything else is under **Menu**, at the bar's right end.
2. Pick your **Camera** and an **Avatar**, press **Start**. Sit still and
   look at the camera for a second: the status line counts the frames and
   says when the avatar has centred on you.
3. Open **Menu → Calibrate me**. Say yes. An oval appears on the picture:
   sit so your head fills it — not too far from the camera, not too close,
   in the middle, facing it. It turns green and says *Good. Hold still.*;
   a second later the instructions begin. Follow them for about four and a half
   minutes (sit still, blink, close your eyes halfway and then fully,
   open your mouth a little and then wide, read a short text aloud, smile
   a little, smile, smile with your teeth, laugh, raise your eyebrows a
   little and then high, glance left and right, turn your head). Each one
   is read aloud and shown for three seconds, then a big **3, 2, 1** is
   counted on the screen — start on the end of the count — then a
   countdown while you do it. For the reading, the text appears on the
   screen when the count ends: read all of it, at your own pace (20
   seconds). Keep your head in the green oval the whole time; if it
   leaves, a line under the instruction says what to fix. A movement too
   small to measure is asked **once more** right after it. At the end the
   app asks your name, keeps
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
  face for more than a moment. Lost while your head was turned away, the
  other side sees the last picture, still, for up to eight seconds first --
  a look at someone in the room, not a dead camera -- and the avatar is
  back the moment your face is.
- **Menu → Avatar's hair**: the avatar's own hair drawn over your head, not
  only its face. Off, your hair stays and the avatar's face is joined under it
  at your hairline -- read from your face in the first second (look at the
  camera), and from your calibration once you have one, so a shorter forehead
  gets its join lower. Try it and look: it can look worse than off on some
  pairs of faces, and your own hair past the avatar's edge stays yours.
- **Menu → Quality**: 480p makes the picture look like an ordinary webcam and
  hides small defects; 360p looks like a poor line, 240p like one about to drop.
- **Menu → That looked wrong**: choose it *during* anything that looks wrong. It
  saves the last ten seconds of tracking (numbers, no picture) plus the
  frame on screen and the camera frame at that instant, under `captures/`,
  so the moment can be studied.

## Adding and deleting an avatar (the owner only)

**Menu → Login** takes the owner's passphrase; then **Add avatar** and
**Delete avatar** appear in the menu. Add avatar makes a new avatar from
one photograph; Delete avatar removes the avatar chosen in the bar from
this computer, for good, after a question (stop first). Without the
passphrase neither is there. An avatar is a folder under `avatars/` next
to the application, on this computer only; nothing is uploaded. A
good photograph: one face looking straight at the camera, eyes open, lips
closed or barely parted (a grin becomes the avatar's resting face), even
light, no glasses, a SHAVED head (or at least hair off the forehead), and
room below the chin. The app checks all of these and says what it found;
it pads and scales the picture itself and keeps the original beside it.

The window that opens first carries three prompts for an image generator
(Gemini, ChatGPT), in the order they are made: the face with hair, from
words; the same face with the head shaved, made from that photograph --
**add this one as the avatar** (the app joins the avatar's face to your
own forehead and keeps your own hair, so the avatar's hair only gets in
the way); and, optional, the shaved one with the lips barely parted.
Every prompt says `[your skin color]`. With the camera running, press
**Read my skin from the camera**, look at the camera in the light your
calls will have, and hold still for a second: your skin's words and
colour code take the placeholder's place in all three prompts, and the
Copy buttons copy them filled. When the photograph is added, the app
reads its skin colour too and says how far it is from yours.

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
