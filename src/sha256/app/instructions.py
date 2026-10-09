"""The instructions a person reads inside the application (Menu ->
Instructions; day 16 night, his ask: "which button does what, fully, in
text -- and that will be it").

One text, in sections: what the computer and the camera must be, the
first run, every control on the bar and in the Menu, what the status
line says, the calls, what to do when something goes wrong, and what
stays on this computer. Written from the code as it is -- every name here
is the name on the screen -- so the dialog and the USER_GUIDE say the same
thing. Kept as plain text: the dialog shows it as it is, and a person can
select and copy any of it.
"""

from __future__ import annotations

__all__ = ["INSTRUCTIONS_SECTIONS", "instructions_text"]

_WHAT_YOU_NEED = """\
1. WHAT YOU NEED

Computer
  - Windows 10 (64-bit, version 21H2 or newer) or Windows 11.
  - An NVIDIA graphics card with about 6 GB of memory free: an RTX 3060
    at the least, an RTX 4060 or better is comfortable. Without an NVIDIA
    card the application still opens and runs the plain photograph
    avatar, which is a weaker picture.
  - A processor of the last four years with six cores or more (the face
    tracker runs on the processor), 16 GB of memory, 6 GB of free disk.
  - OBS Studio installed once (free). It registers the "OBS Virtual
    Camera" device the application publishes to. OBS itself does not have
    to be open -- and must not have its own virtual camera started.

Camera
  - Any camera at 720p 30 fps at the least; 1080p 30 is fine.
    A Sony ZV-E10 works over USB in USB Streaming mode (NTSC for 30 fps,
    Smartphone Connect off, USB Connection on Auto, then USB Streaming ->
    Execute, and plug the cable in after that; a charge-only cable shows
    no camera). A good webcam of the Logitech C920 class works too.
  - At eye height, the face about a third of the picture's height.
  - Soft, even light on your face from the front. No window or lamp
    behind you: a shadow across the face puts the tracker's brows along
    the shadow.
  - Close Discord and anything else that holds the camera before Start.
"""

_FIRST_TIME = """\
2. THE FIRST TIME

  1. Start the application. If Windows warns about an unknown publisher,
     choose More info -> Run anyway; the build is not signed, on purpose.
  2. Pick your Camera and an Avatar on the bar, press Start.
  3. Sit still and look at the camera for a second. The status line
     counts the frames and says when the avatar has centred on you.
  4. Menu -> Calibrate me. Say yes, follow the large instructions for
     about a minute and a half (sit still, blink, close your eyes, open
     your mouth, talk, smile, raise your eyebrows, glance left and right,
     turn your head). Each one is read aloud and shown with a few seconds
     to get ready, then a countdown while you do it. At the end the
     application asks your name, keeps your profile under it and reloads
     the avatar with it. This is what makes the avatar follow YOUR face
     rather than someone else's.
  5. Every time you sit down: Menu -> Person -> your name.
  6. Tick Virtual camera. In your call application choose "OBS Virtual
     Camera" as the camera (see section 6).
"""

_THE_BAR = """\
3. THE BAR (along the bottom)

  Start           Opens the camera, starts the tracking and the avatar.
                  The preview switches from the background picture to
                  what the other side will see.
  Stop            Stops everything. The camera is released, the preview
                  goes back to the background picture.
  Camera          Which camera to use. Pick it before Start.
  Avatar          Which face the other side sees. Picking one loads it
                  (a few seconds on the graphics card).
  Virtual camera  Tick it to publish the avatar as a webcam named "OBS
                  Virtual Camera". Greyed out when no virtual camera
                  device is installed (install OBS Studio once). It can
                  be ticked before or after Start.
  Camera battery died
                  The red button. Blacks the published picture with that
                  text, as if the camera's battery went -- for a moment
                  you do not want seen. Press again to come back. The
                  same from any window: Ctrl+Alt+B. The application does
                  the same on its own when it loses your face for more
                  than a moment, and comes back when the face is back.
                  Lost while your head was turned away, the last picture
                  stays for eight seconds first -- a look at someone in
                  the room, not a dead camera.
  (status line)   What the application wants you to know right now --
                  see section 5.
  (fps)           How many pictures a second the avatar is drawn at.
  Menu            Everything else -- see section 4.
"""

_THE_MENU = """\
4. THE MENU

  View -> Avatar              The preview shows the avatar (the product).
  View -> Tracking (no avatar)
                              The preview shows your own camera picture
                              with the tracking on it -- the answer to
                              "is it following me". The other side is
                              not affected.
  Dots                        Draw the tracked landmarks. Tracking view only.
  Mirror                      Mirror the preview, the way a mirror would.
                              Affects this preview only, never the
                              published picture.
  Avatar's hair               Draw the avatar's own hair over your head,
                              not only its face. Off, your own hair stays
                              and the avatar's face is joined under it at
                              your hairline (read from your face in the
                              first second, and from your calibration
                              once you have one). On, the avatar's hair covers
                              yours as far as it reaches; your own hair
                              past its edge stays yours, and hair the
                              picture cuts (long, or very tall) fades
                              out. Try it and look; it can look worse
                              than off on some pairs of faces.
  Quality -> Native / 1080p / 720p / 480p / 360p
                              The quality of the published picture. Lower
                              hides small defects of the avatar, like a
                              cheaper webcam would: 480p looks like an
                              ordinary webcam, 360p like a poor line.
                              Native is the camera's own size.
  That looked wrong           Choose it DURING anything that looks wrong.
                              It saves the last ten seconds of tracking
                              (numbers, no video), the frame on screen
                              and the camera frame at that instant under
                              captures/ on this computer, so the moment
                              can be studied and fixed.
  Re-centre                   Learn your resting face again -- if you
                              moved your chair or started mid-word. Look
                              at the camera for a second.
  Person -> Nobody / names    Whose profile the avatar follows. Calibrate
                              me adds you to this list; choose your name
                              when you sit down. Nobody runs the avatar
                              with its own values.
  Calibrate me                The guided minute and a half of section 2,
                              measured into your own profile. The session
                              is recorded (your camera frames and their
                              tracking) under recordings/ on this
                              computer only, after you say yes; you can
                              delete that folder at any time. While it
                              runs the entry reads "Stop calibrating":
                              choosing it cancels and deletes the
                              recording.
  Add avatar                  Makes a new avatar from one photograph: a
                              window first tells what the photograph must
                              be, with prompts for an image generator,
                              then the file is chosen, checked and added
                              to the Avatar list. The avatar is kept on
                              this computer only, in the avatars folder.
  Delete avatar               Removes the avatar chosen in the bar from
                              this computer, for good --
                              its photograph, settings and profile. It
                              asks first; stop the application before.
  Instructions                This text.
"""

_STATUS_LINE = """\
5. THE STATUS LINE

  "Sit still and look at the camera to centre the avatar (n/15)"
        The avatar is learning your resting face from 15 still frames.
        Talking, nodding or reading the screen keeps it waiting.
  "Face not found -- holding the avatar"
        The tracker cannot see enough of a face. The avatar is held
        where it was; the camera is never passed through.
  "Face not found -- turned away, the last frame is held for the other side"
        The face was lost with the head turned well to the side or
        down. The other side sees the last picture, still, for up to
        eight seconds; the avatar is back the moment the face is.
  "Calibrating -- follow the instruction above"
        Calibrate me is running.
  "BLACK SCREEN -- the other side sees 'Camera battery died' (...)"
        The black screen is on: by the button, or because the face was
        lost.
  "Profile: <name>"
        Whose profile is in force, when nothing else needs saying.
  "Stopped: ..."
        The pipeline stopped on an error; the text says which.
"""

_ON_A_CALL = """\
6. ON A CALL

  Choose "OBS Virtual Camera" as the camera:
    Google Meet   the gear icon -> Video -> Camera.
    Zoom          Settings -> Video -> Camera.
    Microsoft Teams
                  Settings -> Devices -> Camera.
    Discord       User Settings -> Voice & Video -> Camera. Discord holds
                  the camera it last used: close it before Start, start
                  the application, then open Discord.
  Press Start here and tick Virtual camera before joining.
  Mirror on the preview does not change what the other side sees.
  Re-centre if you moved your chair. The red button (or Ctrl+Alt+B) for a
  moment you do not want seen. Quality 480p if the picture looks too
  clean to be a webcam.
  Turning your head far to the side, or looking well down, the avatar's
  head stops at the angle a photograph can still be drawn at (about 45
  degrees sideways, 30 down) while your own goes on; the face keeps
  blinking and, near the side, calms. If the tracker loses your face
  altogether while you are turned away, the other side sees the last
  picture, still, for up to eight seconds, then the black screen; the
  application keeps looking for the face in a close-up around where it
  was, and the avatar is back the moment it is found.
"""

_TROUBLE = """\
7. IF SOMETHING GOES WRONG

  "delivered no frames" at Start
        Another program holds the camera (Discord, Zoom's preview, the
        camera app). Close it.
  The camera is not in the list
        A charge-only cable, or Smartphone Connect is on (Sony).
  "no virtual camera device"
        Install OBS Studio once; or OBS is running with its own virtual
        camera on -- stop that one.
  "no usable NVIDIA GPU" at start
        The photograph avatar is running instead; the neural avatar needs
        an NVIDIA card.
  The avatar stays a still picture after Start
        It is waiting to centre on you: sit still and look at the camera
        for a second (the status line counts to 15).
  The avatar's mouth or brows do not follow you
        Check Menu -> Person shows YOUR name; if it does, run Calibrate
        me again, sitting where you sit on calls, with the light you use.
  The application says it has been switched off by its owner
        The owner's courtesy switch; nothing is wrong with your machine.
"""

_PRIVACY = """\
8. WHAT STAYS ON THIS COMPUTER

  Nothing leaves this computer and nothing is installed. Your camera is
  never passed through to the other side. Two things hold your face and
  are yours to delete: recordings/ (Calibrate me sessions) and captures/
  (That looked wrong moments). Your measured profile is under
  configs/drivers/ -- numbers, no picture.
"""

INSTRUCTIONS_SECTIONS: tuple[str, ...] = (
    _WHAT_YOU_NEED,
    _FIRST_TIME,
    _THE_BAR,
    _THE_MENU,
    _STATUS_LINE,
    _ON_A_CALL,
    _TROUBLE,
    _PRIVACY,
)


def instructions_text() -> str:
    """The whole text, the sections in order, one blank line between."""
    return "SHA256 -- INSTRUCTIONS\n\n" + "\n".join(INSTRUCTIONS_SECTIONS)
