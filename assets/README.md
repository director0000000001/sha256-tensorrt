# Assets

Application chrome that ships with the build — distinct from `avatars/`, which
is user data with its own lifecycle.

```
assets/
  icons/      application and UI icons
  samples/    sample media for tests and demos
  avatars/    reserved for bundled sample avatars, if any ever ship
```

Keep files here small and committable. Anything large, generated, or
license-restricted belongs in `models/` or `avatars/`, both of which are
gitignored.

## Window background

Put an image at **`assets/background.jpg`** and the application window uses it.
No configuration change is needed; if the file is absent the window falls back
to a plain colour, which is why the default configuration can point at a path
that may not exist.

| | |
|---|---|
| Recommended size | **1920×1080** (16:9) |
| Format | JPG or PNG |
| Framing | Scaled to *cover* the window and cropped from the centre — keep anything important away from the edges |
| Brightness | Darkened by `ui.background_dim` (default 0.55) so the controls stay readable. Lower it in `configs/local.toml` for a darker photo |

The image is not committed: a background is the owner's choice, not the
project's.
