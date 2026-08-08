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

Empty for now — the foundation shell is deliberately unstyled and uses no icons.
