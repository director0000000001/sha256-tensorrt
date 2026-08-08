# sha256 — Privacy

sha256 processes a webcam feed — among the most sensitive data streams a
desktop application can touch. This document states what happens to it.

**Scope note:** this describes the design and the current foundation build,
which does not yet capture from a real camera. Statements about the current
build are verifiable by reading the code; statements about the target design are
commitments, marked as such.

---

## 1. The core commitment

**Webcam frames never leave the machine.**

Not to a server, not to a log file, not to a crash report, not to an analytics
endpoint. There is no code path that transmits a frame, and adding one would be
a privacy regression requiring an explicit decision recorded in
`docs/decisions/`.

Everything the real-time pipeline needs — capture, tracking, animation,
rendering, output — is designed to run locally. **sha256 does not require a
cloud service to function, and is not intended to ever require one.**

---

## 2. What the current build actually does

Verifiable claims about the foundation build as it stands:

| | |
|---|---|
| Network requests made | **None.** There is no HTTP client, no socket, no telemetry SDK in the dependency tree. |
| Camera accessed | **None.** The only camera backend generates frames procedurally. |
| Data written to disk | Log files under `logs/` only. |
| Data read from disk | Configuration and avatar manifests. |
| Analytics / telemetry | **None.** |
| Crash reporting | **None.** |
| Auto-update | **None.** |
| Accounts / licensing checks | **None.** |

The runtime dependency list is deliberately short — PySide6, pydantic,
pydantic-settings, numpy, platformdirs — partly so this claim stays auditable. A
large dependency tree makes "we send nothing" hard to verify.

`SHA256_MODELS__ALLOW_DOWNLOADS` exists and defaults to `false`. Nothing
currently reads it to perform a download; it is a pre-declared gate so that when
model fetching is implemented, it is off unless explicitly enabled.

---

## 3. What is processed, and where it goes

| Data | Where it lives | Persisted | Leaves the machine |
|---|---|---|---|
| Webcam frames | Process memory | **No** | **No** |
| Facial landmarks / pose / expressions | Process memory | **No** | **No** |
| Rendered avatar frames | Process memory → virtual camera | **No** | Only to the application *you* point at the virtual camera |
| Avatar manifests and weights | Local disk | Yes | **No** |
| Configuration | Local disk | Yes | **No** |
| Performance metrics | Process memory | **No** | **No** |
| Logs | Local disk (`logs/`) | Yes, rotating | **No** |

### On the virtual camera

The rendered avatar is the one thing that *is* meant to leave — that is the
product. It goes to whichever application you select the sha256 camera in:
Meet, Zoom, Discord, OBS. **What that application does with it is governed by
its own privacy policy, not this one.**

The avatar frame contains your facial motion, not your face.

### On logs

Logs record lifecycle events and errors: components starting, avatars loading,
failures. They do **not** contain frames, landmarks, or expression data.

The real-time loop deliberately does not log per frame — partly because file I/O
in the capture path causes dropped frames, and partly because a per-frame log
would be a record of your facial movements sitting on disk.

Log files rotate at 5 MB with 3 backups. Unbounded logs are both a disk risk and
a privacy one.

---

## 4. Performance metrics stay in memory

The performance monitor collects frame timings, throughput, and resource
figures. All of it lives in process memory and is discarded on exit. It exists
to render the local debug panel and to support benchmark runs that explicitly
export data.

It is **not** telemetry. Nothing is aggregated, uploaded, or phoned home. The
word "metrics" in this codebase means "numbers on your screen".

---

## 5. Avatars are fictional

Every avatar in sha256 is a fictional, AI-generated identity. Manifests carry
an explicit `fictional: true` assertion.

This is a stated intent rather than a technical enforcement — the application
cannot verify that an image is not of a real person. It is recorded because it
governs what the project is for. **sha256 is not built for impersonating real
people**, and features that would primarily serve that purpose are out of scope.

Avatar assets you add are yours. They stay on your disk, are not uploaded, and
are not tracked in Git (`avatars/**/model/` is gitignored — for size, and
because redistribution terms differ per model).

---

## 6. Features that would need networking — and how they stay separated

Some plausible future features cannot be local. Each is listed with the boundary
it must respect.

| Feature | Why it needs the network | Required separation |
|---|---|---|
| **Software updates** | Fetching a new version | Separate process or explicit user action. Must never run inside the pipeline. Opt-in. |
| **Avatar packs / downloads** | Fetching avatar assets | Gated behind `models.allow_downloads`, off by default. User-initiated, never automatic. Downloads verified against manifest checksums before use. |
| **Model weight downloads** | First-run acquisition of large files | Same gate. Explicit consent, visible progress, checksum-verified. |
| **Licensing / activation** | Commercial distribution, if it ever happens | Must not gate the real-time pipeline. A network failure must never stop a video call already in progress. |
| **Optional cloud rendering** | Higher quality than local hardware allows | **Strictly opt-in, off by default, per-session.** Would transmit frames — the one thing this document otherwise rules out. Would require its own explicit consent flow and its own documentation. |

### The rule that makes this workable

**Networking is never part of the real-time path.**

The pipeline — capture, tracking, animation, rendering, output — must remain
fully functional with the machine offline, permanently. Any networked feature is
a separate subsystem that the pipeline does not call into and does not depend
on. If pulling the network cable can degrade the avatar, the boundary has been
violated.

This is architecturally enforceable: none of the component protocols in
`core/interfaces.py` has any notion of a network, and adding one would be a
visible change to a reviewed contract.

---

## 7. Your controls

| Concern | Control |
|---|---|
| Stop all processing | Stop button, or close the application |
| Camera access | Not yet implemented. When it is, capture starts only on explicit Start. |
| Where data is stored | `SHA256_AVATARS__LIBRARY_PATH`, `SHA256_MODELS__ROOT_PATH` |
| Log verbosity | `SHA256_LOG_LEVEL`; logs can be deleted at any time |
| Network access | `SHA256_MODELS__ALLOW_DOWNLOADS=false` by default |
| Remove everything | Delete the application directory and the user data directory |

sha256 stores nothing outside its own directories and writes nothing to the
registry.

---

## 8. Third-party components

The current runtime dependencies are PySide6 (Qt), pydantic, pydantic-settings,
numpy, and platformdirs. None performs network access as part of the operations
sha256 uses.

Future AI components — face trackers, avatar engines — run **local inference on
local weights**. Their licensing is analysed in
`docs/OPEN_SOURCE_EVALUATION.md`. Any component that required a network call to
function would be rejected on that basis alone; it would break the local-first
commitment regardless of its quality.

---

## 9. What this document does not claim

Stated so the commitments above are not read more broadly than they are meant:

- sha256 cannot control what the video-call application receiving the virtual
  camera does with the frames.
- It cannot prevent other software on your machine from accessing your webcam.
- It does not encrypt avatar assets or configuration at rest; they have the same
  protection as any other file in your user profile.
- Operating-system-level camera indicators and permissions are Windows's
  responsibility, and sha256 does not attempt to circumvent them.

---

## See also

- `docs/SECURITY.md` — threat model and security posture
- `docs/architecture/SYSTEM_ARCHITECTURE.md` — where data actually flows
