# Security

Generated kernels and Python extensions are executable code. The evaluator runs
them in a subprocess for timeout handling, **not in a security sandbox**. Run
untrusted generations in a disposable environment with restricted filesystem,
network and credential access. A timeout is not a containment boundary.

Keep API credentials in a private `.env` file or process environment. Generated
launcher scripts do not embed key values. Detached runs should read a private
`.env` through the configured `--env-file` path. Run artifacts can contain code,
logs and local paths and are excluded from Git by default.

Do not post credentials or exploit details in public issues. Use GitHub private
vulnerability reporting if available, or request a private contact without
including sensitive details. Revoke exposed credentials promptly. Removing a
file from a branch does not remove copies in Git history or existing clones.
