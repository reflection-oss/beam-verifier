# Mirror runtime

The wheel carries unmodified shared libraries from Debian Bookworm packages:

- glibc (`libc6`), version `2.36-9+deb12u14`: the dynamic loader, libc,
  libm, libpthread and libdl. See `glibc-copyright.txt` and `LGPL-2.1.txt`.
- GCC (`libgcc-s1`), version `12.2.0-14+deb12u1`: libgcc_s. See
  `gcc-copyright.txt` (including the GCC Runtime Library Exception) and `GPL-3.txt`.

Exact binary URLs and SHA-256 checksums are in `pyproject.toml`. Corresponding
Debian source packages and build metadata are available alongside the binaries:

- https://deb.debian.org/debian/pool/main/g/glibc/glibc_2.36-9+deb12u14.dsc
- https://deb.debian.org/debian/pool/main/g/gcc-12/gcc-12_12.2.0-14+deb12u1.dsc

The copyright notices are copied from the matching Debian `libc6` and
`gcc-12-base` packages. The license texts are the standard GNU LGPL 2.1 and GPL 3.

The runtime is isolated from the task image's system libraries. The adapter
invokes its loader with an explicit library directory; it does not install
Debian packages or set a process-wide `LD_LIBRARY_PATH` for the task's tools.
