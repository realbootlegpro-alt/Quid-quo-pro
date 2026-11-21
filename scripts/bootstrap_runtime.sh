#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/.tools"
BLENDER_DIR="${TOOLS_DIR}/blender"
BLENDER_BIN="${BLENDER_DIR}/blender"
PYDEPS_DIR="${ROOT_DIR}/.pydeps"

mkdir -p "${TOOLS_DIR}" "${PYDEPS_DIR}"

if [ ! -e /usr/local/bin/python ] && command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python || true
fi

# 1. Ensure OS Xvfb & OpenGL/Mesa shared libraries are installed
NEED_APT=0
if ! command -v xvfb-run >/dev/null 2>&1; then
  NEED_APT=1
fi
if [ ! -f /usr/lib/x86_64-linux-gnu/libGL.so.1 ] && ! ldconfig -p 2>/dev/null | grep -q 'libGL\.so\.1'; then
  NEED_APT=1
fi

if [ "${NEED_APT}" = "1" ] && command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  dpkg --configure -a --force-confdef --force-confold || true
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    xvfb xauth libgl1 libegl1 libglx0 libgl1-mesa-dri libglu1-mesa \
    libxi6 libxrender1 libxxf86vm1 libxfixes3 libxkbcommon0 \
    libsm6 libice6 libxext6 libxtst6 python3-pip xz-utils curl ca-certificates
fi

# 2. Ensure outer Python dependencies exist in persistent .pydeps
if ! PYTHONPATH="${PYDEPS_DIR}" python3 -c "import PIL, faster_whisper, av, rapidfuzz, ddgs, httpx" >/dev/null 2>&1; then
  if ! python3 -m pip --version >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq python3-pip
  fi
  python3 -m pip install --break-system-packages --no-cache-dir -q --upgrade --target "${PYDEPS_DIR}" -r "${ROOT_DIR}/requirements.txt"
fi

SITE_PKG="$(python3 -c 'import site; print((site.getsitepackages() or ["/usr/local/lib/python3.11/dist-packages"])[0])' 2>/dev/null || true)"
if [ -n "${SITE_PKG}" ] && [ -d "${SITE_PKG}" ]; then
  echo "${PYDEPS_DIR}" > "${SITE_PKG}/persephone_pydeps.pth" || true
fi

# 3. Ensure Blender 4.5.x LTS exists in persistent .tools/blender
if [ ! -x "${BLENDER_BIN}" ]; then
  if [ -x /opt/blender/blender ]; then
    cp -a /opt/blender "${BLENDER_DIR}"
  else
    TMP_TAR="${TOOLS_DIR}/blender.tar.xz"
    BLENDER_URL="https://download.blender.org/release/Blender4.5/blender-4.5.16-linux-x64.tar.xz"
    curl -fsSL -o "${TMP_TAR}" "${BLENDER_URL}"
    rm -rf "${BLENDER_DIR}.tmp"
    mkdir -p "${BLENDER_DIR}.tmp"
    tar -xJf "${TMP_TAR}" -C "${BLENDER_DIR}.tmp" --strip-components=1
    rm -f "${TMP_TAR}"
    rm -rf "${BLENDER_DIR}"
    mv "${BLENDER_DIR}.tmp" "${BLENDER_DIR}"
  fi
fi

if [ -x "${BLENDER_BIN}" ] && [ ! -e /usr/local/bin/blender ]; then
  ln -sf "${BLENDER_BIN}" /usr/local/bin/blender || true
fi
