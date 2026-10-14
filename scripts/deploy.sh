#!/usr/bin/env bash
#
# deploy.sh — install tinycoder-openai build artifacts to the paths described by
# a JSON deploy manifest (default: deploy/deploy.config.json).
#
#   ./scripts/deploy.sh                        install to the configured paths (needs root)
#   ./scripts/deploy.sh --dry-run              show the actions, change nothing
#   ./scripts/deploy.sh --destdir /tmp/pkg     stage the tree under DIR (no root needed)
#   ./scripts/deploy.sh --prefix /srv/tinycoder  override the ${prefix} manifest variable
#   ./scripts/deploy.sh --user tc --group tc   override the artifact owner
#   ./scripts/deploy.sh --no-service           copy files only, skip systemd
#
# The manifest maps every artifact (the Go binary, the native bridge .so, the
# JSON config and the systemd unit) to a destination path, mode, owner and
# behaviour flags, and describes how the systemd unit should be
# reloaded/enabled/restarted. Destination paths may reference ${prefix},
# ${config_dir} and ${systemd_dir}, which are resolved from the manifest and can
# be overridden on the command line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="$ROOT/deploy/deploy.config.json"
DESTDIR=""
PREFIX_OVERRIDE=""
USER_OVERRIDE=""
GROUP_OVERRIDE=""
DRY_RUN=0
SKIP_SERVICE=0

note() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
    cat <<'EOF'
deploy.sh — install tinycoder-openai artifacts the paths in a deploy manifest

Usage: scripts/deploy.sh [options]

Options:
  --config PATH    Deploy manifest (default: deploy/deploy.config.json)
  --destdir DIR    Prepend DIR to every destination (staging tree; no root needed)
  --prefix DIR     Override the ${prefix} manifest variable
  --user NAME      Override/define the artifact owner
  --group NAME     Override/define the artifact owner group
  --no-service     Copy files only; do not touch systemd
  --dry-run, -n    Print the actions without performing them
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2"; shift 2 ;;
        --destdir)       DESTDIR="$2"; shift 2 ;;
        --prefix)        PREFIX_OVERRIDE="$2"; shift 2 ;;
        --user)          USER_OVERRIDE="$2"; shift 2 ;;
        --group)         GROUP_OVERRIDE="$2"; shift 2 ;;
        --no-service)    SKIP_SERVICE=1; shift ;;
        --dry-run|-n)    DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$CONFIG" != /* ]]; then
    CONFIG="$ROOT/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "deploy: manifest not found: $CONFIG" >&2
    exit 1
fi

# manifest_rows emits one record per line, with fields joined by the ASCII unit
# separator (0x1f). It is NOT tab-separated on purpose: empty fields (e.g. an
# artifact with no explicit owner) must survive parsing, and bash's IFS collapses
# adjacent tab/space delimiters but keeps adjacent non-whitespace delimiters.
#   setting <key> <value>
#   artifact <source> <dest> <mode> <owner> <required> <if_missing>
# It uses jq when available and falls back to python3, so the deploy host needs
# only one of the two.
manifest_rows() {
    if have jq; then
        jq -r '
          def b: if . then "true" else "false" end;
          (
            [["setting","prefix",(.prefix//"")],
             ["setting","config_dir",(.config_dir//"")],
             ["setting","systemd_dir",(.systemd_dir//"")],
             ["setting","user",(.user//"")],
             ["setting","group",(.group//"")],
             ["setting","service.unit",(.service.unit//"")],
             ["setting","service.reload",((.service.reload//false)|b)],
             ["setting","service.enable",((.service.enable//false)|b)],
             ["setting","service.restart",((.service.restart//false)|b)]]
            +
            [ (.artifacts//[])[] |
                ["artifact",(.source//""),(.dest//""),(.mode//"0644"),(.owner//""),
                 ((.required//false)|b),((.if_missing//false)|b)] ]
          ) | .[] | map(tostring) | join("\u001f")
        ' "$CONFIG"
    elif have python3; then
        python3 - "$CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)
svc = data.get("service") or {}


def b(v):
    return "true" if v else "false"


def emit(*fields):
    print("\x1f".join(str(f) for f in fields))


def setting(key, val):
    emit("setting", key, "" if val is None else val)


setting("prefix", data.get("prefix", ""))
setting("config_dir", data.get("config_dir", ""))
setting("systemd_dir", data.get("systemd_dir", ""))
setting("user", data.get("user", ""))
setting("group", data.get("group", ""))
setting("service.unit", svc.get("unit", ""))
setting("service.reload", b(svc.get("reload", False)))
setting("service.enable", b(svc.get("enable", False)))
setting("service.restart", b(svc.get("restart", False)))

for art in (data.get("artifacts") or []):
    emit("artifact",
         art.get("source", ""),
         art.get("dest", ""),
         art.get("mode", "0644"),
         art.get("owner", ""),
         b(art.get("required", False)),
         b(art.get("if_missing", False)))
PY
    else
        warn "need 'jq' or 'python3' to read $CONFIG"
        return 1
    fi
}

ROWS="$(manifest_rows)"

declare -A SET=()
declare -a A_SRC=() A_DEST=() A_MODE=() A_OWNER=() A_REQ=() A_IFMISS=()

while IFS=$'\x1f' read -r kind f1 f2 f3 f4 f5 f6; do
    if [[ -z "${kind:-}" ]]; then
        continue
    fi
    case "$kind" in
        setting)
            SET["$f1"]="$f2"
            ;;
        artifact)
            A_SRC+=("$f1")
            A_DEST+=("$f2")
            A_MODE+=("${f3:-0644}")
            A_OWNER+=("$f4")
            A_REQ+=("${f5:-false}")
            A_IFMISS+=("${f6:-false}")
            ;;
        *)
            warn "ignoring unknown manifest record: $kind"
            ;;
    esac
done <<<"$ROWS"

if [[ ${#A_SRC[@]} -eq 0 ]]; then
    echo "deploy: manifest $CONFIG defines no artifacts" >&2
    exit 1
fi

PREFIX="${PREFIX_OVERRIDE:-${SET[prefix]:-}}"
CONFIG_DIR="${SET[config_dir]:-/etc/tinycoder}"
SYSTEMD_DIR="${SET[systemd_dir]:-/etc/systemd/system}"
DEPLOY_USER="${USER_OVERRIDE:-${SET[user]:-}}"
DEPLOY_GROUP="${GROUP_OVERRIDE:-${SET[group]:-}}"

# subst replaces a literal token in a string (quoted pattern => literal match).
subst() { printf '%s' "${1//"$2"/$3}"; }

# expand resolves ${prefix}/${config_dir}/${systemd_dir} in a destination path.
expand() {
    local s="$1"
    s="$(subst "$s" '${prefix}' "$PREFIX")"
    s="$(subst "$s" '${config_dir}' "$CONFIG_DIR")"
    s="$(subst "$s" '${systemd_dir}' "$SYSTEMD_DIR")"
    printf '%s' "$s"
}

run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '  [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

maybe_chown() {
    local owner="$1" path="$2"
    if [[ -z "$owner" ]]; then
        return 0
    fi
    if [[ "$(id -u)" == 0 ]]; then
        run chown "$owner" "$path"
    elif [[ "$DRY_RUN" == 1 ]]; then
        printf '  [dry-run] chown %s %s\n' "$owner" "$path"
    else
        warn "not root; leaving owner of $path unchanged"
    fi
}

if [[ "$DRY_RUN" == 0 && -z "$DESTDIR" && "$(id -u)" != 0 ]]; then
    warn "not running as root; installs to system paths will likely fail"
    warn "re-run with sudo, or use --destdir DIR to stage the tree"
fi

if [[ -n "$DEPLOY_USER" && "$DRY_RUN" == 0 && "$(id -u)" == 0 ]]; then
    if ! id "$DEPLOY_USER" >/dev/null 2>&1; then
        warn "user '$DEPLOY_USER' does not exist; create it with:"
        warn "  useradd --system --no-create-home --shell /usr/sbin/nologin $DEPLOY_USER"
    fi
fi

installed=0
skipped=0

for i in "${!A_SRC[@]}"; do
    src="${A_SRC[$i]}"
    if [[ "$src" != /* ]]; then
        src="$ROOT/$src"
    fi

    dest="$(expand "${A_DEST[$i]}")"
    if [[ "$dest" != /* ]]; then
        dest="$ROOT/$dest"
    fi
    if [[ -n "$DESTDIR" ]]; then
        dest="$DESTDIR$dest"
    fi

    mode="${A_MODE[$i]}"
    owner="${A_OWNER[$i]}"
    if [[ -z "$owner" && -n "$DEPLOY_USER" ]]; then
        if [[ -n "$DEPLOY_GROUP" ]]; then
            owner="$DEPLOY_USER:$DEPLOY_GROUP"
        else
            owner="$DEPLOY_USER"
        fi
    fi
    req="${A_REQ[$i]}"
    ifmiss="${A_IFMISS[$i]}"

    if [[ ! -e "$src" ]]; then
        if [[ "$req" == "true" ]]; then
            warn "required artifact missing: $src"
            warn "build it first, e.g.: ./scripts/build.sh"
            exit 1
        fi
        warn "skipping (not built): $src"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ "$ifmiss" == "true" && -e "$dest" ]]; then
        note "keeping existing $dest"
        skipped=$((skipped + 1))
        continue
    fi

    note "install $src -> $dest (mode $mode${owner:+, owner $owner})"
    run install -D -m "$mode" "$src" "$dest"
    maybe_chown "$owner" "$dest"
    installed=$((installed + 1))
done

if [[ "$SKIP_SERVICE" == 1 ]]; then
    note "service management disabled (--no-service)"
elif [[ -n "$DESTDIR" ]]; then
    note "staged under $DESTDIR; skipping systemd (enable the unit on the target host)"
elif [[ -z "${SET[service.unit]:-}" ]]; then
    note "no service.unit in manifest; skipping systemd"
elif ! have systemctl; then
    note "systemctl not found; skipping systemd"
elif [[ "$(id -u)" != 0 ]]; then
    warn "not root; skipping systemctl for ${SET[service.unit]:-}"
else
    unit="${SET[service.unit]:-}"
    if [[ "${SET[service.reload]:-false}" == "true" ]]; then
        run systemctl daemon-reload
    fi
    if [[ "${SET[service.enable]:-false}" == "true" ]]; then
        run systemctl enable "$unit"
    fi
    if [[ "${SET[service.restart]:-false}" == "true" ]]; then
        run systemctl restart "$unit"
    fi
    if [[ "$DRY_RUN" == 0 ]]; then
        systemctl --no-pager --lines=0 status "$unit" >&2 || true
    fi
fi

note "done: $installed installed, $skipped skipped"
