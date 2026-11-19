#!/usr/bin/env bash
# agent_sandbox_audit.sh — read-only security-posture audit for agent sandboxes
#
# AUTHORIZATION: Run only inside environments you own or are explicitly
# authorized to assess. This is a posture scanner, not an exploitation tool.
#
# DESIGN PRINCIPLES
#   * Strictly read-only. No writes outside $TMP, no config changes, no payloads.
#   * Detect & report — never exfiltrate. IMDS/credential endpoints are tested
#     for REACHABILITY only; secret key material is never fetched or printed.
#     Env vars that look sensitive are reported by NAME with values redacted.
#   * Reachability != exploitation. Open management ports are reported; the
#     script never speaks the Docker/kubelet/API protocols to act on them.
#   * Portable: bash + coreutils only; every external tool is optional with a
#     /proc or /sys fallback. Safe to run in a blackbox container.
#
# OUTPUT: single JSON object on stdout (schema documented at bottom of file).

set -u
export LC_ALL=C
TMP="$(mktemp -d 2>/dev/null || echo /tmp)"
trap 'rm -rf "$TMP" 2>/dev/null' EXIT
TIMEOUT_BIN="$(command -v timeout || true)"

# output format: json (default) | table | both  (e.g. ./audit.sh --format=table)
FORMAT=json
for a in "$@"; do case "$a" in --format=table) FORMAT=table;; --format=both) FORMAT=both;; --format=json) FORMAT=json;; esac; done

# ---- helpers ---------------------------------------------------------------
j()  { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/ /g' | tr -d '\000-\037'; }
have(){ command -v "$1" >/dev/null 2>&1; }
run() { # run with a hard timeout; never hang the audit
  if [ -n "$TIMEOUT_BIN" ]; then "$TIMEOUT_BIN" "$1" "${@:2}" 2>/dev/null; else "${@:2}" 2>/dev/null; fi
}
FINDINGS=()  # each: "SEVERITY|id|evidence"
add() { FINDINGS+=("$1|$2|$(j "$3")"); }

# ===========================================================================
# 1. RUNTIME & KERNEL ISOLATION
# ===========================================================================
KERNEL="$(uname -a 2>/dev/null)"
VIRT="$(have systemd-detect-virt && systemd-detect-virt 2>/dev/null || echo unknown)"
CONTAINER="none"
[ -f /.dockerenv ]        && CONTAINER="docker"
[ -f /run/.containerenv ] && CONTAINER="podman/cri"
CGROUP1="$(head -n1 /proc/1/cgroup 2>/dev/null)"
echo "$CGROUP1" | grep -qiE 'docker|kube|containerd|lxc' && CONTAINER="${CONTAINER}+orchestrated"

# ---- isolation-class fingerprint (bare metal / VM / microVM / gVisor / container)
HYPERVISOR_FLAG=no; grep -q '^flags.* hypervisor' /proc/cpuinfo 2>/dev/null && HYPERVISOR_FLAG=yes
SYS_VENDOR="$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null)"
PRODUCT="$(cat /sys/class/dmi/id/product_name 2>/dev/null)"
PCI_PRESENT=yes; [ -z "$(ls /sys/bus/pci/devices 2>/dev/null)" ] && PCI_PRESENT=no
VIRTIO_MMIO=no; ls /sys/bus/virtio/devices 2>/dev/null | grep -q . && [ "$PCI_PRESENT" = no ] && VIRTIO_MMIO=yes
# gVisor/runsc exposes a distinctive sentry fingerprint
GVISOR=no
{ dmesg 2>/dev/null | grep -qi gvisor; } || grep -qi 'gvisor\|runsc' /proc/version 2>/dev/null && GVISOR=yes
# gVisor also fakes a fixed kernel string + a sparse /proc; treat the dmesg/version tell as authoritative
if [ "$GVISOR" = yes ]; then
  ISO_CLASS="gvisor-sandbox"
elif echo "$CONTAINER" | grep -qv '^none$' && [ "$CONTAINER" != none ]; then
  ISO_CLASS="container:$CONTAINER"
elif [ "$VIRT" = kvm ] || [ "$VIRT" = qemu ] || echo "$SYS_VENDOR$PRODUCT" | grep -qiE 'qemu|firecracker|kata|cloud-hypervisor'; then
  if echo "$SYS_VENDOR$PRODUCT" | grep -qi firecracker || { [ "$PCI_PRESENT" = no ] && [ "$VIRTIO_MMIO" = yes ]; }; then
    ISO_CLASS="microvm"            # Firecracker / Cloud Hypervisor / Kata: virtio-mmio, no PCI bus
  else
    ISO_CLASS="full-vm:${VIRT}"
  fi
elif [ "$HYPERVISOR_FLAG" = yes ]; then
  ISO_CLASS="virtualized:${VIRT}"
else
  ISO_CLASS="bare-metal-or-unknown"
fi

# kernel hardening sysctls (read via /proc, no sysctl binary needed)
rd() { cat "$1" 2>/dev/null || echo "n/a"; }
BPF_DIS="$(rd /proc/sys/kernel/unprivileged_bpf_disabled)"
USERNS="$(rd /proc/sys/kernel/unprivileged_userns_clone)"
PTRACE="$(rd /proc/sys/kernel/yama/ptrace_scope)"
[ "$USERNS" = "1" ]  && add MEDIUM unprivileged_userns_enabled "unprivileged_userns_clone=1 (user-namespace escape surface)"
[ "$BPF_DIS" = "0" ] && add MEDIUM unprivileged_bpf_enabled "unprivileged_bpf_disabled=0"

# security modules
SECCOMP="$(grep -i '^Seccomp:' /proc/self/status 2>/dev/null | awk '{print $2}')"
case "$SECCOMP" in 0|"") add HIGH seccomp_disabled "Seccomp=${SECCOMP:-absent} (no syscall filter)";; esac
APPARMOR="$( ([ -d /sys/kernel/security/apparmor ] && echo enabled) || echo absent)"
SELINUX="$( (have getenforce && getenforce) || rd /sys/fs/selinux/enforce)"

# ===========================================================================
# 2. CAPABILITIES, PRIVILEGES & SENSITIVE MOUNTS
# ===========================================================================
UID_INFO="$(id 2>/dev/null)"
echo "$UID_INFO" | grep -q 'uid=0(' && add HIGH running_as_root "Process runs as uid=0"

# decode CapEff bitmask -> flag dangerous caps without needing capsh
CAPEFF="$(grep '^CapEff:' /proc/self/status 2>/dev/null | awk '{print $2}')"
declare -A DANGER=( [21]=CAP_SYS_ADMIN [19]=CAP_SYS_PTRACE [16]=CAP_SYS_MODULE \
                    [2]=CAP_DAC_READ_SEARCH [13]=CAP_NET_RAW [1]=CAP_DAC_OVERRIDE \
                    [7]=CAP_SETUID [8]=CAP_SETGID )
CAPS_HELD=""
if [ -n "${CAPEFF:-}" ]; then
  mask=$((16#$CAPEFF))
  for bit in "${!DANGER[@]}"; do
    if (( (mask >> bit) & 1 )); then
      CAPS_HELD="$CAPS_HELD ${DANGER[$bit]}"
      case "${DANGER[$bit]}" in
        CAP_SYS_ADMIN|CAP_SYS_MODULE|CAP_SYS_PTRACE|CAP_DAC_READ_SEARCH)
          add HIGH "cap_${DANGER[$bit]}" "Process holds ${DANGER[$bit]} (container-escape primitive)";;
      esac
    fi
  done
fi

# sensitive mounts & exposed sockets
MOUNTS="$(cat /proc/mounts 2>/dev/null)"
for sock in /var/run/docker.sock /run/docker.sock /run/containerd/containerd.sock \
            /run/crio/crio.sock /var/run/libvirt/libvirt-sock; do
  [ -S "$sock" ] && add CRITICAL exposed_runtime_socket "$sock mounted in sandbox (host takeover surface)"
done
[ -d /var/run/secrets/kubernetes.io ] && add HIGH k8s_sa_token_mounted "Kubernetes service-account token present at /var/run/secrets/kubernetes.io"
echo "$MOUNTS" | grep -qE ' /host(/| )' && add HIGH host_fs_mounted "Host filesystem path mounted inside sandbox"
echo "$MOUNTS" | grep -qE 'proc .*/proc .*rw' && add MEDIUM proc_writable "/proc mounted rw"
echo "$MOUNTS" | grep -qE 'sysfs .*rw' && add LOW sys_writable "/sys mounted rw"

# environment: report NAMES of sensitive-looking vars, REDACT values
ENV_FLAGS=""
while IFS='=' read -r k _; do
  case "$k" in
    *SECRET*|*TOKEN*|*PASSWORD*|*PASSWD*|*APIKEY*|*API_KEY*|*ACCESS_KEY*|*PRIVATE_KEY*|*_KEY|*CREDENTIAL*|*AUTH*|*COOKIE*|*SESSION*|XAUTHORITY|KUBECONFIG|AWS_*|GCP_*|AZURE_*)
      ENV_FLAGS="$ENV_FLAGS $k=<redacted>";;
  esac
done < <(env 2>/dev/null)
[ -n "$ENV_FLAGS" ] && add MEDIUM env_secret_names "Sensitive-looking env vars present (values redacted):$ENV_FLAGS"

# ---- interception surface: proxy / injected-CA configuration ---------------
# A proxy or custom CA hidden in app/tool config can silently reroute or MITM
# egress. NOTE: in an agent sandbox a managed egress proxy is often EXPECTED
# infrastructure — this reports what it finds for human review. It does not
# judge intent and never bypasses, disables, or extracts anything.
PROXY_HITS=""
redact_proxy(){ sed -E 's#(https?://)[^/[:space:]@]+@#\1<redacted>@#g'; }
add_proxy(){ [ -n "${2:-}" ] && PROXY_HITS="$PROXY_HITS $1=[$(printf '%s' "$2" | redact_proxy | tr '\n' ';' | cut -c1-120)]"; }
add_proxy env     "$(env 2>/dev/null | grep -iE '^(https?|all|no)_proxy=')"
for f in /etc/environment /etc/profile "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc" "$HOME/.curlrc" "$HOME/.wgetrc"; do
  [ -r "$f" ] && add_proxy "$f" "$(grep -iE 'proxy' "$f" 2>/dev/null)"
done
[ -r "$HOME/.npmrc" ]              && add_proxy npmrc  "$(grep -iE 'proxy' "$HOME/.npmrc" 2>/dev/null)"
[ -r "$HOME/.gitconfig" ]          && add_proxy git    "$(have git && git config --get-regexp '.*proxy' 2>/dev/null)"
[ -r "$HOME/.config/pip/pip.conf" ]&& add_proxy pip    "$(grep -iE 'proxy' "$HOME/.config/pip/pip.conf" 2>/dev/null)"
[ -r "$HOME/.docker/config.json" ] && add_proxy docker "$(grep -iE 'proxy' "$HOME/.docker/config.json" 2>/dev/null)"
[ -d /etc/apt/apt.conf.d ]         && add_proxy apt    "$(grep -rhiE 'Acquire::http[s]?::Proxy' /etc/apt/apt.conf.d 2>/dev/null)"
for f in "$HOME"/.config/Code*/User/settings.json ./.vscode/settings.json "$HOME/.config/JetBrains"/*/options/proxy.settings.xml; do
  [ -r "$f" ] && add_proxy "app:$(basename "$(dirname "$f")")" "$(grep -iE 'proxy' "$f" 2>/dev/null)"
done
# injected CA trust — the TLS-interception companion to a proxy
CA_HITS=""
for v in NODE_EXTRA_CA_CERTS REQUESTS_CA_BUNDLE SSL_CERT_FILE CURL_CA_BUNDLE GIT_SSL_CAINFO; do
  eval "val=\${$v:-}"; [ -n "$val" ] && CA_HITS="$CA_HITS $v=$val"
done
[ -n "$PROXY_HITS" ] && add LOW proxy_config_present "Proxy config present — confirm it is the sanctioned egress proxy, not injected:$PROXY_HITS"
[ -n "$CA_HITS" ]    && add LOW custom_ca_trust "Custom CA trust configured (enables TLS interception; confirm it is expected):$CA_HITS"

# ===========================================================================
# 2b. HOST POSTURE  (relevant when there is no container boundary: plain OS / VM / microVM)
#     Inventory only — listening services, firewall backend, SUID surface.
# ===========================================================================
LISTEN_PORTS="$( ( (have ss && ss -tlnH) || (have netstat && netstat -tln) ) 2>/dev/null \
                 | awk '{print $4}' | sed 's/.*://' | grep -E '^[0-9]+$' | sort -un | tr '\n' ' ')"
LISTEN_COUNT="$(printf '%s' "$LISTEN_PORTS" | wc -w | tr -d ' ')"

FW="none"
have nft       && run 2 nft list ruleset   2>/dev/null | grep -q . && FW="nftables"
have iptables  && run 2 iptables -S         2>/dev/null | grep -qvE '^-P (INPUT|FORWARD|OUTPUT) ACCEPT$' && FW="${FW}+iptables"
# A host-class environment with no firewall and open listeners is a real finding.
case "$ISO_CLASS" in
  bare-metal*|full-vm*|microvm|virtualized*)
    [ "$FW" = none ] && [ "${LISTEN_COUNT:-0}" -gt 0 ] && \
      add MEDIUM no_host_firewall "No nftables/iptables ruleset; $LISTEN_COUNT listening TCP port(s): $LISTEN_PORTS";;
esac

# SUID inventory (privilege surface) — bounded find, count + sample, no exploitation.
SUID_LIST="$(run 20 find / -xdev -perm -4000 -type f 2>/dev/null | head -40 | tr '\n' ' ')"
SUID_COUNT="$(printf '%s' "$SUID_LIST" | wc -w | tr -d ' ')"
# flag only clearly-unexpected SUID binaries (common shells/interpreters as SUID = escalation path)
echo "$SUID_LIST" | grep -qE '/(bash|sh|python[0-9.]*|perl|env|find|vim|nmap|tar)( |$)' && \
  add HIGH suid_interpreter "SUID set on an interpreter/shell/utility (direct root escalation): $SUID_LIST"

# ===========================================================================
# 3. NETWORK EGRESS & LATERAL-MOVEMENT PATHS  (reachability only)
# ===========================================================================
IPADDR="$( (have ip && ip -o addr show) || cat /proc/net/fib_trie 2>/dev/null | head -40)"
ROUTES="$( (have ip && ip route) || rd /proc/net/route)"
GW="$(have ip && ip route 2>/dev/null | awk '/default/{print $3; exit}')"
RESOLV="$(rd /etc/resolv.conf)"

http_code() { # reachability probe; captures status, not body
  if have curl; then run 3 curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$@"; else echo "no-curl"; fi
}

# IMDS — provider-aware REACHABILITY. We confirm the service answers and whether
# a credentials endpoint is exposed. We deliberately do NOT fetch key material.
IMDS_AWS="$(http_code http://169.254.169.254/latest/meta-data/)"
IMDS_AWS_V2TOKEN="$(have curl && run 3 curl -s -o /dev/null -w '%{http_code}' --max-time 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || echo n/a)"
IMDS_GCP="$(http_code -H 'Metadata-Flavor: Google' http://169.254.169.254/computeMetadata/v1/)"
IMDS_AZURE="$(http_code -H 'Metadata: true' 'http://169.254.169.254/metadata/instance?api-version=2021-02-01')"
if [ "$IMDS_AWS" = "200" ] && [ "$IMDS_AWS_V2TOKEN" != "200" ]; then
  add CRITICAL imds_v1_reachable "AWS IMDSv1 reachable from sandbox (role credentials exposed via SSRF); enforce IMDSv2 + hop-limit=1"
elif [ "$IMDS_AWS" = "200" ] || [ "$IMDS_GCP" = "200" ] || [ "$IMDS_AZURE" = "200" ]; then
  add HIGH imds_reachable "Cloud metadata service reachable from sandbox (AWS=$IMDS_AWS GCP=$IMDS_GCP AZURE=$IMDS_AZURE); block 169.254.169.254 egress unless required"
fi

# DNS + outbound egress (is the sandbox network-isolated at all?)
DNS_EGRESS="fail"; have dig && run 3 dig +short +time=2 @8.8.8.8 example.com >/dev/null 2>&1 && DNS_EGRESS="ok"
[ "$DNS_EGRESS" = fail ] && have nslookup && run 3 nslookup example.com 8.8.8.8 >/dev/null 2>&1 && DNS_EGRESS="ok"
OUT_HTTPS="$(http_code https://example.com)"
[ "$DNS_EGRESS" = ok ] && add MEDIUM dns_egress_open "Arbitrary external DNS (8.8.8.8) reachable — no DNS egress control"
[ "$OUT_HTTPS" = "200" ] && add MEDIUM internet_egress_open "Unrestricted outbound HTTPS to the internet"

# host gateway management-port REACHABILITY (TCP connect only, no protocol spoken)
GW_PORTS=""
if [ -n "${GW:-}" ]; then
  for p in 22 2375 2376 6443 10250 10255 2379; do
    if run 2 bash -c "exec 3<>/dev/tcp/$GW/$p" 2>/dev/null; then
      GW_PORTS="$GW_PORTS $p"; exec 3>&- 2>/dev/null
      case $p in
        2375) add CRITICAL docker_api_open  "Host gateway $GW:2375 open — unauthenticated Docker API = host root";;
        10250) add HIGH kubelet_open        "Host gateway $GW:10250 open — kubelet API reachable";;
        6443) add HIGH kube_apiserver_open  "Host gateway $GW:6443 open — Kubernetes API reachable";;
        2379) add HIGH etcd_open            "Host gateway $GW:2379 open — etcd reachable";;
      esac
    fi
  done
fi

# ===========================================================================
# 4. EMIT JSON
# ===========================================================================
findings_json=""
for f in "${FINDINGS[@]:-}"; do
  [ -z "$f" ] && continue
  IFS='|' read -r sev id ev <<<"$f"
  findings_json="$findings_json{\"severity\":\"$sev\",\"id\":\"$id\",\"evidence\":\"$ev\"},"
done
findings_json="[${findings_json%,}]"

# ---- per-class risk scoring -------------------------------------------------
# Severity has a base weight; the isolation_class adjusts findings by CONTEXT,
# because what counts as a failure differs by boundary type:
#   * microVM / gVisor are *supposed* to be the strong boundary, so an egress or
#     escape finding there defeats the entire design -> weighted up (x1.5).
#   * On bare metal / plain OS there is no container to escape, so "root" or a
#     held capability isn't an escape finding -> weighted down (x0.5); host-net
#     issues (firewall, SUID, IMDS) still count fully.
#   * Containers are scored at the standard baseline.
score10=0
for f in "${FINDINGS[@]:-}"; do
  [ -z "$f" ] && continue
  IFS='|' read -r sev id _ <<<"$f"
  case "$sev" in CRITICAL) base=40;; HIGH) base=25;; MEDIUM) base=10;; LOW) base=4;; *) base=0;; esac
  case "$id" in
    imds*|docker_api_open|exposed_runtime_socket|kubelet_open|kube_apiserver_open|etcd_open|host_fs_mounted|k8s_sa_token_mounted) cat=escape_egress;;
    running_as_root|cap_*|proc_writable|sys_writable|unprivileged_*|seccomp_disabled) cat=containment;;
    *) cat=host_net;;
  esac
  mult=10
  case "$ISO_CLASS" in
    microvm|gvisor-sandbox)  [ "$cat" = escape_egress ] && mult=15;;
    bare-metal-or-unknown)   [ "$cat" = containment ]   && mult=5;;
  esac
  score10=$(( score10 + base*mult ))
done
SCORE=$(( score10 / 10 )); [ "$SCORE" -gt 100 ] && SCORE=100
if   [ "$SCORE" -eq 0 ];  then GRADE="A (hardened)"
elif [ "$SCORE" -lt 15 ]; then GRADE="B"
elif [ "$SCORE" -lt 35 ]; then GRADE="C"
elif [ "$SCORE" -lt 60 ]; then GRADE="D"
else GRADE="F (critical exposure)"; fi

print_table() {
  printf '\n  SANDBOX POSTURE  —  isolation class: %s\n' "$ISO_CLASS"
  printf '  risk score: %s/100    grade: %s\n\n' "$SCORE" "$GRADE"
  printf '  %-9s %-26s %s\n' SEVERITY FINDING EVIDENCE
  printf '  %-9s %-26s %s\n' --------- -------------------------- ----------------------------------
  local n=0
  for want in CRITICAL HIGH MEDIUM LOW; do
    for f in "${FINDINGS[@]:-}"; do
      [ -z "$f" ] && continue
      IFS='|' read -r s id ev <<<"$f"
      [ "$s" = "$want" ] && { printf '  %-9s %-26s %s\n' "$s" "$id" "$ev"; n=$((n+1)); }
    done
  done
  [ "$n" -eq 0 ] && printf '  (no findings — posture clean for this isolation class)\n'
  printf '\n'
}

print_json() {
cat <<JSON
{
  "schema_version": "1.2",
  "risk_score": $SCORE,
  "grade": "$(j "$GRADE")",
  "runtime": {
    "kernel": "$(j "$KERNEL")",
    "isolation_class": "$(j "$ISO_CLASS")",
    "virtualization": "$(j "$VIRT")",
    "hypervisor_flag": "$(j "$HYPERVISOR_FLAG")",
    "dmi_vendor_product": "$(j "$SYS_VENDOR / $PRODUCT")",
    "pci_bus_present": "$(j "$PCI_PRESENT")",
    "container": "$(j "$CONTAINER")",
    "cgroup_pid1": "$(j "$CGROUP1")",
    "seccomp": "$(j "$SECCOMP")",
    "apparmor": "$(j "$APPARMOR")",
    "selinux": "$(j "$SELINUX")",
    "sysctl": {"unprivileged_bpf_disabled":"$BPF_DIS","unprivileged_userns_clone":"$USERNS","ptrace_scope":"$PTRACE"}
  },
  "privileges": {
    "id": "$(j "$UID_INFO")",
    "cap_eff_hex": "$(j "${CAPEFF:-}")",
    "dangerous_caps_held": "$(j "${CAPS_HELD# }")",
    "sensitive_env_names": "$(j "${ENV_FLAGS# }")"
  },
  "host_posture": {
    "firewall": "$(j "$FW")",
    "listening_tcp_ports": "$(j "${LISTEN_PORTS% }")",
    "listening_count": "${LISTEN_COUNT:-0}",
    "suid_count": "${SUID_COUNT:-0}",
    "suid_sample": "$(j "${SUID_LIST% }")"
  },
  "interception": {
    "proxy_config": "$(j "${PROXY_HITS# }")",
    "custom_ca_trust": "$(j "${CA_HITS# }")"
  },
  "network": {
    "default_gateway": "$(j "${GW:-none}")",
    "imds": {"aws_v1":"$IMDS_AWS","aws_v2_token":"$IMDS_AWS_V2TOKEN","gcp":"$IMDS_GCP","azure":"$IMDS_AZURE"},
    "dns_egress": "$DNS_EGRESS",
    "outbound_https": "$OUT_HTTPS",
    "gateway_open_ports": "$(j "${GW_PORTS# }")"
  },
  "findings": $findings_json
}
JSON
}

case "$FORMAT" in
  json)  print_json;;
  table) print_table;;
  both)  print_table >&2; print_json;;   # table to stderr so stdout stays parseable JSON
esac

# ---------------------------------------------------------------------------
# JSON SCHEMA (the piece missing from the original prompt)
#   schema_version : string
#   runtime        : {kernel, isolation_class, virtualization, hypervisor_flag,
#                     dmi_vendor_product, pci_bus_present, container, cgroup_pid1,
#                     seccomp, apparmor, selinux, sysctl{...}}
#                     isolation_class ∈ {bare-metal-or-unknown, virtualized:*,
#                       full-vm:*, microvm, gvisor-sandbox, container:*}
#   privileges     : {id, cap_eff_hex, dangerous_caps_held, sensitive_env_names}
#   host_posture   : {firewall, listening_tcp_ports, listening_count,
#                     suid_count, suid_sample}
#   interception   : {proxy_config, custom_ca_trust}  (proxy/CA found in app
#                     and tool config; reported for review, not judged)
#   network        : {default_gateway, imds{aws_v1,aws_v2_token,gcp,azure},
#                     dns_egress, outbound_https, gateway_open_ports}
#   findings[]     : {severity: CRITICAL|HIGH|MEDIUM|LOW, id, evidence}
# ---------------------------------------------------------------------------
