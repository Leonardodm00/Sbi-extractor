#!/bin/bash
##########################################################################
# discover_campaigns.sh -- read-only inventory of a campaign tree.
#
# Run this on the LOGIN NODE before submitting anything. It writes nothing,
# submits nothing, and takes seconds. Its job is to answer the three
# questions that decide whether an export can run at all:
#
#   1. Where are the mea_iter_*.npz (DETECTED spikes -> the observable)?
#   2. Where are the matching iter_*.npz (parameter vectors AND the
#      topology block p0_conn / d0_conn / beta_conn)?
#   3. Do manifest.json and job_args.json exist, and what simtime do they
#      declare?
#
# Question 2 is the one that bites. process_campaign.py writes conn_prob
# into mea_iter_*.npz but NOT p0_conn, d0_conn or beta_conn: those exist
# only in the original sweep output. If the MEA outputs live in one tree and
# the sweep outputs in another, the export needs BOTH paths, and this script
# tells you whether they are co-located or split.
#
# USAGE:
#     ./discover_campaigns.sh <ROOT> [GLOB]
#
# EXAMPLE:
#     ./discover_campaigns.sh \
#         /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs 'campaign_cadex_rho1300v*'
##########################################################################

set -uo pipefail        # NOT -e: a probe that finds nothing must not abort

ROOT="${1:-}"
GLOB="${2:-campaign_*}"

if [ -z "${ROOT}" ]; then
    sed -n '2,28p' "$0"
    exit 2
fi
if [ ! -d "${ROOT}" ]; then
    echo "ERROR: not a directory: ${ROOT}" >&2
    exit 3
fi

echo "########################################################################"
echo "# root : ${ROOT}"
echo "# glob : ${GLOB}"
echo "# date : $(date -Is)"
echo "########################################################################"
echo ""

shopt -s nullglob
CAMPS=("${ROOT}"/${GLOB})
shopt -u nullglob

if [ "${#CAMPS[@]}" -eq 0 ]; then
    echo "No directories matched. Contents of ${ROOT}:"
    ls -1 "${ROOT}" | head -40
    exit 4
fi

echo "matched ${#CAMPS[@]} campaign director(ies)"
echo ""

for camp in "${CAMPS[@]}"; do
    [ -d "${camp}" ] || continue
    name="$(basename "${camp}")"
    echo "======================================================================"
    echo "CAMPAIGN: ${name}"
    echo "  path: ${camp}"

    # --- top-level shape -------------------------------------------------
    echo "  --- top level (first 12 entries) ---"
    ls -1 "${camp}" 2>/dev/null | head -12 | sed 's/^/      /'
    n_top=$(ls -1 "${camp}" 2>/dev/null | wc -l)
    echo "      (${n_top} entries total)"

    # --- where are the detections? ---------------------------------------
    mea_files=$(find "${camp}" -name 'mea_iter_*.npz' -type f 2>/dev/null | head -1)
    n_mea=$(find "${camp}" -name 'mea_iter_*.npz' -type f 2>/dev/null | wc -l)
    echo "  --- detections (mea_iter_*.npz) ---"
    if [ -n "${mea_files}" ]; then
        mea_dir=$(dirname "$(dirname "${mea_files}")")
        echo "      count      : ${n_mea}"
        echo "      example    : ${mea_files}"
        echo "      MEA_OUT -> : ${mea_dir}"
    else
        echo "      NONE FOUND. process_campaign.py has not been run over this"
        echo "      campaign, or its output is in a different tree."
        mea_dir=""
    fi

    # --- where are the sweep outputs (the topology join source)? ---------
    it_files=$(find "${camp}" -name 'iter_*.npz' -not -name 'mea_iter_*.npz' \
               -type f 2>/dev/null | head -1)
    n_it=$(find "${camp}" -name 'iter_*.npz' -not -name 'mea_iter_*.npz' \
           -type f 2>/dev/null | wc -l)
    echo "  --- sweep output (iter_*.npz -- holds p0/d0/beta) ---"
    if [ -n "${it_files}" ]; then
        it_dir=$(dirname "$(dirname "${it_files}")")
        echo "      count      : ${n_it}"
        echo "      example    : ${it_files}"
        echo "      CAMPAIGN ->: ${it_dir}"
    else
        echo "      NONE FOUND IN THIS TREE."
        echo "      The topology block (p0_conn/d0_conn/beta_conn) is NOT in"
        echo "      mea_iter_*.npz. You must point --campaign at the ORIGINAL"
        echo "      sweep output directory. Locate it with e.g.:"
        echo "        find \$HOME -name 'manifest.json' -path '*topo*' 2>/dev/null | head"
        it_dir=""
    fi

    # --- provenance files -------------------------------------------------
    echo "  --- provenance ---"
    for f in manifest.json job_args.json; do
        hit=$(find "${camp}" -maxdepth 3 -name "${f}" -type f 2>/dev/null | head -1)
        if [ -n "${hit}" ]; then
            echo "      ${f}: ${hit}"
        else
            echo "      ${f}: NOT FOUND"
        fi
    done

    # --- the two numbers that decide the export --------------------------
    ja=$(find "${camp}" -maxdepth 3 -name 'job_args.json' -type f 2>/dev/null | head -1)
    if [ -n "${ja}" ]; then
        python3 - "${ja}" <<'PYEOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print("      simtime      : %r   <- the IFR grid duration T" % d.get("simtime"))
print("      conn_prob    : [%r, %r]" % (d.get("conn_prob_lo"), d.get("conn_prob_hi")))
print("      conn_rule    : %r" % d.get("conn_rule"))
print("      sweep_group  : %r" % d.get("sweep_group"))
PYEOF
    fi
    mf=$(find "${camp}" -maxdepth 3 -name 'manifest.json' -type f 2>/dev/null | head -1)
    if [ -n "${mf}" ]; then
        python3 - "${mf}" <<'PYEOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
ai = d.get("active_indices")
print("      active_indices: %s axes" % (len(ai) if ai else "MISSING"))
print("      manifest_ver  : %r" % d.get("manifest_version"))
PYEOF
    fi

    # --- npz schema, from one real file ----------------------------------
    if [ -n "${mea_files}" ]; then
        echo "  --- keys in one mea_iter_*.npz ---"
        python3 - "${mea_files}" <<'PYEOF' 2>/dev/null
import numpy as np, sys
d = np.load(sys.argv[1], allow_pickle=False)
ks = sorted(d.files)
print("      %s" % ks)
need = ["det_t", "det_ch", "theta", "electrode_centers"]
miss = [k for k in need if k not in ks]
print("      required present: %s" % ("YES" if not miss else "NO -- missing %r" % miss))
if "electrode_centers" in ks:
    print("      n_electrodes    : %d" % np.asarray(d["electrode_centers"]).shape[0])
if "theta" in ks:
    print("      theta width     : %d" % np.asarray(d["theta"]).ravel().shape[0])
if "simtime" in ks:
    print("      simtime IN NPZ  : %r  <- INFERRED from last spike, do NOT use"
          % float(d["simtime"]))
PYEOF
    fi
    if [ -n "${it_files}" ]; then
        echo "  --- keys in one iter_*.npz ---"
        python3 - "${it_files}" <<'PYEOF' 2>/dev/null
import numpy as np, sys
d = np.load(sys.argv[1], allow_pickle=False)
ks = sorted(d.files)
print("      %s" % ks)
need = ["conn_prob", "p0_conn", "d0_conn", "beta_conn"]
miss = [k for k in need if k not in ks]
print("      topology block  : %s" % ("YES" if not miss else "NO -- missing %r" % miss))
for k in need:
    if k in ks:
        try:
            print("        %-10s = %r" % (k, float(d[k])))
        except Exception:
            pass
PYEOF
    fi

    # --- verdict ----------------------------------------------------------
    echo "  --- VERDICT ---"
    if [ -n "${mea_dir}" ] && [ -n "${it_dir}" ]; then
        echo "      READY. Submit with:"
        echo "        CAMPAIGN=${it_dir}"
        echo "        MEA_OUT=${mea_dir}"
    elif [ -n "${mea_dir}" ]; then
        echo "      BLOCKED: detections found, sweep output NOT in this tree."
        echo "      Find the sweep tree and pass it as CAMPAIGN."
    elif [ -n "${it_dir}" ]; then
        echo "      BLOCKED: sweep output found, no detections."
        echo "      Run process_campaign.py over this campaign first."
    else
        echo "      BLOCKED: neither detections nor sweep output found here."
    fi
    echo ""
done

echo "======================================================================"
echo "Paste this output back and the launcher can be pinned to your layout."
