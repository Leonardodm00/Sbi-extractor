"""
pipeline_protocol_fixture.py -- a fixture for the tests of the pipeline's job
protocol in this repository's job scripts (D-113). Not tool code.

    jobs_dir(hpc=None)      Simulation-Based-Inference's hpc/pipeline/jobs
                            (stage_job.sh, activate_env.sh, chain.sh), from
                            `hpc` (the suite's DSN tree's hpc/), SBI_HPC_DIR
                            or a few known places; None if absent
    make_env_folder(root)   an environment's own folder as a node without
                            conda finds it: conda-meta/ and bin/python,
                            bin/python3 that run this interpreter
    protocol_env(work_root, env_prefix=..., hpc=...)
                            the variables the pipeline gives the launchers,
                            the environment's prefix recorded under the work
                            root as the pipeline's submission records it
    no_conda_path(path)     PATH without any folder that holds a `conda`

Pure ASCII, LF only; stdlib.
"""

import os
import sys

_KNOWN_SBI = ("/home/claude/simulation-based-inference",
              os.path.expanduser("~/SBI"),
              os.path.expanduser("~/Simulation-Based-Inference"))


def jobs_dir(hpc=None):
    cands = []
    for h in (hpc, os.environ.get("SBI_HPC_DIR")):
        if h:
            cands.append(os.path.join(h, "pipeline", "jobs"))
    cands += [os.path.join(r, "hpc", "pipeline", "jobs") for r in _KNOWN_SBI]
    for c in cands:
        if all(os.path.isfile(os.path.join(c, f))
               for f in ("stage_job.sh", "activate_env.sh", "chain.sh")):
            return os.path.realpath(c)
    return None


def make_env_folder(root, name="sbi_export"):
    p = os.path.join(root, "env_folders", name)
    os.makedirs(os.path.join(p, "conda-meta"), exist_ok=True)
    os.makedirs(os.path.join(p, "bin"), exist_ok=True)
    for exe in ("python", "python3"):
        f = os.path.join(p, "bin", exe)
        with open(f, "w") as fh:
            fh.write('#!/bin/bash\nexec "%s" "$@"\n' % sys.executable)
        os.chmod(f, 0o755)
    return p


def protocol_env(work_root, env_prefix=None, env_name="sbi_export", chain="pl_smoke",
                 avoid_file=None, hold="0", hpc=None):
    jd = jobs_dir(hpc)
    if jd is None:
        raise RuntimeError("no Simulation-Based-Inference hpc/pipeline/jobs with stage_job.sh "
                           "(D-113): set SBI_HPC_DIR to a checkout at or after it")
    os.makedirs(work_root, exist_ok=True)
    if env_prefix:
        os.makedirs(os.path.join(work_root, "envs"), exist_ok=True)
        with open(os.path.join(work_root, "envs", env_name + ".prefix"), "w") as fh:
            fh.write(env_prefix + "\n")
    e = {"PIPELINE_JOBS_DIR": jd, "WORK_ROOT": work_root, "CHAIN_NAME": chain,
         "AVOID_HOLD_S": str(hold)}
    if avoid_file:
        e["AVOID_FILE"] = avoid_file
    return e


def no_conda_path(path):
    return os.pathsep.join(d for d in path.split(os.pathsep)
                           if d and not os.path.exists(os.path.join(d, "conda")))
