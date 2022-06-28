# #!/usr/bin/env python3
# *******************************************************************************
# Copyright 2022 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
# *******************************************************************************

import json
import logging
import subprocess
import argparse
import os
import re
import statistics

logging.basicConfig()
logger = logging.getLogger("bm-runner")
logger.setLevel(logging.INFO)

def print_command_output(stdout, stderr):
    if stdout:
        logger.info("### STDOUT ####################################")
        print(stdout, end="")
        logger.info("### END STDOUT ####################################")
    if stderr:
        logger.info("### STDERR ####################################")
        print(stderr, end="")
        logger.info("### END STDERR ####################################")


def run_command(cmd, env = None, verbose=False):
    """Invoke a command with optional environment."""

    save_env = {}
    if env is not None:
        for envvar in env.keys():
            if envvar in os.environ:
                save_env[envvar] = os.environ[envvar]
            else:
                save_env[envvar] = None
            os.environ[envvar] = env[envvar]
            logger.debug(f"Set {envvar} to {env[envvar]}")

    logger.debug(cmd)

    rc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout = rc.stdout.decode("utf-8")
    stderr = rc.stderr.decode("utf-8")
    if verbose:
        print_command_output(stdout, stderr)

    if env is not None:
        for envvar in env.keys():
            if save_env[envvar] is not None:
                os.environ[envvar] = save_env[envvar]
                logger.debug(f"Reset {envvar} to {save_env[envvar]}")
            else:
                os.environ.pop(envvar)
                logger.debug(f"Unset {envvar}")

    return stdout, stderr

def convert_to_dollars(results, machine):
    result = statistics.median(results)
    # Seconds in an hour divided by runs then multiply by 1000 for ms
    iterations_per_hour = (3600 / result) * 1000
    dollars_per_iteration = MACHINE_COSTS[machine] / iterations_per_hour

    return dollars_per_iteration

def parse_args(all_benchmarks_list, frameworks_list, machines_list):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bm",
        nargs='+',
        help=f"Benchmarks to run: all or {','.join(all_benchmarks_list)}",
        choices=["all"] + all_benchmarks_list,
        required=True,
    )
    parser.add_argument(
        "--fw",
        help=f"Framework to run for: {','.join(frameworks_list)}",
        choices=frameworks_list,
        required=True,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of inference runs",
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="Verbosity (can be used multiple times for more details)",
    )
    parser.add_argument(
        "--mach",
        choices=machines_list,
        help=f"Convert times to machine cost for: {','.join(machines_list)}",
    )
    return parser.parse_args()


MACHINE_COSTS = {
    'aarch64': 2.176, # c6g = $2.176 per hour
    'x86_64': 2.72, # c6i = $2.72 per hour
}

TF_PY_EXAMPLES = "docker/tensorflow-aarch64/examples/py-api"
TF_JIT_XLA_ENV = { "TF_XLA_FLAGS": "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" }

PYTORCH_EXAMPLES = "docker/pytorch-aarch64/examples"

PYTEST_RESNET = "python classify_image.py -m ./resnet_v1-50.yml -i https://upload.wikimedia.org/wikipedia/commons/3/32/Weimaraner_wb.jpg"
PYTEST_SSD_RESNET = "python detect_objects.py -m ./ssd_resnet34.yml -i https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/street.jpg"
PYTEST_BERT = "python answer_questions.py -id 56de16ca4396321400ee25c7"

RESULT_REGEXP =  r"Inference time: ([0-9]+) ms"

FRAMEWORKS = ["TF-PY", "PYTORCH"]

BENCHMARKS = {}
BENCHMARKS["resnet"] = {
        "name": "resnet_v1-50",
        "framework": {
            "TF-PY": {
                "dir": TF_PY_EXAMPLES,
                "cmd": PYTEST_RESNET,
            },
            "PYTORCH": {
                "dir": PYTORCH_EXAMPLES,
                "cmd": PYTEST_RESNET,
            }
        },
        "result": RESULT_REGEXP,
    }
BENCHMARKS["resnet-xla"] = {
        "name": "resnet_v1-50 with XLA",
        "framework": {
            "TF-PY": {
                "dir": TF_PY_EXAMPLES,
                "cmd": PYTEST_RESNET,
                "env": TF_JIT_XLA_ENV,
            },
            "PYTORCH": {
                "dir": PYTORCH_EXAMPLES,
                "cmd": PYTEST_RESNET + " --xla",
            }
        },
        "result": RESULT_REGEXP,
    }
BENCHMARKS["ssd_resnet"] = {
        "name": "resnet34-ssd1200",
        "framework": {
            "TF-PY": {
                "dir": TF_PY_EXAMPLES,
                "cmd": PYTEST_SSD_RESNET,
            },
            "PYTORCH": {
                "dir": PYTORCH_EXAMPLES,
                "cmd": PYTEST_SSD_RESNET,
            }
        },
        "result": RESULT_REGEXP,
    }
BENCHMARKS["ssd_resnet-xla"] = {
        "name": "resnet34-ssd1200 with XLA",
        "framework": {
            "TF-PY": {
                "dir": TF_PY_EXAMPLES,
                "cmd": PYTEST_SSD_RESNET,
                "env": TF_JIT_XLA_ENV,
            },
            "PYTORCH": {
                "dir": PYTORCH_EXAMPLES,
                "cmd": PYTEST_SSD_RESNET + " --xla",
            }
        },
        "result": RESULT_REGEXP,
    }
BENCHMARKS["bert"] = {
        "name": "DistilBERT",
        "framework": {
            "TF-PY": {
                "dir": TF_PY_EXAMPLES,
                "cmd": PYTEST_BERT,
            },
            "PYTORCH": {
                "dir": PYTORCH_EXAMPLES,
                "cmd": PYTEST_BERT,
            }
        },
        "result": RESULT_REGEXP,
    }
BENCHMARKS["bert-xla"] = {
        "name": "DistilBERT with XLA",
        "framework": {
            "TF-PY": {
                "dir": TF_PY_EXAMPLES,
                "cmd": PYTEST_BERT,
                "env": TF_JIT_XLA_ENV,
            },
            "PYTORCH": {
                "dir": PYTORCH_EXAMPLES,
                "cmd": PYTEST_BERT + " --xla",
            }
        },
        "result": RESULT_REGEXP,
    }


def main():
    all_benchmarks = list(BENCHMARKS.keys())
    logger.debug(all_benchmarks)

    args = parse_args(all_benchmarks, FRAMEWORKS, list(MACHINE_COSTS.keys()))

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if "all" in args.bm:
        bms_to_run = all_benchmarks
    else:
        bms_to_run = args.bm

    fw_to_run = args.fw

    for bm_name in bms_to_run:
        if bm_name in BENCHMARKS:
            bm = BENCHMARKS[bm_name]
            if fw_to_run in bm["framework"]:
                logger.info(f"Running {bm['name']} {args.runs} times...")
                fw = bm["framework"][fw_to_run]

                cwd = os.getcwd()
                os.chdir(fw["dir"])

                results = []
                cmd = f"{fw['cmd']} --runs {args.runs}"
                env = fw["env"] if "env" in fw else None
                out, err = run_command(cmd, env, args.verbose)
                matches = re.findall(bm["result"], out)
                if matches:
                    results = [int(r) for r in matches]
                    logger.info(f"Results: {','.join(matches)} ms")
                else:
                    logger.error(f"No result from {bm_name}")
                    if not args.verbose:
                        print_command_output(out, err)

                logger.debug(results)
                if args.mach:
                    dollars_per_iteration = convert_to_dollars(results, args.mach)
                    dollars_per_million = dollars_per_iteration * 1000000
                    print(f"$$$$$ Dollars per a million iterations: {dollars_per_million:0.2f}")

                os.chdir(cwd)
            else:
                logger.warning(f"Framework {fw_to_run} not supported on {bm_name}")
        else:
            logger.warning(f"Unknown benchmark {bm_name}")

    return 0


if __name__ == "__main__":
    exit(main())
