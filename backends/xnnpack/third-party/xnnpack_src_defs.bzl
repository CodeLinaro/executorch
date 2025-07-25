load(
    "//backends/xnnpack/third-party/XNNPACK:build_srcs.bzl",
    _LOGGING_SRCS = "LOGGING_SRCS",
    _OPERATOR_SRCS = "OPERATOR_SRCS",
    _SUBGRAPH_SRCS = "SUBGRAPH_SRCS",
    _TABLE_SRCS = "TABLE_SRCS",
)
load("//backends/xnnpack/third-party/XNNPACK/gen:microkernels.bzl", "prod_srcs_for_arch")
load("@fbsource//xplat/executorch/third-party:glob_defs.bzl", "subdir_glob")

# To get from XNNPACK:build_srcs.bzl in the future
def define_xnnpack_build_src(xnnpack_build_src):
    return ["XNNPACK/{}".format(src) for src in xnnpack_build_src]

def prod_srcs_for_arch_wrapper(arch):
    prod_srcs = prod_srcs_for_arch(arch)
    return define_xnnpack_build_src(prod_srcs)

def get_xnnpack_headers():
    src_headers = subdir_glob([
        ("XNNPACK", "src/xnnpack/*.h"),
        ("XNNPACK", "src/xnnpack/**/*.h"),
        ("XNNPACK", "src/**/*.h"),
    ])
    include_headers = subdir_glob([
        ("XNNPACK", "include/*.h"),
    ])
    ukernel_headers = subdir_glob([
        ("XNNPACK", "src/**/*.inc"),
    ])
    return src_headers | include_headers | ukernel_headers

def get_ukernel_config_srcs():
    return subdir_glob([
        ("XNNPACK/src/configs", "*.c"),
    ]).values()

OPERATOR_SRCS = define_xnnpack_build_src(_OPERATOR_SRCS)
SUBGRAPH_SRCS = define_xnnpack_build_src(_SUBGRAPH_SRCS)
TABLE_SRCS = define_xnnpack_build_src(_TABLE_SRCS)
LOGGING_SRCS = define_xnnpack_build_src(_LOGGING_SRCS)
