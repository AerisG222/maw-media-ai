#!/usr/bin/env bash
#
# Export the Places365 scene classifier to ONNX, once, on any machine.
#
# WHY THIS EXISTS
#
# Scene scoring runs on onnxruntime, which this project already depends on for
# face detection.  The published Places365 weights are PyTorch, so converting
# them needs torch -- a ~200MB dependency used for about ten seconds and never
# again at runtime.  Rather than add it to requirements.txt and carry it
# forever, this script builds a throwaway venv, exports the model, verifies the
# export, and deletes the venv.  Nothing torch-shaped survives.
#
# Re-run it on a new machine (or after clearing the cache) to get a
# byte-comparable model: the SHA256 of every download is printed, so two
# machines can be checked against each other.
#
#   ./export-scene-model.sh            # export if missing
#   ./export-scene-model.sh --force    # re-export over an existing model
#
# Outputs (override the directory with SCENE_MODEL_DIR):
#
#   ~/.cache/maw-media-ai/models/resnet18_places365.onnx
#   ~/.cache/maw-media-ai/models/categories_places365.txt
#   ~/.cache/maw-media-ai/models/IO_places365.txt
#
# The two text files are as important as the weights: categories names the 365
# scene classes, and IO marks each one indoor(1) or outdoor(2).  The outdoor
# score is the softmax mass over the outdoor classes, so it is derived from IO
# rather than from a threshold someone picked.
#
# SUPPLY CHAIN NOTE.  The weights are served over plain HTTP by MIT CSAIL (that
# is the only URL upstream publishes) and are a pickle, which torch.load
# executes.  Two mitigations: the load uses weights_only=True so only tensors
# and primitives are unpickled, and every SHA256 is printed so a second machine
# can be compared against the first.  If you have a known-good hash from a
# previous run, set EXPECTED_WEIGHTS_SHA256 and this script will refuse to
# proceed on a mismatch.

set -euo pipefail

SCENE_MODEL_DIR="${SCENE_MODEL_DIR:-$HOME/.cache/maw-media-ai/models}"
# Unpinned by default, deliberately.  torch publishes wheels per python minor
# version and lags new releases -- 2.9.1 has no cp314 wheel, so a hard default
# pin fails on exactly the machine you are trying to reproduce on.  pip picks
# something compatible with the local interpreter and the resolved version is
# printed at the end, so an exact repeat is TORCH_VERSION=<that> away.
TORCH_VERSION="${TORCH_VERSION:-}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
EXPECTED_WEIGHTS_SHA256="${EXPECTED_WEIGHTS_SHA256:-}"

WEIGHTS_URL="http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
CATEGORIES_URL="https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
IO_URL="https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt"

ONNX_OUT="$SCENE_MODEL_DIR/resnet18_places365.onnx"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ -f "$ONNX_OUT" ] && [ "$FORCE" -eq 0 ]; then
    echo "Model already present: $ONNX_OUT"
    echo "  sha256: $(sha256sum "$ONNX_OUT" | cut -d' ' -f1)"
    echo "Pass --force to re-export."
    exit 0
fi

# Overridable because torch may not have a wheel for the newest interpreter.
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || { echo "$PYTHON not found" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl not found" >&2; exit 1; }

WORK="$(mktemp -d)"
# The venv is the whole point of the script; never leave it behind, including on
# failure or ^C.
trap 'rm -rf "$WORK"' EXIT INT TERM

echo "==> Work dir: $WORK"
mkdir -p "$SCENE_MODEL_DIR"

echo "==> Downloading Places365 artifacts"
curl -fsSL --retry 3 -o "$WORK/weights.pth.tar" "$WEIGHTS_URL"
curl -fsSL --retry 3 -o "$WORK/categories_places365.txt" "$CATEGORIES_URL"
curl -fsSL --retry 3 -o "$WORK/IO_places365.txt" "$IO_URL"

WEIGHTS_SHA="$(sha256sum "$WORK/weights.pth.tar" | cut -d' ' -f1)"
echo "    weights    sha256 $WEIGHTS_SHA"
echo "    categories sha256 $(sha256sum "$WORK/categories_places365.txt" | cut -d' ' -f1)"
echo "    io         sha256 $(sha256sum "$WORK/IO_places365.txt" | cut -d' ' -f1)"

if [ -n "$EXPECTED_WEIGHTS_SHA256" ] && [ "$WEIGHTS_SHA" != "$EXPECTED_WEIGHTS_SHA256" ]; then
    echo "SHA256 mismatch for the weights!" >&2
    echo "  expected $EXPECTED_WEIGHTS_SHA256" >&2
    echo "  got      $WEIGHTS_SHA" >&2
    exit 1
fi

echo "==> Building throwaway venv (cpu-only torch${TORCH_VERSION:+ $TORCH_VERSION})"
"$PYTHON" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet --upgrade pip
# The cpu index keeps this to ~200MB instead of pulling the whole CUDA stack,
# which the export does not need -- it runs the model once on random input.
TORCH_SPEC="torch${TORCH_VERSION:+==$TORCH_VERSION}"
TV_SPEC="torchvision${TORCHVISION_VERSION:+==$TORCHVISION_VERSION}"
if ! "$WORK/venv/bin/pip" install --quiet \
        --index-url https://download.pytorch.org/whl/cpu \
        "$TORCH_SPEC" "$TV_SPEC"; then
    echo >&2
    echo "torch install failed.  The usual cause is that torch has no wheel for" >&2
    echo "python $("$WORK/venv/bin/python" -c 'import sys;print(\"%d.%d\"%sys.version_info[:2])') yet." >&2
    echo "Re-run against an older interpreter, e.g.:" >&2
    echo "  PYTHON=python3.11 ./export-scene-model.sh" >&2
    exit 1
fi
# onnxscript: torch's current exporter is dynamo-based and imports it lazily, so
# without it torch.onnx.export dies with ModuleNotFoundError rather than
# anything that hints at the cause.
# onnxruntime: so the export is verified inside the same venv that produced it,
# rather than trusting it and finding out at scan time.
"$WORK/venv/bin/pip" install --quiet onnxscript onnxruntime numpy

echo "==> Exporting to ONNX"
"$WORK/venv/bin/python" - "$WORK" <<'PYEOF'
import sys
import numpy as np
import torch
from torchvision.models import resnet18

work = sys.argv[1]

# weights_only=True: the checkpoint holds tensors plus a few ints and strings,
# so nothing needs arbitrary unpickling.  See the supply chain note above.
ckpt = torch.load(f"{work}/weights.pth.tar", map_location="cpu", weights_only=True)

# Trained with DataParallel, so every key carries a "module." prefix.
state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}

model = resnet18(num_classes=365)
model.load_state_dict(state)
model.eval()

# Batch is dynamic so the scanner can size batches to the GPU; 224x224 is fixed
# because that is what the network was trained on.
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy,
    f"{work}/resnet18_places365.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    # 18, not a lower number: the exporter emits opset 18 natively and asking it
    # to down-convert fails noisily (version_converter cannot adapt the axes
    # inputs) while still producing a working model.  onnxruntime 1.25 is happy
    # with 18.
    opset_version=18,
    # Without this the dynamo exporter writes the weights to a sidecar
    # "<name>.onnx.data" and leaves a 92KB graph behind that is useless on its
    # own.  One self-contained file is worth the 45MB.
    external_data=False,
)

# Record what the real model says, so the INSTALLED file can be checked against
# it after the move.  Verifying here would only prove the work dir was fine.
rng = np.random.default_rng(0)
probe = rng.standard_normal((4, 3, 224, 224), dtype=np.float32)
with torch.no_grad():
    torch_out = model(torch.from_numpy(probe)).numpy()

np.save(f"{work}/probe.npy", probe)
np.save(f"{work}/torch_out.npy", torch_out)
print("    exported; reference output recorded for post-install verification")
PYEOF

echo "==> Installing to $SCENE_MODEL_DIR"
mv "$WORK/resnet18_places365.onnx" "$ONNX_OUT"
mv "$WORK/categories_places365.txt" "$SCENE_MODEL_DIR/"
mv "$WORK/IO_places365.txt" "$SCENE_MODEL_DIR/"

# Verify the file that actually shipped, from its final location.  An earlier
# version of this script checked the export inside the work dir and then moved
# only the .onnx, leaving the weights behind in a sidecar -- the check passed
# and the installed model was unloadable.
echo "==> Verifying the installed model"
"$WORK/venv/bin/python" - "$WORK" "$ONNX_OUT" <<'PYEOF'
import sys
import numpy as np
import onnxruntime as ort

work, installed = sys.argv[1], sys.argv[2]
probe = np.load(f"{work}/probe.npy")
reference = np.load(f"{work}/torch_out.npy")

sess = ort.InferenceSession(installed, providers=["CPUExecutionProvider"])
out = sess.run(["logits"], {"input": probe})[0]

diff = float(np.abs(reference - out).max())
agree = int((reference.argmax(1) == out.argmax(1)).sum())
print(f"    max |torch - onnx| = {diff:.2e}")
print(f"    argmax agreement   = {agree}/{len(probe)}")
print(f"    output shape       = {out.shape} (365 scene classes)")

if diff > 1e-4 or agree != len(probe):
    raise SystemExit("installed model does not match the torch model - refusing to ship it")
PYEOF

echo
echo "Done."
echo "  $ONNX_OUT"
echo "    sha256 $(sha256sum "$ONNX_OUT" | cut -d' ' -f1)"
echo "    $(du -h "$ONNX_OUT" | cut -f1)"
echo "  $SCENE_MODEL_DIR/categories_places365.txt"
echo "  $SCENE_MODEL_DIR/IO_places365.txt"
echo
RESOLVED_TORCH="$("$WORK/venv/bin/pip" show torch 2>/dev/null | awk '/^Version:/{print $2}')"
RESOLVED_TV="$("$WORK/venv/bin/pip" show torchvision 2>/dev/null | awk '/^Version:/{print $2}')"
echo "Built with torch $RESOLVED_TORCH / torchvision $RESOLVED_TV on python \
$("$WORK/venv/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo
echo "To reproduce this exact model elsewhere:"
echo "  EXPECTED_WEIGHTS_SHA256=$WEIGHTS_SHA \\"
echo "  TORCH_VERSION=$RESOLVED_TORCH TORCHVISION_VERSION=$RESOLVED_TV \\"
echo "  ./export-scene-model.sh"
