#!/bin/bash

# Set MoltenVK library paths
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export VK_ICD_FILENAMES="/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_LAYER_PATH="/opt/homebrew/share/vulkan/explicit_layer.d"

# Run the Rust application
cargo run "$@"