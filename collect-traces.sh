#!/bin/bash
# collect_traces.sh - Collect all traces for Perfetto demo

set -e  # Exit on error

FRACTAL_BIN="./target/release/fractal-renderer-vk"
OUTPUT_DIR="./traces"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Perfetto Demo Trace Collection${NC}"
echo "================================"

# Check if fractal renderer exists
if [ ! -f "$FRACTAL_BIN" ]; then
    echo -e "${RED}Error: Fractal renderer not found at $FRACTAL_BIN${NC}"
    echo "Please build it first with: cargo build --release"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Step 1: Starting system-wide scheduler trace...${NC}"
sudo trace-cmd record -e sched:sched_switch -e sched:sched_waking -C mono \
    -o "$OUTPUT_DIR/fractal.dat" > /dev/null 2>&1 &
TRACE_PID=$!
sleep 2  # Let trace-cmd start

echo -e "${YELLOW}Step 2: Starting perf CPU profiling...${NC}"

# Create temporary files for capturing output
FRACTAL_STDOUT=$(mktemp)
FRACTAL_STDERR=$(mktemp)
PERF_STDERR=$(mktemp)

# Run the fractal binary with error capture
# Capture perf's stderr separately to diagnose perf issues
# --call-graph=dwarf enables unwind tables for better stack traces
{ perf record -k mono -g --call-graph=dwarf -o "$OUTPUT_DIR/perf.data" -- \
    "$FRACTAL_BIN" --trace "$OUTPUT_DIR/fractal.pftrace" > "$FRACTAL_STDOUT" 2> "$FRACTAL_STDERR"; } 2> "$PERF_STDERR" &
PERF_PID=$!

echo -e "${YELLOW}Step 3: Recording... Press Ctrl+C to stop${NC}"

# Set up trap to handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Stopping recording...${NC}"; kill $PERF_PID 2>/dev/null || true' INT

# Wait for perf to finish and check exit status
wait $PERF_PID
PERF_EXIT_CODE=$?

# Clear the trap
trap - INT

# Debug: Show what happened
echo -e "${YELLOW}Debug: perf exit code = $PERF_EXIT_CODE${NC}"

# Check for perf-specific errors first
if [ -s "$PERF_STDERR" ]; then
    echo -e "${YELLOW}Perf stderr output:${NC}"
    cat "$PERF_STDERR"
fi

# Check if the command ran successfully
# Exit code 0 means it exited cleanly
# Exit code 130 means interrupted by Ctrl+C (expected)
# Exit code 143 means killed by SIGTERM (also OK)
# Exit code 141 means broken pipe (might be OK)
if [ $PERF_EXIT_CODE -ne 0 ] && [ $PERF_EXIT_CODE -ne 130 ] && [ $PERF_EXIT_CODE -ne 143 ] && [ $PERF_EXIT_CODE -ne 141 ]; then
    echo -e "${RED}Error: Command failed (exit code: $PERF_EXIT_CODE)${NC}"
    
    if [ -s "$FRACTAL_STDOUT" ]; then
        echo -e "${YELLOW}Fractal STDOUT:${NC}"
        cat "$FRACTAL_STDOUT"
    fi
    
    if [ -s "$FRACTAL_STDERR" ]; then
        echo -e "${YELLOW}Fractal STDERR:${NC}"
        cat "$FRACTAL_STDERR"
    fi
    
    # Check if perf.data was created at all
    if [ ! -f "$OUTPUT_DIR/perf.data" ]; then
        echo -e "${RED}Error: perf.data was not created. This might be a perf issue.${NC}"
        echo "Possible causes:"
        echo "  - Missing perf permissions (try: sudo sysctl kernel.perf_event_paranoid=-1)"
        echo "  - Perf not installed properly"
        echo "  - Invalid perf options"
    fi
    
    # Clean up temp files
    rm -f "$FRACTAL_STDOUT" "$FRACTAL_STDERR" "$PERF_STDERR"
    
    echo -e "${RED}Stopping trace collection due to failure...${NC}"
    sudo kill -INT $TRACE_PID 2>/dev/null || true
    wait $TRACE_PID 2>/dev/null || true
    exit 1
else
    echo -e "${GREEN}Recording completed successfully${NC}"
    # The trace file is written directly by the fractal binary
    if [ ! -f "$OUTPUT_DIR/fractal.pftrace" ] || [ ! -s "$OUTPUT_DIR/fractal.pftrace" ]; then
        echo -e "${YELLOW}Warning: No trace output from fractal binary${NC}"
    fi
fi

# Clean up temp files
rm -f "$FRACTAL_STDOUT" "$FRACTAL_STDERR" "$PERF_STDERR"

echo -e "${YELLOW}Step 4: Stopping scheduler trace...${NC}"
sudo kill -INT $TRACE_PID 2>/dev/null || true
wait $TRACE_PID 2>/dev/null || true

echo -e "${YELLOW}Step 5: Converting traces to text format that Perfetto understands...${NC}"

# Convert perf data
echo "  Converting perf data to text..."
if ! perf script -i "$OUTPUT_DIR/perf.data" > "$OUTPUT_DIR/fractal.perftext" 2>/dev/null; then
    echo -e "${RED}Warning: Failed to convert perf data${NC}"
    rm -f "$OUTPUT_DIR/fractal.perftext"
else
    # Compress the perf text output
    echo "  Compressing perf text output..."
    gzip -c "$OUTPUT_DIR/fractal.perftext" > "$OUTPUT_DIR/fractal.perftext.gz"
    
    # Generate flamegraph if FlameGraph tools are available
    echo "  Generating flamegraph..."
    FLAMEGRAPH_DIR="$HOME/FlameGraph"
    if [ -d "$FLAMEGRAPH_DIR" ] && [ -x "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" ] && [ -x "$FLAMEGRAPH_DIR/flamegraph.pl" ]; then
        if gunzip -c "$OUTPUT_DIR/fractal.perftext.gz" | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" --all | "$FLAMEGRAPH_DIR/flamegraph.pl" > "$OUTPUT_DIR/fractal.svg" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Flamegraph generated: $OUTPUT_DIR/fractal.svg"
        else
            echo -e "  ${YELLOW}Warning: Failed to generate flamegraph${NC}"
        fi
    else
        echo -e "  ${YELLOW}Warning: FlameGraph tools not found at $FLAMEGRAPH_DIR${NC}"
        echo "           Install with: git clone https://github.com/brendangregg/FlameGraph.git ~/FlameGraph"
    fi
fi

# Convert trace-cmd data
echo "  Converting scheduler trace to text..."
if ! trace-cmd report -i "$OUTPUT_DIR/fractal.dat" -N > "$OUTPUT_DIR/fractal.sched" 2>/dev/null; then
    echo -e "${RED}Warning: Failed to convert scheduler trace${NC}"
    rm -f "$OUTPUT_DIR/fractal.sched"
fi

# Clean up intermediate files
rm -f "$OUTPUT_DIR/fractal.dat"
rm -f "$OUTPUT_DIR/perf.data"

echo -e "${GREEN}✓ Trace collection complete!${NC}"
echo ""

# Check which files were actually created and show status
CREATED_FILES=0
echo "Files ready for Perfetto UI:"

if [ -f "$OUTPUT_DIR/fractal.perftext" ] && [ -s "$OUTPUT_DIR/fractal.perftext" ]; then
    echo -e "  ${GREEN}✓${NC} $OUTPUT_DIR/fractal.perftext   (CPU profiling)"
    CREATED_FILES=$((CREATED_FILES + 1))
else
    echo -e "  ${RED}✗${NC} $OUTPUT_DIR/fractal.perftext   (CPU profiling) - FAILED"
fi

if [ -f "$OUTPUT_DIR/fractal.sched" ] && [ -s "$OUTPUT_DIR/fractal.sched" ]; then
    echo -e "  ${GREEN}✓${NC} $OUTPUT_DIR/fractal.sched       (Scheduler events)"
    CREATED_FILES=$((CREATED_FILES + 1))
else
    echo -e "  ${RED}✗${NC} $OUTPUT_DIR/fractal.sched       (Scheduler events) - FAILED"
fi

if [ -f "$OUTPUT_DIR/fractal.pftrace" ] && [ -s "$OUTPUT_DIR/fractal.pftrace" ]; then
    echo -e "  ${GREEN}✓${NC} $OUTPUT_DIR/fractal.pftrace     (Application trace)"
    CREATED_FILES=$((CREATED_FILES + 1))
else
    echo -e "  ${RED}✗${NC} $OUTPUT_DIR/fractal.pftrace     (Application trace) - FAILED"
fi

if [ -f "$OUTPUT_DIR/fractal.svg" ] && [ -s "$OUTPUT_DIR/fractal.svg" ]; then
    echo -e "  ${GREEN}✓${NC} $OUTPUT_DIR/fractal.svg         (Flamegraph)"
    CREATED_FILES=$((CREATED_FILES + 1))
else
    echo -e "  ${RED}✗${NC} $OUTPUT_DIR/fractal.svg         (Flamegraph) - FAILED or skipped"
fi

if [ -f "$OUTPUT_DIR/fractal.perftext.gz" ] && [ -s "$OUTPUT_DIR/fractal.perftext.gz" ]; then
    echo -e "  ${GREEN}✓${NC} $OUTPUT_DIR/fractal.perftext.gz (Compressed perf text)"
fi

echo ""
if [ $CREATED_FILES -gt 0 ]; then
    echo "To analyze ($CREATED_FILES file(s) available):"
    echo "  • For interactive analysis: Open ui.perfetto.dev and drag trace files"
    echo "  • For CPU profiling: Open $OUTPUT_DIR/fractal.svg in a browser"
    echo "  • Watch the correlation magic happen!"
else
    echo -e "${RED}No trace files were created successfully. Check the errors above.${NC}"
fi
echo ""
