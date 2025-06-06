#!/bin/bash

set -euo pipefail

# Configuration
BINARY_NAME="wave2D"
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME=$(date -u '+%Y-%m-%d_%H:%M:%S')
GO_VERSION=$(go version | awk '{print $3}')
COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Advanced build flags for maximum performance
LDFLAGS="-s -w -X main.Version=${VERSION} -X main.BuildTime=${BUILD_TIME} -X main.GoVersion=${GO_VERSION} -X main.CommitHash=${COMMIT_HASH}"
GCFLAGS="-B -C -m=2" # Aggressive optimization
ASMFLAGS="-B"
TAGS="netgo,osusergo" # Static linking

# Performance optimization flags
CGO_ENABLED=0
GOPROXY="https://proxy.golang.org,direct"
GOSUMDB="sum.golang.org"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_build() {
    echo -e "${PURPLE}[BUILD]${NC} $1"
}

# Check Go version and environment
check_environment() {
    local required_version="1.19"
    local current_version=$(go version | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    if ! printf '%s\n%s\n' "$required_version" "$current_version" | sort -V -C; then
        print_error "Go version $required_version or higher required. Current: $current_version"
        exit 1
    fi
    
    # Check for required tools
    local tools=("git" "upx")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            print_warning "$tool not found. Some optimizations will be skipped."
        fi
    done
    
    print_success "Environment checks passed"
}

# Performance-optimized clean
clean() {
    print_status "Cleaning previous builds..."
    rm -f ${BINARY_NAME}
    rm -rf build/
    rm -rf dist/
    rm -f *.prof
    rm -f coverage.out
    
    # Clean Go caches for fresh build
    go clean -cache -modcache -testcache 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Enhanced testing with benchmarks
test() {
    print_status "Running comprehensive tests..."
    
    # Run tests with race detection
    go test -v -race -coverprofile=coverage.out ./...
    if [ $? -ne 0 ]; then
        print_error "Tests failed"
        exit 1
    fi
    
    # Generate coverage report
    if [ -f coverage.out ]; then
        go tool cover -html=coverage.out -o coverage.html
        coverage=$(go tool cover -func=coverage.out | grep total | awk '{print $3}')
        print_success "Test coverage: $coverage"
    fi
    
    print_success "All tests passed"
}

# Performance benchmarks
benchmark() {
    print_status "Running performance benchmarks..."
    
    # Run benchmarks with memory profiling
    go test -bench=. -benchmem -cpuprofile=cpu.prof -memprofile=mem.prof -count=3
    
    # Generate benchmark comparison if previous exists
    if [ -f bench_previous.txt ]; then
        print_status "Comparing with previous benchmarks..."
        go test -bench=. -count=5 > bench_current.txt
        if command -v benchcmp >/dev/null 2>&1; then
            benchcmp bench_previous.txt bench_current.txt
        fi
        mv bench_current.txt bench_previous.txt
    else
        go test -bench=. -count=5 > bench_previous.txt
    fi
    
    print_success "Benchmarks completed"
}

# Build for current platform with maximum optimization
build_native() {
    print_build "Building optimized native binary..."
    
    # Set performance environment variables
    export CGO_ENABLED=0
    export GOOS=$(go env GOOS)
    export GOARCH=$(go env GOARCH)
    
    # Advanced optimization build
    go build \
        -ldflags="${LDFLAGS}" \
        -gcflags="${GCFLAGS}" \
        -asmflags="${ASMFLAGS}" \
        -tags="${TAGS}" \
        -trimpath \
        -buildmode=exe \
        -o ${BINARY_NAME} \
        main.go
    
    print_success "Native build completed: ${BINARY_NAME}"
}

# Cross-compilation with advanced optimization
build_cross() {
    print_build "Cross-compiling optimized binaries for multiple platforms..."
    
    mkdir -p build dist
    
    # Extended platform list including modern architectures
    platforms=(
        "linux/amd64"
        "linux/arm64"
        "linux/386"
        "darwin/amd64"
        "darwin/arm64"
        "windows/amd64"
        "windows/386"
        "windows/arm64"
        "freebsd/amd64"
        "freebsd/arm64"
        "openbsd/amd64"
        "netbsd/amd64"
        "dragonfly/amd64"
    )
    
    for platform in "${platforms[@]}"; do
        IFS='/' read -r os arch <<< "$platform"
        output="build/${BINARY_NAME}-${os}-${arch}"
        
        if [ "$os" = "windows" ]; then
            output="${output}.exe"
        fi
        
        print_build "Building for ${os}/${arch}..."
        
        # Set build environment
        export CGO_ENABLED=0
        export GOOS=$os
        export GOARCH=$arch
        
        # Build with maximum optimization
        go build \
            -ldflags="${LDFLAGS}" \
            -gcflags="${GCFLAGS}" \
            -asmflags="${ASMFLAGS}" \
            -tags="${TAGS}" \
            -trimpath \
            -buildmode=exe \
            -o "$output" \
            main.go
        
        if [ $? -eq 0 ]; then
            # Get binary size
            size=$(ls -lh "$output" | awk '{print $5}')
            print_success "Built ${os}/${arch} (${size})"
            
            # Compress with UPX if available
            if command -v upx >/dev/null 2>&1 && [ "$os" != "darwin" ]; then
                print_status "Compressing ${os}/${arch} with UPX..."
                upx --best --lzma "$output" 2>/dev/null || print_warning "UPX compression failed for ${os}/${arch}"
            fi
            
            # Create compressed archives
            archive_name="dist/${BINARY_NAME}-${VERSION}-${os}-${arch}"
            if [ "$os" = "windows" ]; then
                zip "${archive_name}.zip" "$output" >/dev/null 2>&1
            else
                tar -czf "${archive_name}.tar.gz" -C "$(dirname "$output")" "$(basename "$output")" >/dev/null 2>&1
            fi
        else
            print_error "Failed to build for ${os}/${arch}"
        fi
    done
    
    print_success "Cross-compilation completed"
}

# Generate checksums and metadata
generate_metadata() {
    print_status "Generating metadata and checksums..."
    
    cd build
    
    # Generate checksums
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum * > checksums.sha256
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 * > checksums.sha256
    fi
    
    # Generate build info
    cat > build_info.json << EOF
{
    "version": "${VERSION}",
    "build_time": "${BUILD_TIME}",
    "go_version": "${GO_VERSION}",
    "commit_hash": "${COMMIT_HASH}",
    "ldflags": "${LDFLAGS}",
    "gcflags": "${GCFLAGS}",
    "platforms": [
EOF
    
    first=true
    for file in *; do
        if [ "$file" != "checksums.sha256" ] && [ "$file" != "build_info.json" ]; then
            if [ "$first" = true ]; then
                first=false
            else
                echo "," >> build_info.json
            fi
            echo -n "        \"$(basename "$file")\"" >> build_info.json
        fi
    done
    
    cat >> build_info.json << EOF

    ]
}
EOF
    
    cd ..
    print_success "Metadata generated"
}

# Show detailed build information
info() {
    print_status "Build Information:"
    echo "  Version: $VERSION"
    echo "  Build Time: $BUILD_TIME"
    echo "  Go Version: $GO_VERSION"
    echo "  Commit Hash: $COMMIT_HASH"
    echo "  Binary Name: $BINARY_NAME"
    echo "  LDFLAGS: $LDFLAGS"
    echo "  GCFLAGS: $GCFLAGS"
    echo "  Tags: $TAGS"
}

# Advanced binary optimization
optimize() {
    if [ -f "${BINARY_NAME}" ]; then
        print_status "Applying advanced optimizations..."
        
        original_size=$(ls -lh "${BINARY_NAME}" | awk '{print $5}')
        
        # Strip symbols if not already done
        if command -v strip >/dev/null 2>&1; then
            strip "${BINARY_NAME}" 2>/dev/null || true
        fi
        
        # Compress with UPX if available and not macOS
        if command -v upx >/dev/null 2>&1 && [[ "$(uname)" != "Darwin" ]]; then
            print_status "Compressing with UPX..."
            upx --best --lzma "${BINARY_NAME}" 2>/dev/null || print_warning "UPX compression failed"
        fi
        
        optimized_size=$(ls -lh "${BINARY_NAME}" | awk '{print $5}')
        print_success "Binary optimization completed: ${original_size} → ${optimized_size}"
    else
        print_error "Binary not found. Build first."
        exit 1
    fi
}

# Performance analysis build
performance() {
    print_build "Building with performance analysis enabled..."
    
    export CGO_ENABLED=0
    go build \
        -ldflags="${LDFLAGS}" \
        -gcflags="-m=2 -l=4" \
        -race \
        -o "${BINARY_NAME}-perf" \
        main.go
    
    print_success "Performance analysis build completed: ${BINARY_NAME}-perf"
}

# Complete release build pipeline
release() {
    print_status "Starting complete release build pipeline..."
    
    check_environment
    clean
    test
    benchmark
    build_cross
    generate_metadata
    info
    
    print_success "Release build pipeline completed successfully"
    print_status "Artifacts available in:"
    print_status "  - build/ (binaries)"
    print_status "  - dist/ (archives)"
    print_status "  - coverage.html (test coverage)"
}

# Development build with hot reload preparation
dev() {
    print_build "Development build with debugging symbols..."
    
    clean
    
    # Build with debug info for development
    export CGO_ENABLED=1
    go build \
        -ldflags="-X main.Version=${VERSION} -X main.BuildTime=${BUILD_TIME} -X main.GoVersion=${GO_VERSION}" \
        -gcflags="-N -l" \
        -race \
        -o "${BINARY_NAME}-dev" \
        main.go
    
    print_success "Development build completed: ${BINARY_NAME}-dev"
}

# Docker build optimization
docker() {
    print_build "Building optimized Docker image..."
    
    if [ ! -f Dockerfile ]; then
        print_status "Creating optimized Dockerfile..."
        cat > Dockerfile << 'EOF'
# Multi-stage build for minimal image size
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git upx

WORKDIR /app
COPY . .

# Build with maximum optimization
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -gcflags="-B -C" \
    -tags="netgo,osusergo" \
    -trimpath \
    -o wave2D main.go

# Compress binary
RUN upx --best --lzma wave2D

# Final minimal image
FROM scratch
COPY --from=builder /app/wave2D /wave2D
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
ENTRYPOINT ["/wave2D"]
EOF
    fi
    
    docker build -t "wave2d:${VERSION}" .
    print_success "Docker image built: wave2d:${VERSION}"
}

# Install binary to system
install() {
    if [ ! -f "${BINARY_NAME}" ]; then
        print_error "Binary not found. Build first."
        exit 1
    fi
    
    local install_dir="${HOME}/.local/bin"
    if [ "$EUID" -eq 0 ]; then
        install_dir="/usr/local/bin"
    fi
    
    mkdir -p "$install_dir"
    cp "${BINARY_NAME}" "$install_dir/"
    chmod +x "$install_dir/${BINARY_NAME}"
    
    print_success "Installed to $install_dir/${BINARY_NAME}"
}

# Show usage information
usage() {
    echo "Wave2D Advanced Build System"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  dev         Build for development with debug info"
    echo "  release     Complete release build pipeline (default)"
    echo "  native      Build optimized binary for current platform"
    echo "  cross       Cross-compile for multiple platforms"
    echo "  test        Run comprehensive tests with coverage"
    echo "  benchmark   Run performance benchmarks"
    echo "  clean       Clean build artifacts and caches"
    echo "  optimize    Optimize existing binary"
    echo "  performance Build with performance analysis"
    echo "  docker      Build optimized Docker image"
    echo "  install     Install binary to system"
    echo "  info        Show build information"
    echo "  help        Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  BINARY_NAME Custom binary name (default: wave2D)"
    echo "  BUILD_TAGS  Custom build tags"
    echo "  LDFLAGS     Additional linker flags"
    echo ""
    echo "Examples:"
    echo "  $0 release                    # Full release build"
    echo "  $0 native && $0 optimize      # Optimized native build"
    echo "  BUILD_TAGS=debug $0 dev       # Development build with debug"
}

# Main execution logic
main() {
    # Allow override of binary name
    BINARY_NAME="${BINARY_NAME:-wave2D}"
    
    # Add custom tags if provided
    if [ -n "${BUILD_TAGS:-}" ]; then
        TAGS="${TAGS},${BUILD_TAGS}"
    fi
    
    case "${1:-release}" in
        "dev"|"development")
            dev
            ;;
        "release")
            release
            ;;
        "native")
            check_environment
            clean
            build_native
            ;;
        "cross")
            check_environment
            clean
            build_cross
            generate_metadata
            ;;
        "test")
            check_environment
            test
            ;;
        "benchmark")
            check_environment
            benchmark
            ;;
        "clean")
            clean
            ;;
        "optimize")
            optimize
            ;;
        "performance"|"perf")
            check_environment
            performance
            ;;
        "docker")
            docker
            ;;
        "install")
            install
            ;;
        "info")
            info
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        *)
            print_error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
