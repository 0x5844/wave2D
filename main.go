package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

var (
	Version   = "dev"
	BuildTime = "unknown"
	GoVersion = "unknown"
)

// Vector2D represents a 2D vector with SIMD-friendly alignment
type Vector2D struct {
	X, Y float64
}

func NewVector2D(x, y float64) Vector2D {
	return Vector2D{X: x, Y: y}
}

func (v Vector2D) Add(other Vector2D) Vector2D {
	return Vector2D{X: v.X + other.X, Y: v.Y + other.Y}
}

func (v Vector2D) Sub(other Vector2D) Vector2D {
	return Vector2D{X: v.X - other.X, Y: v.Y - other.Y}
}

func (v Vector2D) Scale(factor float64) Vector2D {
	return Vector2D{X: v.X * factor, Y: v.Y * factor}
}

func (v Vector2D) Dot(other Vector2D) float64 {
	return v.X*other.X + v.Y*other.Y
}

func (v Vector2D) Length() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func (v Vector2D) Distance(other Vector2D) float64 {
	dx := v.X - other.X
	dy := v.Y - other.Y
	return math.Sqrt(dx*dx + dy*dy)
}

// CacheAlignedFloat64 ensures cache line alignment for better performance
type CacheAlignedFloat64 struct {
	value float64
	_     [56]byte // Padding to 64 bytes cache line
}

// WaveField represents a cache-optimized 2D field
type WaveField struct {
	Width, Height int
	data          []float64 // Linear array for cache efficiency
	stride        int       // Row stride for alignment
}

func NewWaveField(width, height int) *WaveField {
	// Add padding for cache alignment
	stride := ((width + 7) / 8) * 8 // Align to 8-element boundaries
	wf := &WaveField{
		Width:  width,
		Height: height,
		data:   make([]float64, stride*height),
		stride: stride,
	}
	return wf
}

func (wf *WaveField) Get(x, y int) float64 {
	if x < 0 || x >= wf.Width || y < 0 || y >= wf.Height {
		return 0
	}
	return wf.data[y*wf.stride+x]
}

func (wf *WaveField) Set(x, y int, value float64) {
	if x >= 0 && x < wf.Width && y >= 0 && y < wf.Height {
		wf.data[y*wf.stride+x] = value
	}
}

func (wf *WaveField) GetPtr(x, y int) *float64 {
	if x < 0 || x >= wf.Width || y < 0 || y >= wf.Height {
		return nil
	}
	return &wf.data[y*wf.stride+x]
}

func (wf *WaveField) Clear() {
	for i := range wf.data {
		wf.data[i] = 0
	}
}

func (wf *WaveField) Copy(other *WaveField) {
	if wf.Width == other.Width && wf.Height == other.Height {
		copy(wf.data, other.data)
	}
}

// MaterialGrid represents material properties with cache-optimized layout
type MaterialGrid struct {
	Width, Height int
	WaveSpeed     *WaveField
	Impedance     *WaveField
	Damping       *WaveField
	Density       *WaveField
}

func NewMaterialGrid(width, height int) *MaterialGrid {
	return &MaterialGrid{
		Width:     width,
		Height:    height,
		WaveSpeed: NewWaveField(width, height),
		Impedance: NewWaveField(width, height),
		Damping:   NewWaveField(width, height),
		Density:   NewWaveField(width, height),
	}
}

// BoundaryCondition types
type BoundaryCondition int

const (
	BoundaryReflecting BoundaryCondition = iota
	BoundaryAbsorbing
	BoundaryTransparent
	BoundaryFixed
	BoundaryPML // Perfectly Matched Layer
)

// WaveGrid with optimized data structures
type WaveGrid struct {
	Width, Height int
	CellSize      float64
	Origin        Vector2D

	// Time-stepping arrays with cache optimization
	Current   *WaveField
	Previous  *WaveField
	Next      *WaveField
	Auxiliary *WaveField // For higher-order schemes

	// Material properties
	Materials *MaterialGrid

	// Boundary conditions
	BoundaryTypes   [][]BoundaryCondition
	PMLCoefficients *WaveField // For PML boundaries

	// Thread-safe access
	rwMutex sync.RWMutex

	// Performance optimization
	updateMask    [][]bool // Only update active cells
	activeRegions []Region // Track active simulation regions
}

type Region struct {
	X1, Y1, X2, Y2 int
}

func NewWaveGrid(width, height int, cellSize float64, origin Vector2D) *WaveGrid {
	wg := &WaveGrid{
		Width:           width,
		Height:          height,
		CellSize:        cellSize,
		Origin:          origin,
		Current:         NewWaveField(width, height),
		Previous:        NewWaveField(width, height),
		Next:            NewWaveField(width, height),
		Auxiliary:       NewWaveField(width, height),
		Materials:       NewMaterialGrid(width, height),
		PMLCoefficients: NewWaveField(width, height),
		updateMask:      make([][]bool, width),
		activeRegions:   make([]Region, 0, 16),
	}

	// Initialize arrays
	wg.BoundaryTypes = make([][]BoundaryCondition, width)
	for i := range wg.BoundaryTypes {
		wg.BoundaryTypes[i] = make([]BoundaryCondition, height)
		wg.updateMask[i] = make([]bool, height)
	}

	wg.initializeDefaults()
	return wg
}

func (wg *WaveGrid) initializeDefaults() {
	// Default material properties
	defaultSpeed := 343.0     // Speed of sound in air (m/s)
	defaultImpedance := 413.0 // Air impedance
	defaultDamping := 0.999
	defaultDensity := 1.225 // Air density

	for i := 0; i < wg.Width; i++ {
		for j := 0; j < wg.Height; j++ {
			wg.Materials.WaveSpeed.Set(i, j, defaultSpeed)
			wg.Materials.Impedance.Set(i, j, defaultImpedance)
			wg.Materials.Damping.Set(i, j, defaultDamping)
			wg.Materials.Density.Set(i, j, defaultDensity)
			wg.updateMask[i][j] = true

			// Set boundary conditions
			if i == 0 || i == wg.Width-1 || j == 0 || j == wg.Height-1 {
				wg.BoundaryTypes[i][j] = BoundaryPML
				wg.setupPMLCoefficient(i, j)
			} else {
				wg.BoundaryTypes[i][j] = BoundaryTransparent
			}
		}
	}

	// Initialize with full grid as active region
	wg.activeRegions = append(wg.activeRegions, Region{0, 0, wg.Width, wg.Height})
}

func (wg *WaveGrid) setupPMLCoefficient(x, y int) {
	// PML absorption coefficient calculation
	pmlWidth := 10
	distanceFromBoundary := math.Min(
		math.Min(float64(x), float64(wg.Width-1-x)),
		math.Min(float64(y), float64(wg.Height-1-y)),
	)

	if distanceFromBoundary < float64(pmlWidth) {
		// Quadratic profile for PML
		ratio := (float64(pmlWidth) - distanceFromBoundary) / float64(pmlWidth)
		coefficient := ratio * ratio * 0.1
		wg.PMLCoefficients.Set(x, y, coefficient)
	}
}

func (wg *WaveGrid) WorldToGrid(pos Vector2D) (int, int) {
	x := int((pos.X - wg.Origin.X) / wg.CellSize)
	y := int((pos.Y - wg.Origin.Y) / wg.CellSize)
	return x, y
}

func (wg *WaveGrid) GridToWorld(x, y int) Vector2D {
	return Vector2D{
		X: wg.Origin.X + float64(x)*wg.CellSize,
		Y: wg.Origin.Y + float64(y)*wg.CellSize,
	}
}

func (wg *WaveGrid) IsValidIndex(x, y int) bool {
	return x >= 0 && x < wg.Width && y >= 0 && y < wg.Height
}

func (wg *WaveGrid) GetPressure(x, y int) float64 {
	wg.rwMutex.RLock()
	defer wg.rwMutex.RUnlock()
	return wg.Current.Get(x, y)
}

func (wg *WaveGrid) SetPressure(x, y int, value float64) {
	wg.rwMutex.Lock()
	defer wg.rwMutex.Unlock()
	wg.Current.Set(x, y, value)
}

// Advanced finite difference schemes
type AdvancedFDScheme struct {
	Order         int
	Coeffs        []float64
	StencilSize   int
	DispersionOpt bool
	Name          string
}

var (
	// Standard 2nd order scheme
	FD2Standard = AdvancedFDScheme{
		Order:       2,
		Coeffs:      []float64{1, -2, 1},
		StencilSize: 3,
		Name:        "Standard2",
	}

	// 4th order standard scheme
	FD4Standard = AdvancedFDScheme{
		Order:       4,
		Coeffs:      []float64{-1.0 / 12, 4.0 / 3, -5.0 / 2, 4.0 / 3, -1.0 / 12},
		StencilSize: 5,
		Name:        "Standard4",
	}

	// Optimized 4th order scheme (reduced dispersion)
	FD4Optimized = AdvancedFDScheme{
		Order:         4,
		Coeffs:        []float64{-0.07918, 1.3875, -2.6158, 1.3875, -0.07918},
		StencilSize:   5,
		DispersionOpt: true,
		Name:          "Optimized4",
	}

	// 6th order scheme for high accuracy
	FD6Standard = AdvancedFDScheme{
		Order:       6,
		Coeffs:      []float64{1.0 / 90, -3.0 / 20, 3.0 / 2, -49.0 / 18, 3.0 / 2, -3.0 / 20, 1.0 / 90},
		StencilSize: 7,
		Name:        "Standard6",
	}

	// Compact 4th order scheme
	FD4Compact = AdvancedFDScheme{
		Order:       4,
		Coeffs:      []float64{0.25, 1.5, -3.0, 1.5, 0.25},
		StencilSize: 5,
		Name:        "Compact4",
	}
)

// WaveSource with enhanced signal generation
type WaveSource struct {
	Position  Vector2D
	Frequency float64
	Amplitude float64
	Phase     float64
	StartTime float64
	Duration  float64
	WaveType  WaveType
	Bandwidth float64 // For filtered signals

	// Source characteristics
	DirectivityPattern []float64 // Angular directivity
	SourceRadius       float64   // Spatial extent

	// State management
	active   int32 // Atomic boolean
	id       uint64
	envelope EnvelopeType
}

type WaveType int
type EnvelopeType int

const (
	WaveSine WaveType = iota
	WaveSquare
	WaveTriangle
	WaveGaussianPulse
	WaveChirp
	WaveWhiteNoise
	WaveBandlimited
	WaveRicker // Ricker wavelet
)

const (
	EnvelopeNone EnvelopeType = iota
	EnvelopeGaussian
	EnvelopeHann
	EnvelopeExponential
)

func NewWaveSource(pos Vector2D, freq, amp float64, waveType WaveType) *WaveSource {
	return &WaveSource{
		Position:     pos,
		Frequency:    freq,
		Amplitude:    amp,
		Phase:        0,
		StartTime:    0,
		Duration:     math.Inf(1),
		WaveType:     waveType,
		Bandwidth:    freq * 0.1,
		SourceRadius: 0.1,
		active:       1,
		id:           uint64(rand.Int63()),
		envelope:     EnvelopeNone,
	}
}

func (ews *WaveSource) GenerateSignal(t float64) float64 {
	if atomic.LoadInt32(&ews.active) == 0 || t < ews.StartTime || t > ews.StartTime+ews.Duration {
		return 0
	}

	relativeTime := t - ews.StartTime
	phase := 2*math.Pi*ews.Frequency*relativeTime + ews.Phase

	var signal float64

	switch ews.WaveType {
	case WaveSine:
		signal = math.Sin(phase)
	case WaveSquare:
		signal = math.Copysign(1.0, math.Sin(phase))
	case WaveTriangle:
		normalized := math.Mod(phase/(2*math.Pi), 1.0)
		if normalized < 0.5 {
			signal = 4*normalized - 1
		} else {
			signal = 3 - 4*normalized
		}
	case WaveGaussianPulse:
		sigma := 1.0 / (2 * math.Pi * ews.Frequency)
		signal = math.Exp(-0.5*math.Pow(relativeTime/sigma, 2)) * math.Sin(phase)
	case WaveChirp:
		chirpRate := ews.Frequency * 0.1
		instantFreq := ews.Frequency + chirpRate*relativeTime
		instantPhase := math.Pi * instantFreq * relativeTime * relativeTime
		signal = math.Sin(instantPhase)
	case WaveRicker:
		// Ricker wavelet for seismic applications
		a := math.Pi * ews.Frequency * relativeTime
		signal = (1 - 2*a*a) * math.Exp(-a*a)
	case WaveWhiteNoise:
		signal = rand.Float64()*2 - 1
	case WaveBandlimited:
		// Simplified bandlimited signal
		signal = math.Sin(phase) + 0.3*math.Sin(3*phase) + 0.1*math.Sin(5*phase)
	default:
		signal = 0
	}

	// Apply envelope
	signal *= ews.applyEnvelope(relativeTime)

	return ews.Amplitude * signal
}

func (ews *WaveSource) applyEnvelope(t float64) float64 {
	switch ews.envelope {
	case EnvelopeGaussian:
		sigma := ews.Duration / 6.0
		center := ews.Duration / 2.0
		return math.Exp(-0.5 * math.Pow((t-center)/sigma, 2))
	case EnvelopeHann:
		if ews.Duration > 0 {
			return 0.5 * (1 - math.Cos(2*math.Pi*t/ews.Duration))
		}
		return 1.0
	case EnvelopeExponential:
		return math.Exp(-t / ews.Duration)
	default:
		return 1.0
	}
}

func (ews *WaveSource) SetActive(active bool) {
	if active {
		atomic.StoreInt32(&ews.active, 1)
	} else {
		atomic.StoreInt32(&ews.active, 0)
	}
}

func (ews *WaveSource) IsActive() bool {
	return atomic.LoadInt32(&ews.active) != 0
}

// WaveReceiver (formerly AdvancedWaveReceiver)
type WaveReceiver struct {
	Position Vector2D
	id       uint64

	// Recording with thread-safe circular buffer
	Recording    int32 // Atomic boolean
	RecordedData []float64
	SampleRate   float64
	writeIndex   int64 // Atomic
	readIndex    int64 // Atomic
	bufferSize   int

	// Signal processing
	Filter     *DigitalFilter
	NoiseLevel float64

	// Statistics
	maxAmplitude float64
	rmsAmplitude float64
	energySum    float64
	sampleCount  int64

	mutex sync.RWMutex
}

type DigitalFilter struct {
	Type        FilterType
	Cutoff      float64
	Order       int
	coeffs      []float64
	history     []float64
	historySize int
}

// Process applies the digital filter to the input value and returns the filtered output.
func (df *DigitalFilter) Process(input float64) float64 {
	if df == nil || df.Type == FilterNone {
		return input
	}

	// Simple first-order IIR filter example for demonstration
	if df.Order < 1 {
		df.Order = 1
	}
	if df.history == nil || len(df.history) < df.Order {
		df.history = make([]float64, df.Order)
	}

	alpha := 0.0
	switch df.Type {
	case FilterLowpass:
		// Simple RC lowpass: alpha = dt / (RC + dt)
		alpha = df.Cutoff
		if alpha <= 0 || alpha > 1 {
			alpha = 0.1
		}
		df.history[0] = alpha*input + (1-alpha)*df.history[0]
		return df.history[0]
	case FilterHighpass:
		alpha = df.Cutoff
		if alpha <= 0 || alpha > 1 {
			alpha = 0.1
		}
		df.history[0] = alpha * (df.history[0] + input - df.history[0])
		return input - df.history[0]
	case FilterBandpass:
		// Not implemented, just pass through for now
		return input
	default:
		return input
	}
}

type FilterType int

const (
	FilterNone FilterType = iota
	FilterLowpass
	FilterHighpass
	FilterBandpass
)

func NewWaveReceiver(pos Vector2D) *WaveReceiver {
	bufferSize := 1024 * 1024 // 1M samples buffer
	return &WaveReceiver{
		Position:     pos,
		id:           uint64(rand.Int63()),
		RecordedData: make([]float64, bufferSize),
		SampleRate:   44100,
		bufferSize:   bufferSize,
		NoiseLevel:   1e-10,
	}
}

func (awr *WaveReceiver) StartRecording() {
	atomic.StoreInt32(&awr.Recording, 1)
	atomic.StoreInt64(&awr.writeIndex, 0)
	atomic.StoreInt64(&awr.readIndex, 0)
	awr.sampleCount = 0
	awr.energySum = 0
	awr.maxAmplitude = 0
}

func (awr *WaveReceiver) StopRecording() {
	atomic.StoreInt32(&awr.Recording, 0)
}

func (awr *WaveReceiver) Record(pressure float64) {
	if atomic.LoadInt32(&awr.Recording) == 0 {
		return
	}

	// Apply filtering if configured
	if awr.Filter != nil {
		pressure = awr.Filter.Process(pressure)
	}

	// Update statistics
	absValue := math.Abs(pressure)
	if absValue > awr.maxAmplitude {
		awr.maxAmplitude = absValue
	}

	atomic.AddInt64(&awr.sampleCount, 1)
	awr.energySum += pressure * pressure

	// Store in circular buffer
	writeIdx := atomic.LoadInt64(&awr.writeIndex)
	awr.RecordedData[writeIdx%int64(awr.bufferSize)] = pressure
	atomic.AddInt64(&awr.writeIndex, 1)
}

func (awr *WaveReceiver) GetStatistics() (max, rms float64, samples int64) {
	awr.mutex.RLock()
	defer awr.mutex.RUnlock()

	samples = atomic.LoadInt64(&awr.sampleCount)
	max = awr.maxAmplitude
	if samples > 0 {
		rms = math.Sqrt(awr.energySum / float64(samples))
	}
	return
}

// WorkerPool (formerly AdvancedWorkerPool)
type WorkerPool struct {
	workers      int
	taskQueues   []chan TaskExecution
	dispatchChan chan TaskExecution
	quit         chan struct{}
	wg           sync.WaitGroup
	once         sync.Once

	// Load balancing
	workerLoads []int64 // Atomic counters
	nextWorker  int64   // Round-robin counter

	// Performance monitoring
	tasksProcessed int64
	totalTime      int64 // Nanoseconds
}

type TaskExecution struct {
	task     Task
	result   chan<- error
	priority int
	id       uint64
}

type Task struct {
	Execute       func() error
	ID            int
	Priority      int
	EstimatedTime time.Duration
}

func NewWorkerPool(workers int) *WorkerPool {
	wp := &WorkerPool{
		workers:      workers,
		taskQueues:   make([]chan TaskExecution, workers),
		dispatchChan: make(chan TaskExecution, workers*16),
		quit:         make(chan struct{}),
		workerLoads:  make([]int64, workers),
	}

	for i := range wp.taskQueues {
		wp.taskQueues[i] = make(chan TaskExecution, 64)
	}

	wp.start()
	return wp
}

func (wp *WorkerPool) start() {
	// Start dispatcher
	wp.wg.Add(1)
	go wp.dispatcher()

	// Start workers
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

func (wp *WorkerPool) dispatcher() {
	defer wp.wg.Done()

	for {
		select {
		case task := <-wp.dispatchChan:
			// Load balancing: find worker with minimum load
			minLoad := atomic.LoadInt64(&wp.workerLoads[0])
			selectedWorker := 0

			for i := 1; i < wp.workers; i++ {
				load := atomic.LoadInt64(&wp.workerLoads[i])
				if load < minLoad {
					minLoad = load
					selectedWorker = i
				}
			}

			// Increment selected worker's load
			atomic.AddInt64(&wp.workerLoads[selectedWorker], 1)

			select {
			case wp.taskQueues[selectedWorker] <- task:
			case <-wp.quit:
				return
			}

		case <-wp.quit:
			return
		}
	}
}

func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()

	for {
		select {
		case execution := <-wp.taskQueues[id]:
			start := time.Now()
			err := execution.task.Execute()
			duration := time.Since(start)

			// Update statistics
			atomic.AddInt64(&wp.tasksProcessed, 1)
			atomic.AddInt64(&wp.totalTime, duration.Nanoseconds())
			atomic.AddInt64(&wp.workerLoads[id], -1)

			select {
			case execution.result <- err:
			case <-wp.quit:
				return
			}

		case <-wp.quit:
			return
		}
	}
}

func (wp *WorkerPool) Submit(task Task, result chan<- error) {
	execution := TaskExecution{
		task:     task,
		result:   result,
		priority: task.Priority,
		id:       uint64(rand.Int63()),
	}

	select {
	case wp.dispatchChan <- execution:
	case <-wp.quit:
		result <- fmt.Errorf("worker pool closed")
	}
}

func (wp *WorkerPool) GetStatistics() (processed int64, avgTime time.Duration, workerLoads []int64) {
	processed = atomic.LoadInt64(&wp.tasksProcessed)
	totalTime := atomic.LoadInt64(&wp.totalTime)

	if processed > 0 {
		avgTime = time.Duration(totalTime / processed)
	}

	workerLoads = make([]int64, len(wp.workerLoads))
	for i := range wp.workerLoads {
		workerLoads[i] = atomic.LoadInt64(&wp.workerLoads[i])
	}

	return
}

func (wp *WorkerPool) Close() {
	wp.once.Do(func() {
		close(wp.quit)
		wp.wg.Wait()
	})
}

// ObjectPool (formerly AdvancedObjectPool)
type ObjectPool struct {
	slicePool   sync.Pool
	vectorPool  sync.Pool
	complexPool sync.Pool

	// Statistics
	allocations   int64
	deallocations int64
}

func NewObjectPool() *ObjectPool {
	return &ObjectPool{
		slicePool: sync.Pool{
			New: func() interface{} {
				return make([]float64, 0, 1024)
			},
		},
		vectorPool: sync.Pool{
			New: func() interface{} {
				return make([]Vector2D, 0, 512)
			},
		},
		complexPool: sync.Pool{
			New: func() interface{} {
				return make([]complex128, 0, 512)
			},
		},
	}
}

func (aop *ObjectPool) GetSlice() []float64 {
	atomic.AddInt64(&aop.allocations, 1)
	slice := aop.slicePool.Get().([]float64)
	return slice[:0]
}

func (aop *ObjectPool) PutSlice(slice []float64) {
	atomic.AddInt64(&aop.deallocations, 1)
	if cap(slice) <= 8192 { // Don't pool overly large slices
		aop.slicePool.Put(slice)
	}
}

func (aop *ObjectPool) GetVectorSlice() []Vector2D {
	atomic.AddInt64(&aop.allocations, 1)
	slice := aop.vectorPool.Get().([]Vector2D)
	return slice[:0]
}

func (aop *ObjectPool) PutVectorSlice(slice []Vector2D) {
	atomic.AddInt64(&aop.deallocations, 1)
	if cap(slice) <= 4096 {
		aop.vectorPool.Put(slice)
	}
}

// Main wave engine with all enhancements
type WaveEngine struct {
	grid      *WaveGrid
	sources   []*WaveSource
	receivers []*WaveReceiver

	// Simulation parameters
	timeStep    float64
	currentTime float64
	stepCount   int64
	maxTime     float64

	// Numerical schemes
	fdScheme       AdvancedFDScheme
	cflNumber      float64
	stabilityLimit float64

	// Performance optimizations
	workerPool *WorkerPool
	maxWorkers int
	objectPool *ObjectPool

	// Adaptive time stepping
	adaptiveTimeStep bool
	minTimeStep      float64
	maxTimeStep      float64

	// Statistics and monitoring
	stepCounter     int64
	sourceCounter   int64
	receiverCounter int64
	computationTime int64 // Nanoseconds
	lastStepTime    int64

	// Thread safety
	sourceMutex   sync.RWMutex
	receiverMutex sync.RWMutex
	statsMutex    sync.RWMutex

	// Performance tuning
	useSimdOptimizations bool
	prefetchDistance     int
	cacheBlockSize       int
}

func NewWaveEngine(gridWidth, gridHeight int, cellSize float64) *WaveEngine {
	maxWorkers := runtime.NumCPU()
	if maxWorkers > 16 {
		maxWorkers = 16 // Diminishing returns beyond 16 cores for this workload
	}

	we := &WaveEngine{
		grid:                 NewWaveGrid(gridWidth, gridHeight, cellSize, Vector2D{X: 0, Y: 0}),
		sources:              make([]*WaveSource, 0, 64),
		receivers:            make([]*WaveReceiver, 0, 64),
		currentTime:          0,
		fdScheme:             FD4Optimized,
		cflNumber:            0.4, // Conservative for stability
		workerPool:           NewWorkerPool(maxWorkers),
		maxWorkers:           maxWorkers,
		objectPool:           NewObjectPool(),
		adaptiveTimeStep:     true,
		prefetchDistance:     64,
		cacheBlockSize:       32,
		useSimdOptimizations: true,
	}

	// Calculate time step based on CFL condition
	maxSpeed := 343.0
	we.timeStep = we.cflNumber * cellSize / (maxSpeed * math.Sqrt(2))
	we.minTimeStep = we.timeStep * 0.1
	we.maxTimeStep = we.timeStep * 2.0
	we.stabilityLimit = cellSize / (maxSpeed * math.Sqrt(2))

	return we
}

func (we *WaveEngine) AddSource(source *WaveSource) {
	we.sourceMutex.Lock()
	we.sources = append(we.sources, source)
	we.sourceMutex.Unlock()
	atomic.AddInt64(&we.sourceCounter, 1)
}

func (we *WaveEngine) AddReceiver(receiver *WaveReceiver) {
	we.receiverMutex.Lock()
	we.receivers = append(we.receivers, receiver)
	we.receiverMutex.Unlock()
	atomic.AddInt64(&we.receiverCounter, 1)
}

func (we *WaveEngine) SetMaterial(x, y int, speed, impedance, damping, density float64) {
	if we.grid.IsValidIndex(x, y) {
		we.grid.Materials.WaveSpeed.Set(x, y, speed)
		we.grid.Materials.Impedance.Set(x, y, impedance)
		we.grid.Materials.Damping.Set(x, y, damping)
		we.grid.Materials.Density.Set(x, y, density)
	}
}

func (we *WaveEngine) SetBoundaryCondition(x, y int, boundary BoundaryCondition) {
	if we.grid.IsValidIndex(x, y) {
		we.grid.BoundaryTypes[x][y] = boundary
		if boundary == BoundaryPML {
			we.grid.setupPMLCoefficient(x, y)
		}
	}
}

func (we *WaveEngine) Step(ctx context.Context) error {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		atomic.AddInt64(&we.computationTime, duration.Nanoseconds())
		atomic.StoreInt64(&we.lastStepTime, duration.Nanoseconds())
	}()

	atomic.AddInt64(&we.stepCounter, 1)

	// Adaptive time stepping
	if we.adaptiveTimeStep {
		we.updateTimeStep()
	}

	// Apply sources
	if err := we.applySources(); err != nil {
		return err
	}

	// Compute wave equation using optimized finite differences
	if err := we.computeWaveEquationOptimized(ctx); err != nil {
		return err
	}

	// Apply boundary conditions
	if err := we.applyAdvancedBoundaryConditions(ctx); err != nil {
		return err
	}

	// Swap time levels with optimized memory operations
	we.swapTimeFieldsOptimized()

	// Update receivers
	we.updateReceivers()

	we.currentTime += we.timeStep

	return nil
}

func (we *WaveEngine) updateTimeStep() {
	// Simple adaptive time stepping based on maximum field values
	maxValue := 0.0
	for i := 0; i < we.grid.Width; i += 4 { // Sample grid
		for j := 0; j < we.grid.Height; j += 4 {
			value := math.Abs(we.grid.Current.Get(i, j))
			if value > maxValue {
				maxValue = value
			}
		}
	}

	// Adjust time step based on field magnitude
	if maxValue > 10.0 {
		we.timeStep = math.Max(we.timeStep*0.9, we.minTimeStep)
	} else if maxValue < 1.0 {
		we.timeStep = math.Min(we.timeStep*1.1, we.maxTimeStep)
	}

	// Ensure stability
	if we.timeStep > we.stabilityLimit {
		we.timeStep = we.stabilityLimit
	}
}

func (we *WaveEngine) applySources() error {
	we.sourceMutex.RLock()
	activeSources := make([]*WaveSource, 0, len(we.sources))
	for _, source := range we.sources {
		if source.IsActive() {
			activeSources = append(activeSources, source)
		}
	}
	we.sourceMutex.RUnlock()

	for _, source := range activeSources {
		x, y := we.grid.WorldToGrid(source.Position)
		if !we.grid.IsValidIndex(x, y) {
			continue
		}

		signal := source.GenerateSignal(we.currentTime)

		// Apply Gaussian spatial distribution with optimized radius
		radius := int(math.Ceil(source.SourceRadius / we.grid.CellSize))
		if radius < 1 {
			radius = 1
		}
		if radius > 5 {
			radius = 5
		}

		sigma := float64(radius) / 3.0
		normalization := 1.0 / (2 * math.Pi * sigma * sigma)

		for dx := -radius; dx <= radius; dx++ {
			for dy := -radius; dy <= radius; dy++ {
				nx, ny := x+dx, y+dy
				if !we.grid.IsValidIndex(nx, ny) {
					continue
				}

				distSq := float64(dx*dx + dy*dy)
				weight := normalization * math.Exp(-distSq/(2*sigma*sigma))

				currentValue := we.grid.Current.Get(nx, ny)
				we.grid.Current.Set(nx, ny, currentValue+signal*weight)
			}
		}
	}

	return nil
}

func (we *WaveEngine) computeWaveEquationOptimized(ctx context.Context) error {
	// Determine optimal chunk size for cache efficiency
	chunkSize := we.cacheBlockSize
	if chunkSize > we.grid.Width/we.maxWorkers {
		chunkSize = max(1, we.grid.Width/we.maxWorkers)
	}

	numChunks := (we.grid.Width + chunkSize - 1) / chunkSize
	results := make(chan error, numChunks)

	stencil := we.fdScheme.StencilSize / 2
	dtSq := we.timeStep * we.timeStep
	dxSq := we.grid.CellSize * we.grid.CellSize
	dySq := we.grid.CellSize * we.grid.CellSize

	for chunk := 0; chunk < numChunks; chunk++ {
		startX := chunk * chunkSize
		endX := min(startX+chunkSize, we.grid.Width)

		task := Task{
			ID: chunk,
			Execute: func() error {
				return we.computeWaveChunkOptimized(startX, endX, stencil, dtSq, dxSq, dySq)
			},
			Priority: 1,
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			we.workerPool.Submit(task, results)
		}
	}

	// Wait for all chunks to complete
	for i := 0; i < numChunks; i++ {
		select {
		case err := <-results:
			if err != nil {
				return err
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return nil
}

func (we *WaveEngine) computeWaveChunkOptimized(startX, endX, stencil int, dtSq, dxSq, dySq float64) error {
	coeffs := we.fdScheme.Coeffs

	for i := startX; i < endX; i++ {
		for j := stencil; j < we.grid.Height-stencil; j++ {
			if i < stencil || i >= we.grid.Width-stencil {
				continue
			}

			// Skip inactive regions
			if !we.grid.updateMask[i][j] {
				continue
			}

			// Skip boundary cells (handled separately)
			if we.grid.BoundaryTypes[i][j] != BoundaryTransparent {
				continue
			}

			// Get material properties at current point
			c := we.grid.Materials.WaveSpeed.Get(i, j)
			cSq := c * c
			damping := we.grid.Materials.Damping.Get(i, j)

			// Compute spatial derivatives using SIMD-style optimization
			var d2u_dx2, d2u_dy2 float64

			// Unrolled finite difference computation for better performance
			if we.fdScheme.StencilSize == 5 {
				// 4th order scheme - unrolled for performance
				u_m2_j := we.grid.Current.Get(i-2, j)
				u_m1_j := we.grid.Current.Get(i-1, j)
				u_00_j := we.grid.Current.Get(i, j)
				u_p1_j := we.grid.Current.Get(i+1, j)
				u_p2_j := we.grid.Current.Get(i+2, j)

				d2u_dx2 = (coeffs[0]*u_m2_j + coeffs[1]*u_m1_j + coeffs[2]*u_00_j +
					coeffs[3]*u_p1_j + coeffs[4]*u_p2_j) / dxSq

				u_i_m2 := we.grid.Current.Get(i, j-2)
				u_i_m1 := we.grid.Current.Get(i, j-1)
				u_i_00 := we.grid.Current.Get(i, j)
				u_i_p1 := we.grid.Current.Get(i, j+1)
				u_i_p2 := we.grid.Current.Get(i, j+2)

				d2u_dy2 = (coeffs[0]*u_i_m2 + coeffs[1]*u_i_m1 + coeffs[2]*u_i_00 +
					coeffs[3]*u_i_p1 + coeffs[4]*u_i_p2) / dySq
			} else {
				// General case with loop
				for k, coeff := range coeffs {
					idx := i - stencil + k
					idy := j - stencil + k

					if idx >= 0 && idx < we.grid.Width {
						d2u_dx2 += coeff * we.grid.Current.Get(idx, j)
					}
					if idy >= 0 && idy < we.grid.Height {
						d2u_dy2 += coeff * we.grid.Current.Get(i, idy)
					}
				}
				d2u_dx2 /= dxSq
				d2u_dy2 /= dySq
			}

			// Wave equation: u_tt = c²(u_xx + u_yy)
			laplacian := d2u_dx2 + d2u_dy2

			currentValue := we.grid.Current.Get(i, j)
			previousValue := we.grid.Previous.Get(i, j)

			nextValue := 2*currentValue - previousValue + cSq*dtSq*laplacian

			// Apply damping
			nextValue *= damping

			// Store result
			we.grid.Next.Set(i, j, nextValue)
		}
	}

	return nil
}

func (we *WaveEngine) applyAdvancedBoundaryConditions(ctx context.Context) error {
	// Parallel boundary condition application
	numTasks := 4 // Top, bottom, left, right boundaries
	results := make(chan error, numTasks)

	// Top and bottom boundaries
	task1 := Task{
		ID: 1,
		Execute: func() error {
			we.applyBoundaryHorizontal(0, we.grid.Height-1)
			return nil
		},
	}

	task2 := Task{
		ID: 2,
		Execute: func() error {
			we.applyBoundaryHorizontal(we.grid.Height-1, we.grid.Height-1)
			return nil
		},
	}

	// Left and right boundaries
	task3 := Task{
		ID: 3,
		Execute: func() error {
			we.applyBoundaryVertical(0, 0)
			return nil
		},
	}

	task4 := Task{
		ID: 4,
		Execute: func() error {
			we.applyBoundaryVertical(we.grid.Width-1, we.grid.Width-1)
			return nil
		},
	}

	// Submit tasks
	we.workerPool.Submit(task1, results)
	we.workerPool.Submit(task2, results)
	we.workerPool.Submit(task3, results)
	we.workerPool.Submit(task4, results)

	// Wait for completion
	for i := 0; i < numTasks; i++ {
		select {
		case err := <-results:
			if err != nil {
				return err
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return nil
}

func (we *WaveEngine) applyBoundaryHorizontal(y, endY int) {
	for j := y; j <= endY; j++ {
		for i := 0; i < we.grid.Width; i++ {
			we.applyBoundaryConditionAt(i, j)
		}
	}
}

func (we *WaveEngine) applyBoundaryVertical(x, endX int) {
	for i := x; i <= endX; i++ {
		for j := 0; j < we.grid.Height; j++ {
			we.applyBoundaryConditionAt(i, j)
		}
	}
}

func (we *WaveEngine) applyBoundaryConditionAt(i, j int) {
	switch we.grid.BoundaryTypes[i][j] {
	case BoundaryReflecting:
		we.applyReflectingBoundary(i, j)
	case BoundaryAbsorbing:
		we.applyAbsorbingBoundary(i, j)
	case BoundaryFixed:
		we.grid.Next.Set(i, j, 0)
	case BoundaryPML:
		we.applyPMLBoundary(i, j)
	case BoundaryTransparent:
		// Already handled in main computation
	}
}

func (we *WaveEngine) applyReflectingBoundary(i, j int) {
	// Perfect reflection with better mirror conditions
	var mirrorValue float64

	if i == 0 && we.grid.Width > 1 {
		mirrorValue = we.grid.Next.Get(1, j)
	} else if i == we.grid.Width-1 && we.grid.Width > 1 {
		mirrorValue = we.grid.Next.Get(we.grid.Width-2, j)
	} else if j == 0 && we.grid.Height > 1 {
		mirrorValue = we.grid.Next.Get(i, 1)
	} else if j == we.grid.Height-1 && we.grid.Height > 1 {
		mirrorValue = we.grid.Next.Get(i, we.grid.Height-2)
	}

	we.grid.Next.Set(i, j, mirrorValue)
}

func (we *WaveEngine) applyAbsorbingBoundary(i, j int) {
	// First-order absorbing boundary condition
	c := we.grid.Materials.WaveSpeed.Get(i, j)
	dt := we.timeStep
	dx := we.grid.CellSize

	absorptionCoeff := c * dt / dx
	if absorptionCoeff > 1.0 {
		absorptionCoeff = 1.0
	}

	currentValue := we.grid.Current.Get(i, j)
	previousValue := we.grid.Previous.Get(i, j)

	// Simple damping
	nextValue := currentValue*(2-absorptionCoeff) - previousValue*(1-absorptionCoeff)
	we.grid.Next.Set(i, j, nextValue)
}

func (we *WaveEngine) applyPMLBoundary(i, j int) {
	// Perfectly Matched Layer implementation
	sigma := we.grid.PMLCoefficients.Get(i, j)
	dt := we.timeStep

	currentValue := we.grid.Current.Get(i, j)
	previousValue := we.grid.Previous.Get(i, j)

	dampingFactor := math.Exp(-sigma * dt)
	nextValue := dampingFactor * (2*currentValue - previousValue)

	we.grid.Next.Set(i, j, nextValue)
}

func (we *WaveEngine) swapTimeFieldsOptimized() {
	// Efficient pointer swapping instead of copying data
	we.grid.rwMutex.Lock()
	defer we.grid.rwMutex.Unlock()

	// Rotate: previous <- current <- next <- previous
	temp := we.grid.Previous
	we.grid.Previous = we.grid.Current
	we.grid.Current = we.grid.Next
	we.grid.Next = temp

	// Clear the new "next" field efficiently
	we.grid.Next.Clear()
}

func (we *WaveEngine) updateReceivers() {
	we.receiverMutex.RLock()
	activeReceivers := make([]*WaveReceiver, len(we.receivers))
	copy(activeReceivers, we.receivers)
	we.receiverMutex.RUnlock()

	for _, receiver := range activeReceivers {
		x, y := we.grid.WorldToGrid(receiver.Position)
		if we.grid.IsValidIndex(x, y) {
			pressure := we.grid.GetPressure(x, y)
			receiver.Record(pressure)
		}
	}
}

func (we *WaveEngine) Run(ctx context.Context, duration float64) error {
	steps := int(duration / we.timeStep)
	if duration <= 0 {
		steps = math.MaxInt32 // Run indefinitely
	}

	for stepCount := 0; stepCount < steps; stepCount++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if err := we.Step(ctx); err != nil {
				return err
			}
		}

		// Optional: yield to scheduler periodically
		if stepCount%1000 == 0 {
			runtime.Gosched()
		}
	}

	return nil
}

func (we *WaveEngine) GetStats() (steps int64, sources int64, receivers int64, compTime time.Duration, avgStepTime time.Duration) {
	steps = atomic.LoadInt64(&we.stepCounter)
	sources = atomic.LoadInt64(&we.sourceCounter)
	receivers = atomic.LoadInt64(&we.receiverCounter)

	totalCompTimeNs := atomic.LoadInt64(&we.computationTime)
	compTime = time.Duration(totalCompTimeNs)

	if steps > 0 {
		avgStepTime = time.Duration(totalCompTimeNs / steps)
	}

	return
}

func (we *WaveEngine) GetDetailedStats() map[string]interface{} {
	steps, sources, receivers, compTime, avgStepTime := we.GetStats()
	processed, avgTaskTime, workerLoads := we.workerPool.GetStatistics()

	return map[string]interface{}{
		"simulation": map[string]interface{}{
			"steps":            steps,
			"sources":          sources,
			"receivers":        receivers,
			"current_time":     we.currentTime,
			"time_step":        we.timeStep,
			"computation_time": compTime.Seconds(),
			"avg_step_time":    avgStepTime.Nanoseconds(),
		},
		"performance": map[string]interface{}{
			"tasks_processed":  processed,
			"avg_task_time":    avgTaskTime.Nanoseconds(),
			"worker_loads":     workerLoads,
			"real_time_factor": we.currentTime / compTime.Seconds(),
		},
		"grid": map[string]interface{}{
			"width":       we.grid.Width,
			"height":      we.grid.Height,
			"cell_size":   we.grid.CellSize,
			"total_cells": we.grid.Width * we.grid.Height,
		},
		"numerical": map[string]interface{}{
			"fd_scheme":         we.fdScheme.Name,
			"fd_order":          we.fdScheme.Order,
			"cfl_number":        we.cflNumber,
			"adaptive_stepping": we.adaptiveTimeStep,
		},
	}
}

func (we *WaveEngine) Close() {
	we.workerPool.Close()
}

// Scene configuration (enhanced from previous version)
type WaveSceneConfig struct {
	Grid      GridConfig       `json:"grid"`
	Sources   []SourceConfig   `json:"sources"`
	Receivers []ReceiverConfig `json:"receivers"`
	Materials []MaterialRegion `json:"materials"`
	Duration  float64          `json:"duration"`
	Numerical NumericalConfig  `json:"numerical"`
}

type NumericalConfig struct {
	Scheme           string  `json:"scheme"`
	CFLNumber        float64 `json:"cfl_number"`
	AdaptiveTimeStep bool    `json:"adaptive_time_step"`
	MaxWorkers       int     `json:"max_workers"`
}

type GridConfig struct {
	Width    int      `json:"width"`
	Height   int      `json:"height"`
	CellSize float64  `json:"cell_size"`
	Origin   Vector2D `json:"origin"`
}

type SourceConfig struct {
	Position  Vector2D `json:"position"`
	Frequency float64  `json:"frequency"`
	Amplitude float64  `json:"amplitude"`
	WaveType  string   `json:"wave_type"`
	StartTime float64  `json:"start_time"`
	Duration  float64  `json:"duration"`
	Bandwidth float64  `json:"bandwidth"`
	Envelope  string   `json:"envelope"`
}

type ReceiverConfig struct {
	Position   Vector2D     `json:"position"`
	SampleRate float64      `json:"sample_rate"`
	Filter     FilterConfig `json:"filter"`
}

type FilterConfig struct {
	Type   string  `json:"type"`
	Cutoff float64 `json:"cutoff"`
	Order  int     `json:"order"`
}

type MaterialRegion struct {
	X1, Y1, X2, Y2 int     `json:"bounds"`
	WaveSpeed      float64 `json:"wave_speed"`
	Impedance      float64 `json:"impedance"`
	Damping        float64 `json:"damping"`
	Density        float64 `json:"density"`
	BoundaryType   string  `json:"boundary_type"`
}

// Scene loading and generation functions (enhanced)
func LoadWaveScene(filename string) (*WaveSceneConfig, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var config WaveSceneConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

func (we *WaveEngine) LoadScene(config *WaveSceneConfig) error {
	// Load numerical configuration
	if config.Numerical.Scheme != "" {
		switch config.Numerical.Scheme {
		case "FD2":
			we.fdScheme = FD2Standard
		case "FD4":
			we.fdScheme = FD4Standard
		case "FD4Opt":
			we.fdScheme = FD4Optimized
		case "FD6":
			we.fdScheme = FD6Standard
		case "FD4Compact":
			we.fdScheme = FD4Compact
		}
	}

	if config.Numerical.CFLNumber > 0 {
		we.cflNumber = config.Numerical.CFLNumber
		we.timeStep = we.cflNumber * we.grid.CellSize / (343.0 * math.Sqrt(2))
	}

	we.adaptiveTimeStep = config.Numerical.AdaptiveTimeStep

	// Load sources
	for _, srcConfig := range config.Sources {
		var waveType WaveType
		switch strings.ToLower(srcConfig.WaveType) {
		case "sine":
			waveType = WaveSine
		case "square":
			waveType = WaveSquare
		case "triangle":
			waveType = WaveTriangle
		case "gaussian":
			waveType = WaveGaussianPulse
		case "chirp":
			waveType = WaveChirp
		case "ricker":
			waveType = WaveRicker
		case "noise":
			waveType = WaveWhiteNoise
		case "bandlimited":
			waveType = WaveBandlimited
		default:
			waveType = WaveSine
		}

		source := NewWaveSource(srcConfig.Position, srcConfig.Frequency, srcConfig.Amplitude, waveType)
		source.StartTime = srcConfig.StartTime
		source.Duration = srcConfig.Duration
		source.Bandwidth = srcConfig.Bandwidth

		var envelope EnvelopeType
		switch strings.ToLower(srcConfig.Envelope) {
		case "gaussian":
			envelope = EnvelopeGaussian
		case "hann":
			envelope = EnvelopeHann
		case "exponential":
			envelope = EnvelopeExponential
		default:
			envelope = EnvelopeNone
		}
		source.envelope = envelope

		we.AddSource(source)
	}

	// Load receivers
	for _, recConfig := range config.Receivers {
		receiver := NewWaveReceiver(recConfig.Position)
		if recConfig.SampleRate > 0 {
			receiver.SampleRate = recConfig.SampleRate
		}
		we.AddReceiver(receiver)
	}

	// Load materials
	for _, matRegion := range config.Materials {
		var boundaryType BoundaryCondition
		switch strings.ToLower(matRegion.BoundaryType) {
		case "reflecting":
			boundaryType = BoundaryReflecting
		case "absorbing":
			boundaryType = BoundaryAbsorbing
		case "fixed":
			boundaryType = BoundaryFixed
		case "pml":
			boundaryType = BoundaryPML
		default:
			boundaryType = BoundaryTransparent
		}

		for i := matRegion.X1; i <= matRegion.X2; i++ {
			for j := matRegion.Y1; j <= matRegion.Y2; j++ {
				we.SetMaterial(i, j, matRegion.WaveSpeed, matRegion.Impedance,
					matRegion.Damping, matRegion.Density)
				we.SetBoundaryCondition(i, j, boundaryType)
			}
		}
	}

	return nil
}

// Enhanced scene generation functions
func (we *WaveEngine) GenerateAcousticRoom(roomWidth, roomHeight float64, wallMaterial string) {
	// Enhanced room with frequency-dependent materials
	var wallSpeed, wallImpedance, wallDamping, wallDensity float64

	switch strings.ToLower(wallMaterial) {
	case "concrete":
		wallSpeed = 4000.0
		wallImpedance = 1e7
		wallDamping = 0.99
		wallDensity = 2400.0
	case "wood":
		wallSpeed = 1200.0
		wallImpedance = 8e5
		wallDamping = 0.95
		wallDensity = 600.0
	case "carpet", "soft":
		wallSpeed = 300.0
		wallImpedance = 1e5
		wallDamping = 0.8
		wallDensity = 200.0
	case "glass":
		wallSpeed = 5000.0
		wallImpedance = 1.2e7
		wallDamping = 0.98
		wallDensity = 2500.0
	default: // "hard"
		wallSpeed = 1500.0
		wallImpedance = 8e6
		wallDamping = 0.95
		wallDensity = 1800.0
	}

	// Set wall boundaries with PML
	gridWidth := int(roomWidth / we.grid.CellSize)
	gridHeight := int(roomHeight / we.grid.CellSize)
	wallThickness := 5

	for i := 0; i < we.grid.Width; i++ {
		for j := 0; j < we.grid.Height; j++ {
			if i < wallThickness || i >= gridWidth-wallThickness ||
				j < wallThickness || j >= gridHeight-wallThickness {
				we.SetMaterial(i, j, wallSpeed, wallImpedance, wallDamping, wallDensity)

				// Use PML for outer boundaries, reflecting for inner
				if i < 2 || i >= we.grid.Width-2 || j < 2 || j >= we.grid.Height-2 {
					we.SetBoundaryCondition(i, j, BoundaryPML)
				} else {
					we.SetBoundaryCondition(i, j, BoundaryReflecting)
				}
			}
		}
	}
}

func (we *WaveEngine) GenerateInterferencePattern() {
	// Two coherent sources with precise phase control
	freq := 1000.0
	source1 := NewWaveSource(Vector2D{X: -1.0, Y: 0}, freq, 1.0, WaveSine)
	source2 := NewWaveSource(Vector2D{X: 1.0, Y: 0}, freq, 1.0, WaveSine)
	source2.Phase = math.Pi / 2 // 90 degree phase shift

	we.AddSource(source1)
	we.AddSource(source2)

	// Array of receivers for measuring interference pattern
	for i := -20; i <= 20; i++ {
		for j := -10; j <= 10; j++ {
			pos := Vector2D{X: float64(i) * 0.1, Y: float64(j) * 0.2}
			receiver := NewWaveReceiver(pos)
			receiver.StartRecording()
			we.AddReceiver(receiver)
		}
	}
}

func (we *WaveEngine) GenerateDiffractionDemo() {
	// Single source with obstacle for diffraction demonstration
	source := NewWaveSource(Vector2D{X: -2, Y: 0}, 2000, 1.0, WaveSine)
	we.AddSource(source)

	// Create obstacle
	obstacleX1, obstacleY1 := we.grid.WorldToGrid(Vector2D{X: -0.5, Y: -1.0})
	obstacleX2, obstacleY2 := we.grid.WorldToGrid(Vector2D{X: 0.5, Y: 1.0})

	for i := obstacleX1; i <= obstacleX2; i++ {
		for j := obstacleY1; j <= obstacleY2; j++ {
			if we.grid.IsValidIndex(i, j) {
				we.SetMaterial(i, j, 5000, 1e8, 0.9, 7800) // Steel-like properties
				we.SetBoundaryCondition(i, j, BoundaryReflecting)
			}
		}
	}

	// Receivers behind obstacle
	for i := 0; i < 20; i++ {
		pos := Vector2D{X: 2.0, Y: float64(i-10) * 0.1}
		receiver := NewWaveReceiver(pos)
		receiver.StartRecording()
		we.AddReceiver(receiver)
	}
}

// Configuration and main execution (enhanced)
type Config struct {
	GridWidth     int
	GridHeight    int
	CellSize      float64
	Duration      float64
	MaxFPS        int
	Workers       int
	SceneFile     string
	SceneType     string
	Frequency     float64
	Amplitude     float64
	Verbose       bool
	Quiet         bool
	StatsInterval float64
	ProfileCPU    string
	ProfileMem    string
	FDScheme      string
	CFLNumber     float64
	Adaptive      bool
	OutputFile    string
}

func parseFlags() *Config {
	config := &Config{}

	flag.IntVar(&config.GridWidth, "width", 400, "grid width")
	flag.IntVar(&config.GridHeight, "height", 400, "grid height")
	flag.Float64Var(&config.CellSize, "cell-size", 0.005, "cell size in meters")
	flag.Float64Var(&config.Duration, "duration", 0, "simulation duration in seconds (0 = infinite)")
	flag.IntVar(&config.MaxFPS, "fps", 0, "maximum frames per second (0 = unlimited)")
	flag.IntVar(&config.Workers, "workers", runtime.NumCPU(), "number of worker threads")
	flag.StringVar(&config.SceneFile, "scene", "", "JSON scene file to load")
	flag.StringVar(&config.SceneType, "scene-type", "interference", "scene type (interference, room, pulse, diffraction)")
	flag.Float64Var(&config.Frequency, "frequency", 1000, "source frequency in Hz")
	flag.Float64Var(&config.Amplitude, "amplitude", 1.0, "source amplitude")
	flag.BoolVar(&config.Verbose, "verbose", false, "verbose output")
	flag.BoolVar(&config.Quiet, "quiet", false, "minimal output")
	flag.Float64Var(&config.StatsInterval, "stats-interval", 2.0, "statistics reporting interval")
	flag.StringVar(&config.ProfileCPU, "profile-cpu", "", "CPU profile output file")
	flag.StringVar(&config.ProfileMem, "profile-mem", "", "memory profile output file")
	flag.StringVar(&config.FDScheme, "fd-scheme", "FD4Opt", "finite difference scheme (FD2, FD4, FD4Opt, FD6, FD4Compact)")
	flag.Float64Var(&config.CFLNumber, "cfl", 0.4, "CFL number for stability")
	flag.BoolVar(&config.Adaptive, "adaptive", true, "enable adaptive time stepping")
	flag.StringVar(&config.OutputFile, "output", "", "output file for results")

	var showVersion bool
	flag.BoolVar(&showVersion, "version", false, "show version information")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "AdvancedWave2D - High-Performance 2D Wave Simulation Engine\n\n")
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nFinite Difference Schemes:\n")
		fmt.Fprintf(os.Stderr, "  FD2       - 2nd order standard\n")
		fmt.Fprintf(os.Stderr, "  FD4       - 4th order standard\n")
		fmt.Fprintf(os.Stderr, "  FD4Opt    - 4th order optimized (default)\n")
		fmt.Fprintf(os.Stderr, "  FD6       - 6th order standard\n")
		fmt.Fprintf(os.Stderr, "  FD4Compact- 4th order compact\n")
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s -scene-type interference -frequency 1000 -fd-scheme FD6\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -scene scene.json -duration 10 -workers 8\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -scene-type diffraction -verbose -adaptive=false\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nVersion: %s\n", Version)
	}

	flag.Parse()

	if showVersion {
		fmt.Printf("AdvancedWave2D version %s\n", Version)
		fmt.Printf("Built: %s\n", BuildTime)
		fmt.Printf("Go: %s\n", GoVersion)
		os.Exit(0)
	}

	return config
}

func generateScene(engine *WaveEngine, config *Config) {
	switch config.SceneType {
	case "interference":
		engine.GenerateInterferencePattern()
	case "room":
		engine.GenerateAcousticRoom(4.0, 3.0, "hard")
		// Add a source in the room
		source := NewWaveSource(Vector2D{X: 1, Y: 1}, config.Frequency, config.Amplitude, WaveSine)
		engine.AddSource(source)
	case "pulse":
		source := NewWaveSource(Vector2D{X: 0, Y: 0}, config.Frequency, config.Amplitude, WaveGaussianPulse)
		source.Duration = 0.1 // Short pulse
		engine.AddSource(source)

		// Circular array of receivers
		for i := 0; i < 16; i++ {
			angle := float64(i) * 2 * math.Pi / 16
			pos := Vector2D{X: 2 * math.Cos(angle), Y: 2 * math.Sin(angle)}
			receiver := NewWaveReceiver(pos)
			receiver.StartRecording()
			engine.AddReceiver(receiver)
		}
	case "diffraction":
		engine.GenerateDiffractionDemo()
	default:
		log.Printf("Unknown scene type: %s, using interference", config.SceneType)
		engine.GenerateInterferencePattern()
	}
}

func reportAdvancedStats(ctx context.Context, engine *WaveEngine, interval float64, verbose bool) {
	ticker := time.NewTicker(time.Duration(interval * float64(time.Second)))
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if verbose {
				stats := engine.GetDetailedStats()
				simStats := stats["simulation"].(map[string]interface{})
				perfStats := stats["performance"].(map[string]interface{})

				log.Printf("=== Detailed Statistics ===")
				log.Printf("Simulation: Steps=%d, Time=%.3fs, TimeStep=%.2eµs",
					simStats["steps"], simStats["current_time"],
					simStats["time_step"].(float64)*1e6)
				log.Printf("Performance: RTF=%.2fx, Tasks=%d, AvgStepTime=%dµs",
					perfStats["real_time_factor"], perfStats["tasks_processed"],
					perfStats["avg_task_time"].(int64)/1000)
				log.Printf("Workers: %v", perfStats["worker_loads"])
			} else {
				steps, _, _, compTime, avgStepTime := engine.GetStats()
				rtf := engine.currentTime / compTime.Seconds()
				log.Printf("Steps: %d | Time: %.3fs | RTF: %.2fx | AvgStep: %dµs",
					steps, engine.currentTime, rtf, avgStepTime.Nanoseconds()/1000)
			}

		case <-ctx.Done():
			return
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	config := parseFlags()

	if config.Quiet {
		log.SetOutput(io.Discard)
	} else if config.Verbose {
		log.SetFlags(log.LstdFlags | log.Lshortfile)
	}

	if config.ProfileCPU != "" {
		f, err := os.Create(config.ProfileCPU)
		if err != nil {
			log.Fatal("Could not create CPU profile:", err)
		}
		defer f.Close()

		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("Could not start CPU profile:", err)
		}
		defer pprof.StopCPUProfile()
	}

	runtime.GOMAXPROCS(config.Workers)
	rand.Seed(time.Now().UnixNano())

	if !config.Quiet {
		log.Printf("Starting AdvancedWave2D Engine v%s", Version)
		log.Printf("Grid: %dx%d, Cell size: %.4fm", config.GridWidth, config.GridHeight, config.CellSize)
		log.Printf("CPU Cores: %d, Workers: %d", runtime.NumCPU(), config.Workers)
		log.Printf("FD Scheme: %s, CFL: %.3f, Adaptive: %v", config.FDScheme, config.CFLNumber, config.Adaptive)
	}

	engine := NewWaveEngine(config.GridWidth, config.GridHeight, config.CellSize)
	engine.maxWorkers = config.Workers
	engine.cflNumber = config.CFLNumber
	engine.adaptiveTimeStep = config.Adaptive

	// Set finite difference scheme
	switch config.FDScheme {
	case "FD2":
		engine.fdScheme = FD2Standard
	case "FD4":
		engine.fdScheme = FD4Standard
	case "FD4Opt":
		engine.fdScheme = FD4Optimized
	case "FD6":
		engine.fdScheme = FD6Standard
	case "FD4Compact":
		engine.fdScheme = FD4Compact
	default:
		log.Printf("Unknown FD scheme: %s, using FD4Opt", config.FDScheme)
		engine.fdScheme = FD4Optimized
	}

	// Recalculate time step with new CFL number
	maxSpeed := 343.0
	engine.timeStep = engine.cflNumber * config.CellSize / (maxSpeed * math.Sqrt(2))

	if config.SceneFile != "" {
		sceneConfig, err := LoadWaveScene(config.SceneFile)
		if err != nil {
			log.Fatalf("Failed to load scene: %v", err)
		}

		if err := engine.LoadScene(sceneConfig); err != nil {
			log.Fatalf("Failed to setup scene: %v", err)
		}

		if sceneConfig.Duration > 0 {
			config.Duration = sceneConfig.Duration
		}

		if !config.Quiet {
			log.Printf("Loaded scene from %s", config.SceneFile)
		}
	} else {
		generateScene(engine, config)
		if !config.Quiet {
			log.Printf("Generated %s scene", config.SceneType)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if config.Duration > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(config.Duration*float64(time.Second)))
		defer cancel()
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	if !config.Quiet {
		go reportAdvancedStats(ctx, engine, config.StatsInterval, config.Verbose)
	}

	go func() {
		select {
		case <-sigChan:
			if !config.Quiet {
				log.Println("Shutting down gracefully...")
			}
			cancel()
		case <-ctx.Done():
		}
	}()

	if !config.Quiet {
		log.Printf("Wave simulation started")
		if config.Duration > 0 {
			log.Printf("Simulation duration: %.2f seconds", config.Duration)
		} else {
			log.Println("Press Ctrl+C to stop")
		}
	}

	if err := engine.Run(ctx, config.Duration); err != nil && err != context.Canceled && err != context.DeadlineExceeded {
		log.Fatalf("Engine error: %v", err)
	}

	if config.ProfileMem != "" {
		f, err := os.Create(config.ProfileMem)
		if err != nil {
			log.Printf("Could not create memory profile: %v", err)
		} else {
			defer f.Close()
			runtime.GC()
			if err := pprof.WriteHeapProfile(f); err != nil {
				log.Printf("Could not write memory profile: %v", err)
			}
		}
	}

	if !config.Quiet {
		stats := engine.GetDetailedStats()
		simStats := stats["simulation"].(map[string]interface{})
		perfStats := stats["performance"].(map[string]interface{})
		numStats := stats["numerical"].(map[string]interface{})

		log.Printf("=== Final Simulation Report ===")
		log.Printf("  Steps: %d", simStats["steps"])
		log.Printf("  Sources: %d", simStats["sources"])
		log.Printf("  Receivers: %d", simStats["receivers"])
		log.Printf("  Simulated time: %.3fs", simStats["current_time"])
		log.Printf("  Computation time: %.3fs", simStats["computation_time"])
		log.Printf("  Real-time factor: %.2fx", perfStats["real_time_factor"])
		log.Printf("  Average step time: %dµs", simStats["avg_step_time"].(int64)/1000)
		log.Printf("  FD scheme: %s (order %d)", numStats["fd_scheme"], numStats["fd_order"])
		log.Printf("  Tasks processed: %d", perfStats["tasks_processed"])

		processed, avgTaskTime, _ := engine.workerPool.GetStatistics()
		if processed > 0 {
			log.Printf("  Worker efficiency: %.1f tasks/s", float64(processed)/simStats["computation_time"].(float64))
			log.Printf("  Average task time: %dµs", avgTaskTime.Nanoseconds()/1000)
		}
	}

	engine.Close()
}
