"""
JavaScript interface generation for WebAssembly photonic neural networks.
"""

import json
from typing import Dict, List, Any, Optional
import os
import textwrap


class JSPhotonNeuro:
    """JavaScript interface generator for PhotonNeuro WASM module."""
    
    def __init__(self, module_name: str = "PhotonNeuro"):
        self.module_name = module_name
        self.exports = {}
        self.type_definitions = {}
        
    def add_export(self, name: str, params: List[Dict[str, str]], 
                   return_type: str = "void", description: str = "") -> None:
        """
        Add a function export to the JavaScript interface.
        
        Args:
            name: Function name
            params: Parameter list with 'name' and 'type'
            return_type: Return type
            description: Function description
        """
        self.exports[name] = {
            'params': params,
            'return_type': return_type,
            'description': description
        }
    
    def generate_typescript_definitions(self) -> str:
        """Generate TypeScript definition file."""
        ts_content = textwrap.dedent(f"""
        /**
         * PhotonNeuro WebAssembly Module
         * Silicon-photonic neural network acceleration
         */
        
        declare module '{self.module_name}' {{
            export interface PhotonNeuroModule {{
                // Memory management
                memory: WebAssembly.Memory;
                HEAP8: Int8Array;
                HEAP16: Int16Array; 
                HEAP32: Int32Array;
                HEAPF32: Float32Array;
                HEAPF64: Float64Array;
                
                // Initialization
                initialize(options?: {{
                    simd?: boolean;
                    threads?: number;
                    memory?: number;
                }}): Promise<PhotonNeuroModule>;
                
                // Core photonic operations
                """)
        
        # Add exports
        for name, info in self.exports.items():
            params_str = ", ".join([f"{p['name']}: {p['type']}" for p in info['params']])
            ts_content += f"        {name}({params_str}): {info['return_type']};\n"
            if info['description']:
                ts_content += f"        /** {info['description']} */\n"
        
        ts_content += textwrap.dedent("""
                // Utility functions
                allocateComplex64Array(length: number): number;
                allocateFloat64Array(length: number): number;
                freeArray(ptr: number): void;
                
                // Performance monitoring
                getPerformanceMetrics(): {
                    simdEnabled: boolean;
                    threadCount: number;
                    memoryUsage: number;
                };
            }
            
            export default function(): Promise<PhotonNeuroModule>;
        }
        """)
        
        return ts_content
    
    def generate_javascript_wrapper(self) -> str:
        """Generate JavaScript wrapper with high-level API."""
        js_content = textwrap.dedent(f"""
        /**
         * PhotonNeuro JavaScript Wrapper
         * High-level interface for photonic neural network operations
         */
        
        class {self.module_name}Wrapper {{
            constructor(wasmModule) {{
                this.module = wasmModule;
                this.initialized = false;
                this.simdEnabled = false;
                this.threadCount = 1;
            }}
            
            async initialize(options = {{}}) {{
                const defaults = {{
                    simd: true,
                    threads: navigator.hardwareConcurrency || 1,
                    memory: 256 * 1024 * 1024 // 256MB
                }};
                
                const config = {{ ...defaults, ...options }};
                
                try {{
                    await this.module.initialize(config);
                    this.initialized = true;
                    this.simdEnabled = config.simd;
                    this.threadCount = config.threads;
                    
                    console.log(`{self.module_name} initialized:`, {{
                        simd: this.simdEnabled,
                        threads: this.threadCount,
                        memory: config.memory / (1024 * 1024) + 'MB'
                    }});
                    
                    return true;
                }} catch (error) {{
                    console.error('{self.module_name} initialization failed:', error);
                    return false;
                }}
            }}
            
            // High-level photonic network operations
            async createMZIMesh(size, topology = 'rectangular') {{
                if (!this.initialized) {{
                    throw new Error('{self.module_name} not initialized');
                }}
                
                const [rows, cols] = Array.isArray(size) ? size : [size, size];
                const numPhases = topology === 'rectangular' ? rows * cols : rows * (rows + 1) / 2;
                
                // Allocate memory for phase shifters
                const phasesPtr = this.module.allocateFloat64Array(numPhases);
                
                return {{
                    size: [rows, cols],
                    topology,
                    phasesPtr,
                    numPhases,
                    
                    setPhases(phases) {{
                        if (phases.length !== numPhases) {{
                            throw new Error(`Expected ${{numPhases}} phases, got ${{phases.length}}`);
                        }}
                        
                        const phasesArray = new Float64Array(this.module.HEAP8.buffer, phasesPtr, numPhases);
                        phasesArray.set(phases);
                    }},
                    
                    forward(inputField) {{
                        // Convert JavaScript array to WASM memory
                        const inputPtr = this._arrayToWasm(inputField, 'complex64');
                        const outputPtr = this.module.allocateComplex64Array(inputField.length);
                        
                        // Call WASM function
                        const result = this.module.mzi_forward_simd(inputPtr, phasesPtr, outputPtr, inputField.length);
                        
                        // Convert result back to JavaScript
                        const output = this._wasmToArray(outputPtr, inputField.length, 'complex64');
                        
                        // Clean up memory
                        this.module.freeArray(inputPtr);
                        this.module.freeArray(outputPtr);
                        
                        return output;
                    }},
                    
                    destroy() {{
                        this.module.freeArray(phasesPtr);
                    }}
                }};
            }}
            
            async createMicroringArray(numRings, fsr = 20e9, quality = 10000) {{
                if (!this.initialized) {{
                    throw new Error('{self.module_name} not initialized');
                }}
                
                // Allocate parameter arrays
                const radiiPtr = this.module.allocateFloat64Array(numRings);
                const couplingPtr = this.module.allocateFloat64Array(numRings);
                const qFactorsPtr = this.module.allocateFloat64Array(numRings);
                
                // Initialize with default values
                const radiiArray = new Float64Array(this.module.HEAP8.buffer, radiiPtr, numRings);
                const couplingArray = new Float64Array(this.module.HEAP8.buffer, couplingPtr, numRings);
                const qArray = new Float64Array(this.module.HEAP8.buffer, qFactorsPtr, numRings);
                
                // Set default ring parameters
                for (let i = 0; i < numRings; i++) {{
                    radiiArray[i] = 5e-6 + (Math.random() * 2e-6); // 5-7 μm radius
                    couplingArray[i] = 0.1 + (Math.random() * 0.2); // 0.1-0.3 coupling
                    qArray[i] = quality * (0.8 + Math.random() * 0.4); // ±20% Q variation
                }}
                
                return {{
                    numRings,
                    radiiPtr,
                    couplingPtr, 
                    qFactorsPtr,
                    
                    setParameters(radii, coupling, qFactors) {{
                        if (radii) radiiArray.set(radii);
                        if (coupling) couplingArray.set(coupling);
                        if (qFactors) qArray.set(qFactors);
                    }},
                    
                    simulate(wavelengths) {{
                        const wavelengthPtr = this._arrayToWasm(wavelengths, 'float64');
                        const outputPtr = this.module.allocateComplex64Array(numRings * wavelengths.length);
                        
                        // Call WASM simulation
                        this.module.microring_sim_simd(
                            wavelengthPtr, radiiPtr, couplingPtr, qFactorsPtr,
                            outputPtr, numRings, wavelengths.length
                        );
                        
                        // Convert result to 2D array
                        const result = this._wasmToArray(outputPtr, numRings * wavelengths.length, 'complex64');
                        const transmission = [];
                        for (let i = 0; i < numRings; i++) {{
                            transmission.push(result.slice(i * wavelengths.length, (i + 1) * wavelengths.length));
                        }}
                        
                        // Cleanup
                        this.module.freeArray(wavelengthPtr);
                        this.module.freeArray(outputPtr);
                        
                        return transmission;
                    }},
                    
                    destroy() {{
                        this.module.freeArray(radiiPtr);
                        this.module.freeArray(couplingPtr);
                        this.module.freeArray(qFactorsPtr);
                    }}
                }};
            }}
            
            // Utility functions
            _arrayToWasm(array, type) {{
                let ptr, TypedArray;
                
                switch (type) {{
                    case 'float64':
                        ptr = this.module.allocateFloat64Array(array.length);
                        TypedArray = Float64Array;
                        break;
                    case 'complex64':
                        ptr = this.module.allocateComplex64Array(array.length);
                        TypedArray = Float32Array; // Re/Im pairs
                        break;
                    default:
                        throw new Error(`Unsupported type: ${{type}}`);
                }}
                
                const wasmArray = new TypedArray(this.module.HEAP8.buffer, ptr, 
                                                type === 'complex64' ? array.length * 2 : array.length);
                
                if (type === 'complex64') {{
                    // Interleave real and imaginary parts
                    for (let i = 0; i < array.length; i++) {{
                        wasmArray[i * 2] = Array.isArray(array[i]) ? array[i][0] : array[i].real || array[i];
                        wasmArray[i * 2 + 1] = Array.isArray(array[i]) ? array[i][1] : array[i].imag || 0;
                    }}
                }} else {{
                    wasmArray.set(array);
                }}
                
                return ptr;
            }}
            
            _wasmToArray(ptr, length, type) {{
                let TypedArray, result;
                
                switch (type) {{
                    case 'float64':
                        TypedArray = Float64Array;
                        result = new TypedArray(this.module.HEAP8.buffer, ptr, length);
                        return Array.from(result);
                    case 'complex64':
                        TypedArray = Float32Array;
                        const complexArray = new TypedArray(this.module.HEAP8.buffer, ptr, length * 2);
                        result = [];
                        for (let i = 0; i < length; i++) {{
                            result.push({{
                                real: complexArray[i * 2],
                                imag: complexArray[i * 2 + 1]
                            }});
                        }}
                        return result;
                    default:
                        throw new Error(`Unsupported type: ${{type}}`);
                }}
            }}
            
            getPerformanceMetrics() {{
                if (!this.initialized) {{
                    return null;
                }}
                
                return this.module.getPerformanceMetrics();
            }}
            
            // Visualization helpers
            visualizeOpticalField(field, canvas) {{
                const ctx = canvas.getContext('2d');
                const width = canvas.width;
                const height = canvas.height;
                
                ctx.clearRect(0, 0, width, height);
                
                // Create intensity plot
                const imageData = ctx.createImageData(width, height);
                const data = imageData.data;
                
                for (let i = 0; i < field.length && i < width; i++) {{
                    const intensity = Math.sqrt(field[i].real * field[i].real + field[i].imag * field[i].imag);
                    const normalizedIntensity = Math.min(intensity / Math.max(...field.map(f => 
                        Math.sqrt(f.real * f.real + f.imag * f.imag))), 1);
                    
                    const colorValue = Math.floor(normalizedIntensity * 255);
                    
                    for (let j = 0; j < height; j++) {{
                        const pixelIndex = (j * width + i) * 4;
                        data[pixelIndex] = colorValue;     // R
                        data[pixelIndex + 1] = 0;          // G  
                        data[pixelIndex + 2] = 255 - colorValue; // B
                        data[pixelIndex + 3] = 255;        // A
                    }}
                }}
                
                ctx.putImageData(imageData, 0, 0);
            }}
        }}
        
        // Export factory function
        async function create{self.module_name}() {{
            const wasmModule = await import('./{self.module_name.lower()}.js');
            const instance = await wasmModule.default();
            return new {self.module_name}Wrapper(instance);
        }}
        
        export default create{self.module_name};
        export {{ {self.module_name}Wrapper }};
        """)
        
        return js_content
    
    def generate_browser_example(self) -> str:
        """Generate browser usage example."""
        html_content = textwrap.dedent(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.module_name} Browser Demo</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.1);
                    padding: 30px;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 30px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                .demo-section {{
                    margin: 20px 0;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}
                canvas {{
                    border: 2px solid rgba(255, 255, 255, 0.3);
                    border-radius: 5px;
                    background: black;
                }}
                button {{
                    background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                    border: none;
                    color: white;
                    padding: 12px 24px;
                    margin: 10px;
                    border-radius: 25px;
                    cursor: pointer;
                    font-size: 16px;
                    transition: all 0.3s ease;
                }}
                button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                }}
                .metrics {{
                    background: rgba(0, 0, 0, 0.3);
                    padding: 15px;
                    border-radius: 8px;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌟 {self.module_name} Browser Demo</h1>
                
                <div class="demo-section">
                    <h2>📊 Performance Metrics</h2>
                    <div id="metrics" class="metrics">
                        Initializing {self.module_name}...
                    </div>
                </div>
                
                <div class="demo-section">
                    <h2>🔬 MZI Mesh Simulation</h2>
                    <button onclick="runMZIDemo()">Run MZI Demo</button>
                    <button onclick="benchmarkMZI()">Benchmark Performance</button>
                    <br><br>
                    <canvas id="mziCanvas" width="800" height="400"></canvas>
                </div>
                
                <div class="demo-section">
                    <h2>💍 Microring Array Simulation</h2>
                    <button onclick="runMicroringDemo()">Run Microring Demo</button>
                    <button onclick="sweepWavelength()">Wavelength Sweep</button>
                    <br><br>
                    <canvas id="microringCanvas" width="800" height="400"></canvas>
                </div>
                
                <div class="demo-section">
                    <h2>📈 Real-time Performance</h2>
                    <div id="performance" class="metrics">
                        Click buttons above to see performance data...
                    </div>
                </div>
            </div>
            
            <script type="module">
                import create{self.module_name} from './photon-neuro-wrapper.js';
                
                let photonNeuro;
                let mziMesh;
                let microringArray;
                
                // Initialize {self.module_name}
                async function initialize() {{
                    try {{
                        photonNeuro = await create{self.module_name}();
                        const success = await photonNeuro.initialize({{
                            simd: true,
                            threads: navigator.hardwareConcurrency,
                            memory: 128 * 1024 * 1024 // 128MB
                        }});
                        
                        if (success) {{
                            updateMetrics();
                            setupDemoComponents();
                            console.log('{self.module_name} initialized successfully!');
                        }} else {{
                            throw new Error('Initialization failed');
                        }}
                    }} catch (error) {{
                        console.error('Failed to initialize {self.module_name}:', error);
                        document.getElementById('metrics').innerHTML = 
                            'Failed to initialize {self.module_name}: ' + error.message;
                    }}
                }}
                
                async function setupDemoComponents() {{
                    // Create 8x8 MZI mesh
                    mziMesh = await photonNeuro.createMZIMesh([8, 8], 'rectangular');
                    
                    // Create 16-ring microring array
                    microringArray = await photonNeuro.createMicroringArray(16, 20e9, 10000);
                }}
                
                function updateMetrics() {{
                    const metrics = photonNeuro.getPerformanceMetrics();
                    document.getElementById('metrics').innerHTML = 
                        `SIMD Enabled: ${{metrics.simdEnabled}}\\n` +
                        `Thread Count: ${{metrics.threadCount}}\\n` +
                        `Memory Usage: ${{(metrics.memoryUsage / (1024*1024)).toFixed(1)}} MB`;
                }}
                
                window.runMZIDemo = async function() {{
                    const startTime = performance.now();
                    
                    // Generate random input field
                    const inputField = [];
                    for (let i = 0; i < 8; i++) {{
                        inputField.push({{
                            real: Math.random() - 0.5,
                            imag: Math.random() - 0.5
                        }});
                    }}
                    
                    // Set random phases
                    const phases = Array.from({{ length: mziMesh.numPhases }}, () => Math.random() * 2 * Math.PI);
                    mziMesh.setPhases(phases);
                    
                    // Run simulation
                    const output = mziMesh.forward(inputField);
                    
                    const endTime = performance.now();
                    
                    // Visualize results
                    photonNeuro.visualizeOpticalField(output, document.getElementById('mziCanvas'));
                    
                    updatePerformanceDisplay('MZI Forward Pass', endTime - startTime, output.length);
                }};
                
                window.runMicroringDemo = async function() {{
                    const startTime = performance.now();
                    
                    // Generate wavelength range (1540-1560 nm)
                    const wavelengths = [];
                    for (let i = 0; i < 200; i++) {{
                        wavelengths.push(1540e-9 + (i / 199) * 20e-9);
                    }}
                    
                    // Run simulation
                    const transmission = microringArray.simulate(wavelengths);
                    
                    const endTime = performance.now();
                    
                    // Plot transmission spectra
                    plotTransmissionSpectra(transmission, wavelengths);
                    
                    updatePerformanceDisplay('Microring Simulation', endTime - startTime, 
                                           microringArray.numRings * wavelengths.length);
                }};
                
                window.benchmarkMZI = async function() {{
                    const sizes = [4, 8, 16, 32];
                    const results = [];
                    
                    for (const size of sizes) {{
                        const testMesh = await photonNeuro.createMZIMesh([size, size]);
                        const inputField = Array.from({{ length: size }}, () => ({{
                            real: Math.random() - 0.5,
                            imag: Math.random() - 0.5
                        }}));
                        
                        const phases = Array.from({{ length: testMesh.numPhases }}, () => Math.random() * 2 * Math.PI);
                        testMesh.setPhases(phases);
                        
                        const startTime = performance.now();
                        const runs = 100;
                        
                        for (let i = 0; i < runs; i++) {{
                            testMesh.forward(inputField);
                        }}
                        
                        const avgTime = (performance.now() - startTime) / runs;
                        results.push({{ size, time: avgTime }});
                        
                        testMesh.destroy();
                    }}
                    
                    // Display benchmark results
                    let benchmarkText = 'MZI Benchmark Results:\\n';
                    benchmarkText += 'Size\\tTime (ms)\\tThroughput\\n';
                    benchmarkText += '----\\t---------\\t----------\\n';
                    
                    for (const result of results) {{
                        const throughput = (result.size * result.size / result.time * 1000).toFixed(0);
                        benchmarkText += `${{result.size}}x${{result.size}}\\t${{result.time.toFixed(2)}}\\t\\t${{throughput}} ops/s\\n`;
                    }}
                    
                    document.getElementById('performance').innerHTML = benchmarkText;
                }};
                
                function plotTransmissionSpectra(transmission, wavelengths) {{
                    const canvas = document.getElementById('microringCanvas');
                    const ctx = canvas.getContext('2d');
                    const width = canvas.width;
                    const height = canvas.height;
                    
                    ctx.clearRect(0, 0, width, height);
                    
                    // Plot each ring's transmission
                    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8', '#f7dc6f'];
                    
                    transmission.slice(0, 8).forEach((spectrum, ringIndex) => {{
                        ctx.strokeStyle = colors[ringIndex % colors.length];
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        
                        spectrum.forEach((value, wIndex) => {{
                            const x = (wIndex / wavelengths.length) * width;
                            const intensity = Math.sqrt(value.real * value.real + value.imag * value.imag);
                            const y = height - (intensity * height * 0.8);
                            
                            if (wIndex === 0) {{
                                ctx.moveTo(x, y);
                            }} else {{
                                ctx.lineTo(x, y);
                            }}
                        }});
                        
                        ctx.stroke();
                    }});
                    
                    // Add wavelength axis labels
                    ctx.fillStyle = 'white';
                    ctx.font = '12px Arial';
                    ctx.fillText('1540 nm', 10, height - 10);
                    ctx.fillText('1560 nm', width - 60, height - 10);
                    ctx.fillText('Transmission', 10, 20);
                }}
                
                function updatePerformanceDisplay(operation, time, dataPoints) {{
                    const throughput = (dataPoints / time * 1000).toFixed(0);
                    const performanceText = 
                        `Last Operation: ${{operation}}\\n` +
                        `Execution Time: ${{time.toFixed(2)}} ms\\n` +
                        `Data Points: ${{dataPoints}}\\n` +
                        `Throughput: ${{throughput}} ops/second`;
                    
                    document.getElementById('performance').innerHTML = performanceText;
                }}
                
                // Initialize on page load
                initialize();
            </script>
        </body>
        </html>
        """)
        
        return html_content


def export_wasm_module(output_dir: str = "wasm_export", 
                      include_examples: bool = True) -> Dict[str, str]:
    """
    Export complete WASM module with JavaScript bindings.
    
    Args:
        output_dir: Output directory for generated files
        include_examples: Include example HTML file
    
    Returns:
        Dictionary of generated files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    js_interface = JSPhotonNeuro("PhotonNeuro")
    
    # Add standard photonic operations
    js_interface.add_export("mzi_forward_simd", [
        {"name": "inputPtr", "type": "number"},
        {"name": "phasesPtr", "type": "number"},
        {"name": "outputPtr", "type": "number"},
        {"name": "n_modes", "type": "number"}
    ], "number", "SIMD-accelerated MZI forward pass")
    
    js_interface.add_export("microring_sim_simd", [
        {"name": "wavelengthPtr", "type": "number"},
        {"name": "radiiPtr", "type": "number"},
        {"name": "couplingPtr", "type": "number"},
        {"name": "qPtr", "type": "number"},
        {"name": "outputPtr", "type": "number"},
        {"name": "n_rings", "type": "number"},
        {"name": "n_wavelengths", "type": "number"}
    ], "number", "SIMD-accelerated microring resonator simulation")
    
    generated_files = {}
    
    # Generate TypeScript definitions
    ts_file = os.path.join(output_dir, "photon-neuro.d.ts")
    with open(ts_file, 'w') as f:
        f.write(js_interface.generate_typescript_definitions())
    generated_files["typescript"] = ts_file
    
    # Generate JavaScript wrapper
    js_file = os.path.join(output_dir, "photon-neuro-wrapper.js")
    with open(js_file, 'w') as f:
        f.write(js_interface.generate_javascript_wrapper())
    generated_files["javascript"] = js_file
    
    # Generate browser example
    if include_examples:
        html_file = os.path.join(output_dir, "photon-neuro-demo.html")
        with open(html_file, 'w') as f:
            f.write(js_interface.generate_browser_example())
        generated_files["example"] = html_file
    
    # Generate package.json for npm distribution
    package_json = {
        "name": "photon-neuromorphics-wasm",
        "version": "0.2.0",
        "description": "WebAssembly acceleration for photonic neural networks",
        "main": "photon-neuro-wrapper.js",
        "types": "photon-neuro.d.ts",
        "files": [
            "*.js",
            "*.wasm", 
            "*.d.ts"
        ],
        "keywords": [
            "photonics",
            "neural-networks",
            "webassembly",
            "simd",
            "silicon-photonics"
        ],
        "author": "Daniel Schmidt",
        "license": "BSD-3-Clause",
        "repository": {
            "type": "git",
            "url": "https://github.com/danieleschmidt/Photon-Neuromorphics-SDK"
        }
    }
    
    package_file = os.path.join(output_dir, "package.json")
    with open(package_file, 'w') as f:
        json.dump(package_json, f, indent=2)
    generated_files["package"] = package_file
    
    return generated_files


if __name__ == "__main__":
    # Generate WASM export
    generated = export_wasm_module("../../../wasm_export")
    
    print("Generated WASM export files:")
    for file_type, path in generated.items():
        print(f"  {file_type}: {path}")