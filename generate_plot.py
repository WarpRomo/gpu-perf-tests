import sys
import argparse
import matplotlib.pyplot as plt
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="Path to CSV data file")
    parser.add_argument("output", help="Path to output PNG")
    parser.add_argument("--title", help="Chart Title", default="Benchmark Results")
    args = parser.parse_args()

    sizes = []
    cpu_times = []
    gpu_times = []

    # Read Data
    with open(args.datafile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            try:
                sizes.append(float(row[0]))
                cpu_val = float(row[1])
                gpu_val = float(row[2])
                
                # Handle error codes or skipped CPU runs
                cpu_times.append(cpu_val if cpu_val > 0 else None)
                gpu_times.append(gpu_val if gpu_val > 0 else None)
            except ValueError:
                continue

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Log-Log Plot
    # CPU: Blue Circles
    plt.loglog(sizes, cpu_times, 'b-o', label='CPU (Intel Xeon)', linewidth=2, markersize=5)
    # GPU: Green Squares (NVIDIA brand color)
    plt.loglog(sizes, gpu_times, 'g-s', label='GPU (NVIDIA T4)', linewidth=2, markersize=5)

    # Use the EXACT title passed from the shell script
    plt.title(args.title, fontsize=14, fontweight='bold')
    
    plt.xlabel("Input Size (N)", fontsize=12)
    plt.ylabel("Execution Time (microseconds)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    plt.annotate('Lower is Better', xy=(0.02, 0.95), xycoords='axes fraction', 
                 fontsize=10, fontweight='bold', color='gray')

    print(f"Generating graph: {args.output}")
    plt.savefig(args.output, dpi=300)

if __name__ == "__main__":
    main()
