import matplotlib.pyplot as plt
import re
import os

def parse_log(file_path):
    events_by_source = {}
    
    pattern = re.compile(r"Source:\s+(\w+),\s+Type:\s+(\w+),\s+Start:\s+([-\d.]+)\s+ms,\s+End:\s+([-\d.]+)\s+ms")
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    source = match.group(1)
                    event_type = match.group(2)
                    start_time = float(match.group(3))
                    end_time = float(match.group(4))
                    duration = end_time - start_time
                    
                    if source not in events_by_source:
                        events_by_source[source] = {'COMMUNICATION': [], 'COMPUTE': []}
                    
                    if event_type in events_by_source[source]:
                        events_by_source[source][event_type].append((start_time, duration))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    return events_by_source

def plot_timeline(events_by_source):
    num_sources = len(events_by_source)
    if num_sources == 0:
        print("No events to plot")
        return
    
    fig, axes = plt.subplots(num_sources, 1, figsize=(15, 3 * num_sources), sharex=True)
    
    # Handle single source case
    if num_sources == 1:
        axes = [axes]
    
    y_positions = {'COMMUNICATION': 1, 'COMPUTE': 0}
    colors = {'COMMUNICATION': 'blue', 'COMPUTE': 'orange'}
    
    for idx, (source, events) in enumerate(sorted(events_by_source.items())):
        ax = axes[idx]
        
        for event_type, data in events.items():
            if data:
                ax.broken_barh(data, (y_positions[event_type] - 0.4, 0.8), 
                               facecolors=colors[event_type], label=event_type)
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Compute', 'Communication'])
        ax.set_ylabel(f'{source}')
        ax.set_title(f'{source} Timeline')
        ax.grid(True)
        
        handles = [plt.Rectangle((0,0),1,1, color=colors[t]) for t in ['COMMUNICATION', 'COMPUTE']]
        ax.legend(handles, ['Communication', 'Compute'])
    
    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Profiler Event Timeline by Source', fontsize=16, y=0.995)
    plt.tight_layout()

    output_path = './output_images/profiler_timeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Timeline saved to {output_path}")

if __name__ == "__main__":
    if os.path.exists('../logs/profiler.log'):
        log_path = '../logs/profiler.log'
    elif os.path.exists('logs/profiler.log'):
        log_path = 'logs/profiler.log'
    else:
        print("Error: profiler.log not found in expected locations.")
        exit(1)

    print(f"Reading log from: {log_path}")
    events_by_source = parse_log(log_path)
    
    if events_by_source:
        plot_timeline(events_by_source)
