import matplotlib.pyplot as plt
import re
import os

def parse_log(file_path):
    events = {'COMMUNICATION': [], 'COMPUTE': []}
    
    pattern = re.compile(r"Type:\s+(\w+),\s+Start:\s+([-\d.]+)\s+ms,\s+End:\s+([-\d.]+)\s+ms")
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    event_type = match.group(1)
                    start_time = float(match.group(2))
                    end_time = float(match.group(3))
                    duration = end_time - start_time
                    
                    if event_type in events:
                        events[event_type].append((start_time, duration))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    return events

def plot_timeline(events):
    fig, ax = plt.subplots(figsize=(15, 5))
    
    y_positions = {'COMMUNICATION': 1, 'COMPUTE': 0}
    colors = {'COMMUNICATION': 'blue', 'COMPUTE': 'orange'}
    
    for event_type, data in events.items():
        if data:
            ax.broken_barh(data, (y_positions[event_type] - 0.4, 0.8), 
                           facecolors=colors[event_type], label=event_type)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Compute', 'Communication'])
    ax.set_xlabel('Time (ms)')
    ax.set_title('Profiler Event Timeline')
    ax.grid(True)
    
    handles = [plt.Rectangle((0,0),1,1, color=colors[t]) for t in ['COMMUNICATION', 'COMPUTE']]
    ax.legend(handles, ['Communication', 'Compute'])

    output_path = './output_images/profiler_timeline.png'
    plt.savefig(output_path)
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
    events = parse_log(log_path)
    
    if events:
        plot_timeline(events)
