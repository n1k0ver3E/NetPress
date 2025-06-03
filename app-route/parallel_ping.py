from mininet.log import output, error
from mininet.net import Mininet
from mininet.topo import SingleSwitchTopo
from mininet.log import setLogLevel, info
import re
from subprocess import PIPE

def parallelPing(net, timeout=1):
    """
    Parallel version of Mininet's pingAll. Sends one ping from each host to every other host in parallel.

    Args:
        net: Mininet object representing the network.
        timeout: Timeout for each ping in seconds (default is 1 second).

    Returns:
        pingall: A string summarizing the ping results.
        loss_percent: Packet loss percentage as an integer.
    """
    hosts = net.hosts
    pingall = ""  # to store ping results as a string
    tasks = []  # list to store (src, dst, popen) tuples
    
    # Launch ping processes for all host pairs
    for src in hosts:
        for dst in hosts:
            if src == dst:
                continue  # skip self-ping
            cmd = f"ping -c1 -W {timeout} {dst.IP()}"  # Add timeout to the ping command
            popen = src.popen(cmd, stdout=PIPE, stderr=PIPE)
            tasks.append((src, dst, popen))
    
    # Record success/failure of each ping
    success = {src: {} for src in hosts}
    total_sent = 0
    total_received = 0
    
    # Process each ping's result
    for (src, dst, proc) in tasks:
        out, err = proc.communicate()  # wait for ping to finish and get output
        output = out.decode('utf-8') if out else ''
        
        # Parse ping output to get sent/received counts
        if 'Network is unreachable' in output:
            sent, received = 1, 0  # unreachable network => 1 packet sent, 0 received
        else:
            match = re.search(r'(\d+) packets transmitted, (\d+) received', output)
            if match:
                sent = int(match.group(1))
                received = int(match.group(2))
            else:
                # If parsing fails, assume 1 packet sent and 0 received
                sent, received = 1, 0
        
        # Determine success (at least one packet received)
        success[src][dst] = (received >= 1)
        total_sent += sent
        total_received += received if received <= sent else sent  # cap received to sent
        
    # Prepare the result lines for each source host
    for src in hosts:
        pingall += f"{src.name} -> "
        for dst in hosts:
            if src == dst:
                continue
            pingall += f"{dst.name} " if success[src].get(dst, False) else "X "
        pingall += "\n"
    
    # Compute loss percentage
    lost = total_sent - total_received
    loss_percent = int((lost * 100.0) / total_sent) if total_sent > 0 else 0
    
    # Prepare the final result string
    pingall += f"*** Results: {loss_percent}% dropped ({total_received}/{total_sent} received)\n"
    
    # Return both pingall (string) and loss_percent (integer)
    return pingall, loss_percent


def test_parallelPing():
    # Set logging level to display info()
    setLogLevel('info')
    
    # Create a simple network topology with 4 hosts and 1 switch
    topo = SingleSwitchTopo(4)  # 4 hosts connected to a single switch
    net = Mininet(topo)
    
    # Start the network
    net.start()
    
    # Call parallelPing to test connectivity between all hosts
    info("*** Running parallel ping test...\n")
    packet_loss = parallelPing(net)  # Call the parallelPing function
    
    # Output the packet loss percentage
    info(f"*** Packet loss rate: {packet_loss}%\n")
    
    # Stop the network
    net.stop()

# Run the test
if __name__ == "__main__":
    test_parallelPing()