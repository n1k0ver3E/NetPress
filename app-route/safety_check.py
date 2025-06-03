from mininet.log import lg

def safety_check(commands):
    if commands is None:
        return True
    if 'sudo' in commands:
        lg.output("Command containing 'sudo' is not allowed.")
        return False
    if 'tcpdump' in commands:
        lg.output("Command containing 'tcpdump' is not allowed.")
        return False
    if "systemctl" in commands:
        lg.output("Command containing 'systemctl' is not allowed.")
        return False
    if "frr" in commands:
        lg.output("Command containing 'frr' is not allowed.")
        return False
    if "ethtool" in commands:
        lg.output("Command containing 'ethtool' is not allowed.")
        return False
    if "ping" in commands:
        lg.output("Command containing 'ping' is not allowed.")
        return False
    return True