#%%
# # update_router_settings.py
# This file is used to set the settings of the MikroTik router to work with the OPX/Octave system
# To use, connect the computer directly to any of the router's ports (excluding port 1) and run the script.
# If the router's current state is unknown, please reset it to the default settings before running this script.
# There is a single parameter that can be changed in this file, `case = ` in line 21.
import os
import shlex
import subprocess
from datetime import datetime
from enum import Enum


class Options(Enum):
    new_router_22x_or_newer = 1  # Used to configure a router after a factory reset, to use with QOP 2.2.x or newer
    new_router_21x_or_older = 2  # Used to configure a router after a factory reset, to use with QOP 2.1.x or older
    upgrade_to_22x = 3  # Used to configure a router after an upgrade to QOP 2.2.x. To be used if the settings were not applied automatically
    downgrade_from_22x = 4  # Should only be used in rare cases with explicit instructions from QM
    change_to_static_dns = 5  # Should only be used in rare cases with explicit instructions from QM


case = Options.new_router_22x_or_newer
# Change the line above to be from one of the options above.


#################################
## DO NOT EDIT BELOW THIS LINE ##
#################################
if case == Options.new_router_22x_or_newer:
    commands = [
        '/ip pool remove 0',
        '/ip pool add name=dhcp ranges=192.168.88.101-192.168.88.254',
        '/ip dhcp-server remove 0',
        '/ip dhcp-server add address-pool=dhcp disabled=no interface=bridge name=defconf',
        '/lcd set default-screen=interfaces',
        '/ip service set www port=81',
        '/ip upnp set enabled=yes show-dummy-rule=yes allow-disable-external-interface=no',
        '/ip upnp interfaces add interface=ether1 type=external',
        '/ip upnp interfaces add interface=bridge type=internal',
        '/for x from 2 to 254 do={ :if ([/ip firewall nat find dst-port=[(10000+$x)]] = "") do={/ip firewall nat add chain=dstnat action=netmap protocol=tcp dst-port=[(10000+$x)] to-addresses=[:put ([/ip address get [/ip address find interface=bridge] network]|[:put ("0.0.0.".$x)])] to-ports=9510}}',
        '/for x from 2 to 254 do={ :if ([/ip firewall nat find dst-port=[(11000+$x)]] = "") do={/ip firewall nat add chain=dstnat action=netmap protocol=tcp dst-port=[(11000+$x)] to-addresses=[:put ([/ip address get [/ip address find interface=bridge] network]|[:put ("0.0.0.".$x)])] to-ports=80}}',
    ]
elif case == Options.new_router_21x_or_older:
    commands = [
        '/ip pool remove 0',
        '/ip pool add name=dhcp ranges=192.168.88.101-192.168.88.254',
        '/ip dhcp-server remove 0',
        '/ip dhcp-server add address-pool=dhcp disabled=no interface=bridge name=defconf',
        '/lcd set default-screen=interfaces',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=1883 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.10 to-ports=1883',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=80 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.10 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=81 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.11 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=82 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.12 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=50 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.50 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=51 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.51 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=52 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.52 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=53 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.53 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=54 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.54 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=55 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.55 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=56 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.56 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=57 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.57 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=58 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.58 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=59 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.59 to-ports=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=60 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.60 to-ports=80'
    ]
elif case == Options.upgrade_to_22x:
    commands = [
        '/ip service set www port=81',
        '/ip upnp set enabled=yes show-dummy-rule=yes allow-disable-external-interface=no',
        '/ip upnp interfaces add interface=ether1 type=external',
        '/ip upnp interfaces add interface=bridge type=internal',
        '/ip firewall nat remove [/ip firewall nat find chain=dstnat dst-port="80"]',
        '/ip firewall nat remove [/ip firewall nat find chain=dstnat dst-port="1883"]',
        '/for x from 2 to 254 do={ :if ([/ip firewall nat find dst-port=[(10000+$x)]] = "") do={/ip firewall nat add chain=dstnat action=netmap protocol=tcp dst-port=[(10000+$x)] to-addresses=[:put ([/ip address get [/ip address find interface=bridge] network]|[:put ("0.0.0.".$x)])] to-ports=9510}}',
        '/for x from 2 to 254 do={ :if ([/ip firewall nat find dst-port=[(11000+$x)]] = "") do={/ip firewall nat add chain=dstnat action=netmap protocol=tcp dst-port=[(11000+$x)] to-addresses=[:put ([/ip address get [/ip address find interface=bridge] network]|[:put ("0.0.0.".$x)])] to-ports=80}}',
    ]
elif case == Options.downgrade_from_22x:
    commands = [
        '/ip service set www port=80',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=1883 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.10 to-ports=1883',
        '/ip firewall nat add action=dst-nat chain=dstnat dst-port=80 in-interface-list=WAN protocol=tcp to-addresses=192.168.88.10 to-ports=80'
    ]
elif case == Options.change_to_static_dns:
    commands = [
        '/ip dns set allow-remote-requests=yes servers=1.1.1.1,8.8.8.8',
        '/ip dhcp-client set 0 use-peer-dns=no'
    ]
else:
    commands = []

args = shlex.split(f'ssh -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" -m hmac-sha1 admin@192.168.88.1')
logs = ""
print('Sending commands to the router...')
for i, cmd in enumerate(commands):
    print("Executing:", ' '.join(args + [f"{cmd}"]))
    res = subprocess.Popen(args + [f"{cmd}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    out = res[0].decode().replace('\r\n', '\n').removesuffix('\n')
    print("----Output----")
    print(out)
    err = res[1].decode().replace('\r\n', '\n').removesuffix('\n')
    if err != 0:
        #one of the commands failed - abort the script
        print("-----ERR------")
        print(err)
        print("------------")
        print("we got an error. Please check the command output and if necessary retry the script!")
    logs += "command: " + cmd + "\n"
    logs += "output: " + out + "\n"
    logs += "errors: " + err + "\n"
    logs += "\n"

    print(f"{100 * (i + 1) / len(commands)}%")

logs_path = rf'{os.getcwd()}\update_router_logs-{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
print(f'Saving logs to {logs_path}')
log_file = open(logs_path, 'w')
log_file.write(logs)
log_file.close()