#!/bin/bash
# SmartTek Demo - Report Web Server

cd /home/daniel/Documents/drone-demo/data/outputs

echo "Starting HTTP server on port 8888..."
echo "Access from network devices at:"
echo ""

# Show all network IPs
ip -4 addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print "  http://"$2":8888/uavid3d/report/rapport_uavid3d.html"}' | sed 's/\/[0-9]*:/:/g'

echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m http.server 8888 --bind 0.0.0.0
