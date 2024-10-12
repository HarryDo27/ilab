#!/bin/bash
set -e

# Change directory to the script's location
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Start the server in the background
echo "Starting the server..."
python server.py &

# Sleep for 10 seconds to give the server time to start
sleep 10

# Loop to start clients in the background
for i in `seq 0 2`; do
    echo "Starting client $i"
    python client.py --client-id=${i} &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Wait for all background processes to complete
wait
