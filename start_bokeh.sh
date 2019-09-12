# Set a high data rate
export PATH=/root/.local/bin:$PATH
bokeh serve ./plot.py  --websocket-max-message-size=$((250*1024*1024)) --port=8888