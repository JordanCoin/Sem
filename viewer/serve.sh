#!/bin/bash
# Copy the latest .sem file and start a simple HTTP server
cp ~/.openclaw/workspace/Sem/generated/stargazer-memory.sem ~/.openclaw/workspace/Sem/viewer/
cd ~/.openclaw/workspace/Sem/viewer
echo "🌐 Starting server at http://localhost:8765"
echo "   Open this URL in your browser to see the mesh"
echo "   Press Ctrl+C to stop"
python3 -m http.server 8765
